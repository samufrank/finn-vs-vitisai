/*
 * finn_mlp_infer.c — board-side hot path for the FINN MLP MNIST deploys.
 *
 * Compiled as a shared library and loaded from benchmark.py via ctypes.
 * Python keeps ownership of: bitstream load (FINNExampleOverlay),
 * PYNQ buffer allocation, MMIO mapping for the AXI-DMA register blocks,
 * dataset loading, and JSON results. The C side does the per-inference
 * sequence (CPU MatMul + MultiThreshold, pack, DMA trigger + poll,
 * unpack, dequant + argmax) so Python overhead doesn't cap FPGA
 * throughput — the FINN-T C runner established this pattern.
 *
 * One shared library supports both INT8 (UINT8 act / INT24 obuf) and
 * INT4 (UINT4 act / INT16 obuf) deploys via runtime dispatch: init
 * wires `pack` and `unpack` function pointers and sizes from the
 * precision arg; the hot loop calls them with no per-sample branch.
 *
 * MultiThreshold semantics: act[c] = #{ j : thres[c, j] <= acc[c] }.
 * This is the `>=` / inclusive convention used by
 * qonnx.custom_op.general.multithreshold and matched by the patched
 * benchmark.py:multithreshold.
 *
 * Build on board (ARM64):
 *   gcc -O2 -shared -fPIC -Wall -o libfinn_mlp_infer.so finn_mlp_infer.c
 * Build on host (x86_64, for the correctness harness):
 *   same command; the ARM `dc cvac/civac` asm is gated behind
 *   `#ifdef __aarch64__` and becomes a no-op. The real inference path
 *   touches MMIO and will crash if called off-board; the harness only
 *   calls `finn_mlp_infer_one_mock` which skips DMA entirely.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>   /* clock_gettime for the one-shot profiled entry */

#define DMA_REG_ADDR_LO  (0x10 / 4)   /* 32-bit register-file index */
#define DMA_REG_ADDR_HI  (0x14 / 4)
#define DMA_REG_COUNT    (0x1C / 4)
#define DMA_REG_CTRL     (0x00 / 4)
#define DMA_AP_DONE_BIT  0x2

#define CACHE_LINE_BYTES 64

/* ---- ARMv8 cache maintenance (unprivileged on PYNQ's Linux, UCI=1) ----
 * Gated so the same .c compiles on x86 for the host-side harness. On
 * non-ARM64 these are no-ops; real inference wouldn't work off-board
 * anyway because `trigger_dma` writes to MMIO. */

static inline void dcache_clean(const void *addr, size_t size)
{
#if defined(__aarch64__)
    uintptr_t start = (uintptr_t)addr & ~((uintptr_t)CACHE_LINE_BYTES - 1);
    uintptr_t end = (uintptr_t)addr + size;
    for (uintptr_t a = start; a < end; a += CACHE_LINE_BYTES) {
        __asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)addr; (void)size;
#endif
}

static inline void dcache_invalidate(const void *addr, size_t size)
{
#if defined(__aarch64__)
    /* DC CIVAC = clean+invalidate; unprivileged on Linux ARM64. obuf is
     * CPU-read-only, so the clean is a no-op. DC IVAC alone is privileged. */
    uintptr_t start = (uintptr_t)addr & ~((uintptr_t)CACHE_LINE_BYTES - 1);
    uintptr_t end = (uintptr_t)addr + size;
    for (uintptr_t a = start; a < end; a += CACHE_LINE_BYTES) {
        __asm__ volatile("dc civac, %0" :: "r"(a) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)addr; (void)size;
#endif
}

static inline void mmio_write(volatile uint32_t *base, unsigned idx, uint32_t v)
{
    base[idx] = v;
}

static inline uint32_t mmio_read(volatile uint32_t *base, unsigned idx)
{
    return base[idx];
}

/* ============================================================
 * Pack / unpack — one pair per FINN dtype, exposed as extern so
 * the host harness can call them directly via ctypes.
 *
 * Both MLP deploys use PE=SIMD=1, i.e. innermost packed dim = 1
 * padded to a byte. Conventions verified byte-identical against
 * finn.util.data_packing.finnpy_to_packed_bytearray with
 * reverse_endian=True, reverse_inner=True:
 *
 *   UINT8 packed (1, N, 1):  ibuf[i] = act[i]           (memcpy)
 *   UINT4 packed (1, N, 1):  ibuf[i] = act[i] & 0x0F    (low nibble)
 *   INT24 packed (1, K, 3):  out[i] = LE 3-byte -> sign-extend bit 23
 *   INT16 packed (1, K, 2):  out[i] = LE 2-byte int16   -> widen to int32
 * ============================================================ */

void finn_mlp_pack_uint8(const uint8_t *act, uint8_t *ibuf, int n)
{
    memcpy(ibuf, act, (size_t)n);
}

void finn_mlp_pack_uint4(const uint8_t *act, uint8_t *ibuf, int n)
{
    /* 1-per-byte: each act element occupies the low nibble of its own ibuf byte.
     * Used when FINN folds with PE=SIMD=1 (baseline MLP INT4 deploy). */
    for (int i = 0; i < n; i++) ibuf[i] = (uint8_t)(act[i] & 0x0Fu);
}

void finn_mlp_pack_uint4_2perbyte(const uint8_t *act, uint8_t *ibuf, int n)
{
    /* 2-per-byte: low nibble = even-index element, high nibble = odd-index.
     * Used when FINN folds with SIMD>1 such that the streamed AXI word packs
     * multiple INT4 elements (verified with mlp_int4_fps500000 build at
     * MVAU SIMD=16: ishape_packed=(1,4,8) for mid_dim=64).
     *
     * Identical implementation to finn_cnn_pack_uint4 — the CNN baseline
     * already uses this 2-per-byte convention because FMPadding feeds 8 INT4
     * channels in parallel. The MLP needed both 1-per-byte (PE=SIMD=1) and
     * 2-per-byte (high SIMD) paths once the target_fps sweep landed on
     * SIMD=16 INT4 builds. select_dispatch picks based on caller-supplied
     * ibuf_bytes vs mid_dim ratio.
     *
     * Caller guarantees n even (mid_dim is even for both observed MLP
     * configs; runner_init enforces it for the 2-per-byte path). */
    int half = n >> 1;
    for (int i = 0; i < half; i++) {
        uint8_t lo = act[2*i]     & 0x0Fu;
        uint8_t hi = act[2*i + 1] & 0x0Fu;
        ibuf[i] = (uint8_t)(lo | (hi << 4));
    }
}

void finn_mlp_unpack_int24_le(const uint8_t *obuf, int32_t *out, int n)
{
    for (int i = 0; i < n; i++) {
        const uint8_t *b = obuf + (size_t)i * 3;
        int32_t v = (int32_t)b[0]
                  | ((int32_t)b[1] << 8)
                  | ((int32_t)b[2] << 16);
        if (v & 0x800000) v |= (int32_t)0xFF000000;   /* sign-extend bit 23 */
        out[i] = v;
    }
}

void finn_mlp_unpack_int16_le(const uint8_t *obuf, int32_t *out, int n)
{
    for (int i = 0; i < n; i++) {
        const uint8_t *b = obuf + (size_t)i * 2;
        int16_t v = (int16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8));
        out[i] = (int32_t)v;
    }
}

/* ============================================================
 * Runner state (single static instance, matching finn_t_infer).
 * Set by finn_mlp_runner_init, cleared by finn_mlp_runner_destroy.
 * ============================================================ */

typedef void (*pack_fn_t)  (const uint8_t *act, uint8_t *ibuf, int n);
typedef void (*unpack_fn_t)(const uint8_t *obuf, int32_t *out, int n);

typedef struct {
    /* Caller-owned buffers + MMIO. Slot 0 is set by runner_init; slot 1
     * is optionally set by finn_mlp_set_second_buffers to enable batch
     * overlap of CPU prep[N+1] with FPGA accel[N]. When unset, slot 1
     * mirrors slot 0 and n_buffers stays 1 (single-buffer lockstep). */
    void     *ibuf_virt[2]; uint64_t ibuf_phys[2];
    void     *obuf_virt[2]; uint64_t obuf_phys[2];
    int       n_buffers;       /* 1 = single-buffered, 2 = double */
    void     *idma_mmio;   void     *odma_mmio;

    int       in_dim;
    int       mid_dim;
    int       num_classes;
    int       num_thresholds;
    int       ibuf_bytes;
    int       obuf_bytes;

    pack_fn_t   pack;
    unpack_fn_t unpack;

    const float *W0;
    const float *thres;
    float        mul;
    const float *add;

    /* Partition layout. 0 = classic (Linear1 on CPU: MatMul + per-channel
     * MultiThreshold). 1 = qi (input QuantIdentity moved Linear1 onto FPGA;
     * CPU only does input MultiThreshold on raw image). For partition=1
     * W0 may be NULL, mid_dim must equal in_dim, and `thres` is shape
     * (1, num_thresholds) — one row reused across all input pixels. */
    int       partition;
    /* Signedness of FPGA-input dtype (see CNN runner for full rationale).
     * 0 = UINT (classic MLP: post-MatMul+MT uint output). 1 = INT (QI with
     * QuantIdentity(Int8...): signed int8). cpu_pre_qi shifts the count
     * by 2^(precision-1) when signed. Classic must be 0. */
    int       idt_signed;
    /* QI affine input-quant scale, derived from the input threshold table at
     * init: (thres[T-1] - thres[0]) / (T-1). Reproduces the 255-entry
     * MultiThreshold scan as q = clip(round_half_up(v/qi_scale), -128, 127)
     * for the signed-INT8 QI input. Only set when partition==1; 0 otherwise. */
    double    qi_scale;
    int       use_cache_ops;
    int       initialized;
} mlp_runner_state_t;

static mlp_runner_state_t g_mlp = {0};

/* Map (precision, caller-supplied ibuf_bytes) → {pack, unpack, output bytes/elem}.
 *
 * ibuf_bytes is the actual packed-input byte count from FINN's driver
 * (prod(ishape_packed[1:])). It captures the SIMD-driven layout choice that
 * the precision alone doesn't determine. Pack-function dispatch:
 *   precision=8: ibuf_bytes must equal mid_dim                 → pack_uint8
 *   precision=4 + ibuf_bytes == mid_dim:     1-per-byte INT4   → pack_uint4
 *   precision=4 + ibuf_bytes == mid_dim/2:   2-per-byte INT4   → pack_uint4_2perbyte
 *   else: -1 (unsupported layout)
 * Output dtype is precision-determined (INT24 vs INT16); no SIMD effect.
 */
static int select_dispatch(int precision,
                           int ibuf_bytes,
                           int mid_dim,
                           pack_fn_t   *pack_out,
                           unpack_fn_t *unpack_out,
                           int *out_bytes_per_elem)
{
    switch (precision) {
        case 8:
            if (ibuf_bytes != mid_dim) return -1;
            *pack_out            = finn_mlp_pack_uint8;
            *unpack_out          = finn_mlp_unpack_int24_le;
            *out_bytes_per_elem  = 3;   /* INT24 */
            return 0;
        case 4:
            *unpack_out          = finn_mlp_unpack_int16_le;
            *out_bytes_per_elem  = 2;   /* INT16 */
            if (ibuf_bytes == mid_dim) {
                *pack_out = finn_mlp_pack_uint4;            /* 1-per-byte */
                return 0;
            }
            if (ibuf_bytes * 2 == mid_dim) {
                if (mid_dim & 1) return -1;                 /* 2-per-byte needs even */
                *pack_out = finn_mlp_pack_uint4_2perbyte;
                return 0;
            }
            return -1;
        default:
            return -1;
    }
}

int finn_mlp_runner_init(
    int       precision,          /* 8 or 4 */
    int       in_dim,
    int       mid_dim,
    int       num_classes,
    int       num_thresholds,
    int       ibuf_bytes,         /* caller-supplied; matches FINN's ishape_packed.
                                   * Mirrors the CNN runner's bytes-per-pixel
                                   * pattern: caller knows the FINN-chosen layout,
                                   * runner doesn't reverse-engineer it. */
    int       use_cache_ops,
    void     *ibuf_virt, uint64_t ibuf_phys,
    void     *obuf_virt, uint64_t obuf_phys,
    void     *idma_mmio, void   *odma_mmio,
    const float *W0,
    const float *thres,
    float        mul,
    const float *add,
    int          partition,         /* 0 = classic, 1 = qi */
    int          idt_signed)        /* 0 = unsigned IDT, 1 = signed (QI only) */
{
    if (in_dim <= 0 || mid_dim <= 0 || num_classes <= 0 || num_thresholds <= 0)
        return -2;
    if (ibuf_bytes <= 0)                         return -2;
    if (ibuf_virt == NULL || obuf_virt == NULL)  return -3;
    if (partition != 0 && partition != 1)        return -7;
    if (idt_signed != 0 && idt_signed != 1)      return -9;
    if (partition == 0 && idt_signed != 0)       return -9;
    /* W0 unused on the QI path; thres + add still required. */
    if (thres == NULL || add == NULL)            return -4;
    if (partition == 0 && W0 == NULL)            return -4;
    /* QI: FPGA accepts the post-MT raw image, so mid_dim must match in_dim. */
    if (partition == 1 && mid_dim != in_dim)     return -8;

    pack_fn_t   pack;
    unpack_fn_t unpack;
    int         out_be;
    if (select_dispatch(precision, ibuf_bytes, mid_dim,
                        &pack, &unpack, &out_be) != 0)
        return -5;

    /* First-row sort spot check (FINN emits ascending thresholds). Catches
     * the most common misuse; full check over all mid_dim rows is overkill. */
    for (int j = 1; j < num_thresholds; j++) {
        if (thres[j] < thres[j - 1]) return -6;
    }

    g_mlp.ibuf_virt[0]   = ibuf_virt;
    g_mlp.ibuf_phys[0]   = ibuf_phys;
    g_mlp.obuf_virt[0]   = obuf_virt;
    g_mlp.obuf_phys[0]   = obuf_phys;
    g_mlp.ibuf_virt[1]   = ibuf_virt;
    g_mlp.ibuf_phys[1]   = ibuf_phys;
    g_mlp.obuf_virt[1]   = obuf_virt;
    g_mlp.obuf_phys[1]   = obuf_phys;
    g_mlp.n_buffers      = 1;
    g_mlp.idma_mmio      = idma_mmio;
    g_mlp.odma_mmio      = odma_mmio;
    g_mlp.in_dim         = in_dim;
    g_mlp.mid_dim        = mid_dim;
    g_mlp.num_classes    = num_classes;
    g_mlp.num_thresholds = num_thresholds;
    g_mlp.ibuf_bytes     = ibuf_bytes;
    g_mlp.obuf_bytes     = num_classes * out_be;
    g_mlp.pack           = pack;
    g_mlp.unpack         = unpack;
    g_mlp.W0             = W0;
    g_mlp.thres          = thres;
    g_mlp.mul            = mul;
    g_mlp.add            = add;
    g_mlp.partition      = partition;
    g_mlp.idt_signed     = idt_signed;
    /* Derive the affine input-quant scale from the loaded table (QI only).
     * Endpoints over (T-1) steps; thres is ascending (checked above). */
    g_mlp.qi_scale       = (partition == 1 && num_thresholds > 1)
        ? ((double)thres[num_thresholds - 1] - (double)thres[0]) / (double)(num_thresholds - 1)
        : 0.0;
    g_mlp.use_cache_ops  = use_cache_ops;
    g_mlp.initialized    = 1;
    return 0;
}

int finn_mlp_runner_destroy(void)
{
    memset(&g_mlp, 0, sizeof(g_mlp));
    return 0;
}

/* Optional second-buffer setter for double-buffered batch inference.
 * After this returns success, finn_mlp_infer_batch overlaps cpu_pre +
 * pack[N+1] with accel[N] and unpack + cpu_post[N-1]. Single-image
 * entries (finn_mlp_infer_one, _profiled) always use slot 0. */
int finn_mlp_set_second_buffers(
    void *ibuf_b_virt, uint64_t ibuf_b_phys,
    void *obuf_b_virt, uint64_t obuf_b_phys)
{
    if (!g_mlp.initialized)                              return -1;
    if (ibuf_b_virt == NULL || obuf_b_virt == NULL)      return -2;
    g_mlp.ibuf_virt[1] = ibuf_b_virt;
    g_mlp.ibuf_phys[1] = ibuf_b_phys;
    g_mlp.obuf_virt[1] = obuf_b_virt;
    g_mlp.obuf_phys[1] = obuf_b_phys;
    g_mlp.n_buffers    = 2;
    return 0;
}

/* ============================================================
 * CPU pre / post stages — shared between the real and mock entries.
 * ============================================================ */

/* Compute act[mid_dim] = MultiThreshold((img / 255) @ W0, thres).
 * Inclusive (>=) convention. Stack-allocated scratch. */
static inline void cpu_pre(const uint8_t *img, uint8_t *act)
{
    const int IN  = g_mlp.in_dim;
    const int MID = g_mlp.mid_dim;
    const int T   = g_mlp.num_thresholds;
    const float *W  = g_mlp.W0;
    const float *TH = g_mlp.thres;

    float acc[MID];
    for (int c = 0; c < MID; c++) acc[c] = 0.0f;

    /* k outer, c inner -> row-major traversal of W (cache-friendly). */
    for (int k = 0; k < IN; k++) {
        const float v = (float)img[k];
        const float *Wk = W + (size_t)k * MID;
        for (int c = 0; c < MID; c++) acc[c] += v * Wk[c];
    }
    const float inv255 = 1.0f / 255.0f;
    for (int c = 0; c < MID; c++) acc[c] *= inv255;

    /* act[c] = count of thresholds on row c that are <= acc[c]. No early
     * break: robust regardless of row-sortedness, and T is 255 (INT8) or
     * 15 (INT4), small enough that the linear scan costs under a few µs. */
    for (int c = 0; c < MID; c++) {
        const float x = acc[c];
        const float *row = TH + (size_t)c * T;
        int count = 0;
        for (int j = 0; j < T; j++) {
            if (row[j] <= x) count++;
        }
        act[c] = (uint8_t)count;
    }
}

/* QI partition input quant. float-normalize the raw image (v = img/255) and
 * quantize to the FPGA's input dtype. Two bit-identical realizations:
 *
 *   cpu_pre_qi_tablewalk — the original 255-entry MultiThreshold scan:
 *     act[k] = #{ j : thres[j] <= v } - bias, bias = (T+1)/2 = 2^(N-1) for
 *     signed IDT. O(T) compares per pixel.
 *
 *   cpu_pre_qi_affine — uniform affine quantizer that reproduces the scan for
 *     the signed-INT8 input QuantIdentity at O(1) per pixel:
 *     q = clip(round_half_up(v / s), -128, 127), s = g_mlp.qi_scale (derived
 *     from the table at init). Valid because the input QuantIdentity is a
 *     symmetric uniform quantizer, so thres[j] = (j - (T-1)/2 - 0.5)*s and the
 *     -2^(N-1) bias is absorbed into zero_point 0.
 *
 * cpu_pre_qi dispatches: affine for signed IDT (the only MLP QI case in this
 * project; verified host-side, bit-identical over all 256 pixel values for all
 * five MLP-INT8-QI builds), table walk otherwise. Linear1 stays on the FPGA for
 * both. The table walk is retained for the verification hook
 * (finn_mlp_qi_selfcheck) and as the unsigned fallback. */
static inline void cpu_pre_qi_tablewalk(const uint8_t *img, uint8_t *act)
{
    const int IN  = g_mlp.in_dim;
    const int T   = g_mlp.num_thresholds;
    const float *row = g_mlp.thres;          /* shape (1, T) */
    const float inv255 = 1.0f / 255.0f;
    const int bias = g_mlp.idt_signed ? ((T + 1) >> 1) : 0;
    for (int k = 0; k < IN; k++) {
        const float x = (float)img[k] * inv255;
        int count = 0;
        for (int j = 0; j < T; j++) {
            if (row[j] <= x) count++;
        }
        act[k] = (uint8_t)(count - bias);
    }
}

static inline void cpu_pre_qi_affine(const uint8_t *img, uint8_t *act)
{
    const int IN = g_mlp.in_dim;
    const float inv255 = 1.0f / 255.0f;
    const double s = g_mlp.qi_scale;
    for (int k = 0; k < IN; k++) {
        const float v = (float)img[k] * inv255;   /* identical to the table walk */
        /* round-half-up: v >= 0 over the whole uint8 domain, so the (int) cast
         * truncates toward zero == floor here. NOT rint/lround (banker's). */
        int q = (int)((double)v / s + 0.5);
        if (q < -128) q = -128;
        else if (q > 127) q = 127;
        act[k] = (uint8_t)q;
    }
}

static inline void cpu_pre_qi(const uint8_t *img, uint8_t *act)
{
    if (g_mlp.idt_signed) cpu_pre_qi_affine(img, act);
    else                  cpu_pre_qi_tablewalk(img, act);
}

/* argmax( hw * mul + add ) over num_classes. */
static inline int cpu_post_argmax(const int32_t *hw)
{
    const int K = g_mlp.num_classes;
    const float mul = g_mlp.mul;
    const float *add = g_mlp.add;
    int best_idx = 0;
    float best = (float)hw[0] * mul + add[0];
    for (int c = 1; c < K; c++) {
        float v = (float)hw[c] * mul + add[c];
        if (v > best) { best = v; best_idx = c; }
    }
    return best_idx;
}

/* ============================================================
 * Real inference: CPU pre -> pack -> DMA -> unpack -> CPU post.
 * ============================================================ */

static inline void trigger_dma(int slot)
{
    volatile uint32_t *idma = (volatile uint32_t *)g_mlp.idma_mmio;
    volatile uint32_t *odma = (volatile uint32_t *)g_mlp.odma_mmio;
    uint64_t ip = g_mlp.ibuf_phys[slot];
    uint64_t op = g_mlp.obuf_phys[slot];

    /* Output DMA armed first (matches v1 finn_t and driver_base.execute_on_buffers). */
    mmio_write(odma, DMA_REG_ADDR_LO, (uint32_t)(op & 0xFFFFFFFFu));
    mmio_write(odma, DMA_REG_ADDR_HI, (uint32_t)((op >> 32) & 0xFFFFFFFFu));
    mmio_write(odma, DMA_REG_COUNT,   1);
    mmio_write(odma, DMA_REG_CTRL,    1);

    mmio_write(idma, DMA_REG_ADDR_LO, (uint32_t)(ip & 0xFFFFFFFFu));
    mmio_write(idma, DMA_REG_ADDR_HI, (uint32_t)((ip >> 32) & 0xFFFFFFFFu));
    mmio_write(idma, DMA_REG_COUNT,   1);
    mmio_write(idma, DMA_REG_CTRL,    1);
}

static inline void wait_dma(void)
{
    volatile uint32_t *odma = (volatile uint32_t *)g_mlp.odma_mmio;
    while ((mmio_read(odma, DMA_REG_CTRL) & DMA_AP_DONE_BIT) == 0) { }
}

int finn_mlp_infer_one(const uint8_t *img)
{
    if (!g_mlp.initialized) return -1;

    uint8_t act[g_mlp.mid_dim];
    int32_t hw[g_mlp.num_classes];

    if (g_mlp.partition == 1) cpu_pre_qi(img, act);
    else                      cpu_pre(img, act);
    g_mlp.pack(act, (uint8_t *)g_mlp.ibuf_virt[0], g_mlp.mid_dim);
    if (g_mlp.use_cache_ops) dcache_clean(g_mlp.ibuf_virt[0], g_mlp.ibuf_bytes);

    trigger_dma(0);
    wait_dma();

    if (g_mlp.use_cache_ops) dcache_invalidate(g_mlp.obuf_virt[0], g_mlp.obuf_bytes);
    g_mlp.unpack((const uint8_t *)g_mlp.obuf_virt[0], hw, g_mlp.num_classes);

    return cpu_post_argmax(hw);
}

/* Double-buffered batch loop. Same overlap structure as the CNN runner:
 * cpu_pre + pack[N+1] runs while accel[N] computes; unpack + post[N-1]
 * runs while accel[N] computes. Single act/hw scratch is fine
 * (sequentially used per iteration); only the FPGA-touched ibuf/obuf
 * pair need to be doubled. */
static int infer_batch_double(
    const uint8_t *images, const int32_t *labels,
    int n_samples, int32_t *predictions_out)
{
    if (n_samples == 0) return 0;
    const int IN = g_mlp.in_dim;
    int correct = 0;

    if (n_samples == 1) {
        int pred = finn_mlp_infer_one(images);
        predictions_out[0] = (int32_t)pred;
        return (labels && pred == labels[0]) ? 1 : 0;
    }

    uint8_t act[g_mlp.mid_dim];
    int32_t hw[g_mlp.num_classes];

    /* Helper: cpu_pre + pack into slot's ibuf + dcache clean. */
    #define STAGE_INTO(slot_, img_ptr_) do { \
        if (g_mlp.partition == 1) cpu_pre_qi(img_ptr_, act); \
        else                      cpu_pre(img_ptr_, act); \
        g_mlp.pack(act, (uint8_t *)g_mlp.ibuf_virt[(slot_)], g_mlp.mid_dim); \
        if (g_mlp.use_cache_ops) \
            dcache_clean(g_mlp.ibuf_virt[(slot_)], g_mlp.ibuf_bytes); \
    } while (0)

    /* Helper: cache invalidate + unpack(slot's obuf) + cpu_post -> pred. */
    #define TAIL_FROM(slot_, pred_lvalue_) do { \
        if (g_mlp.use_cache_ops) \
            dcache_invalidate(g_mlp.obuf_virt[(slot_)], g_mlp.obuf_bytes); \
        g_mlp.unpack((const uint8_t *)g_mlp.obuf_virt[(slot_)], \
                      hw, g_mlp.num_classes); \
        (pred_lvalue_) = cpu_post_argmax(hw); \
    } while (0)

    int slot = 0;
    STAGE_INTO(slot, images);
    trigger_dma(slot);

    for (int i = 1; i < n_samples; i++) {
        int nslot = 1 - slot;
        STAGE_INTO(nslot, images + (size_t)i * IN);
        wait_dma();
        trigger_dma(nslot);
        int pred;
        TAIL_FROM(slot, pred);
        predictions_out[i - 1] = (int32_t)pred;
        if (labels && pred == labels[i - 1]) correct++;
        slot = nslot;
    }

    wait_dma();
    int pred;
    TAIL_FROM(slot, pred);
    predictions_out[n_samples - 1] = (int32_t)pred;
    if (labels && pred == labels[n_samples - 1]) correct++;

    #undef STAGE_INTO
    #undef TAIL_FROM
    return correct;
}

int finn_mlp_infer_batch(
    const uint8_t *images,
    const int32_t *labels,        /* may be NULL */
    int            n_samples,
    int32_t       *predictions_out)
{
    if (!g_mlp.initialized) return -1;
    if (n_samples < 0) return -2;

    if (g_mlp.n_buffers == 2) {
        return infer_batch_double(images, labels, n_samples, predictions_out);
    }
    int correct = 0;
    const int IN = g_mlp.in_dim;
    for (int i = 0; i < n_samples; i++) {
        int pred = finn_mlp_infer_one(images + (size_t)i * IN);
        predictions_out[i] = (int32_t)pred;
        if (labels && pred == labels[i]) correct++;
    }
    return correct;
}

/* ============================================================
 * Mock entry — used by the host-side correctness harness.
 *
 * Same CPU pre + pack + CPU post path as the real entry, but the DMA
 * step is replaced by reading `mock_obuf` as if it were the FPGA's
 * output. Pack result is written to `pack_scratch_out` (may be NULL)
 * so the harness can verify pack bytes in the same call.
 *
 * No MMIO, no cache ops: safe to call off-board.
 * ============================================================ */

int finn_mlp_infer_one_mock(
    const uint8_t *img,
    const uint8_t *mock_obuf,
    uint8_t       *pack_scratch_out)
{
    if (!g_mlp.initialized) return -1;

    uint8_t act[g_mlp.mid_dim];
    int32_t hw[g_mlp.num_classes];

    if (g_mlp.partition == 1) cpu_pre_qi(img, act);
    else                      cpu_pre(img, act);
    if (pack_scratch_out) {
        g_mlp.pack(act, pack_scratch_out, g_mlp.mid_dim);
    }
    g_mlp.unpack(mock_obuf, hw, g_mlp.num_classes);
    return cpu_post_argmax(hw);
}

/* ============================================================
 * QI affine verification hook (host-side correctness harness; no board).
 *
 * Runs BOTH the legacy 255-entry table walk and the new affine quantizer on a
 * single pixel value, through the ACTUAL C helpers (not a re-port), and returns
 * each as a signed int8 in [-128, 127]. The harness inits the runner with a
 * build's input threshold table (partition=qi, signed IDT), then sweeps pixel
 * 0..255 and asserts tablewalk_out == affine_out. 256 values is the entire
 * uint8 input domain, so agreement is a proof of bit-identity, not a sample.
 * Returns 0 on success; negative on misuse.
 * ============================================================ */
int finn_mlp_qi_selfcheck(int pixel, int *tablewalk_out, int *affine_out)
{
    if (!g_mlp.initialized)       return -1;
    if (g_mlp.partition != 1)     return -2;
    if (pixel < 0 || pixel > 255) return -3;
    if (g_mlp.in_dim < 1)         return -4;

    uint8_t img[g_mlp.in_dim];
    uint8_t act_tw[g_mlp.in_dim];
    uint8_t act_af[g_mlp.in_dim];
    for (int k = 0; k < g_mlp.in_dim; k++) img[k] = (uint8_t)pixel;

    cpu_pre_qi_tablewalk(img, act_tw);
    cpu_pre_qi_affine(img, act_af);

    if (tablewalk_out) *tablewalk_out = (int)(int8_t)act_tw[0];
    if (affine_out)    *affine_out    = (int)(int8_t)act_af[0];
    return 0;
}

/* ============================================================
 * Profiled entry — one-shot diagnostic. Same semantics as
 * finn_mlp_infer_one, but splits the work into timed stages and
 * writes nanosecond durations to ns_out[0..5]:
 *   0 = CPU MatMul
 *   1 = MultiThreshold
 *   2 = Pack (+ optional dc cvac on ibuf)
 *   3 = DMA trigger + wait
 *   4 = dc civac on obuf (if enabled) + Unpack
 *   5 = CPU dequant + argmax
 *
 * Not called from the hot loop; Python invokes this once on a
 * representative image to characterize the per-stage cost. Keep the
 * code paths identical to finn_mlp_infer_one so the numbers reflect
 * the real hot path (cache-op placement, pack/unpack, DMA ordering).
 * ============================================================ */

static inline uint64_t mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int finn_mlp_infer_one_profiled(const uint8_t *img, uint64_t *ns_out)
{
    if (!g_mlp.initialized) return -1;

    const int IN  = g_mlp.in_dim;
    const int MID = g_mlp.mid_dim;
    const int T   = g_mlp.num_thresholds;
    const float *W  = g_mlp.W0;
    const float *TH = g_mlp.thres;

    float   acc[MID];
    uint8_t act[MID];
    int32_t hw[g_mlp.num_classes];

    uint64_t t0 = mono_ns();
    uint64_t t1;
    if (g_mlp.partition == 1) {
        /* QI: stage 0 (MatMul) is a no-op — Linear1 is on FPGA. Stage 1
         * (input quant) uses the same affine/table-walk dispatch as the hot
         * path (cpu_pre_qi), so the profiled input_quant time reflects the
         * affine op rather than the retired 255-entry scan. */
        (void)acc; (void)W;
        t1 = t0;
        cpu_pre_qi(img, act);
    } else {
        for (int c = 0; c < MID; c++) acc[c] = 0.0f;
        for (int k = 0; k < IN; k++) {
            const float v = (float)img[k];
            const float *Wk = W + (size_t)k * MID;
            for (int c = 0; c < MID; c++) acc[c] += v * Wk[c];
        }
        const float inv255 = 1.0f / 255.0f;
        for (int c = 0; c < MID; c++) acc[c] *= inv255;

        t1 = mono_ns();
        for (int c = 0; c < MID; c++) {
            const float x = acc[c];
            const float *row = TH + (size_t)c * T;
            int count = 0;
            for (int j = 0; j < T; j++) if (row[j] <= x) count++;
            act[c] = (uint8_t)count;
        }
    }

    uint64_t t2 = mono_ns();
    g_mlp.pack(act, (uint8_t *)g_mlp.ibuf_virt[0], g_mlp.mid_dim);
    if (g_mlp.use_cache_ops) dcache_clean(g_mlp.ibuf_virt[0], g_mlp.ibuf_bytes);

    uint64_t t3 = mono_ns();
    trigger_dma(0);
    wait_dma();

    uint64_t t4 = mono_ns();
    if (g_mlp.use_cache_ops) dcache_invalidate(g_mlp.obuf_virt[0], g_mlp.obuf_bytes);
    g_mlp.unpack((const uint8_t *)g_mlp.obuf_virt[0], hw, g_mlp.num_classes);

    uint64_t t5 = mono_ns();
    int pred = cpu_post_argmax(hw);
    uint64_t t6 = mono_ns();

    ns_out[0] = t1 - t0;
    ns_out[1] = t2 - t1;
    ns_out[2] = t3 - t2;
    ns_out[3] = t4 - t3;
    ns_out[4] = t5 - t4;
    ns_out[5] = t6 - t5;
    return pred;
}
