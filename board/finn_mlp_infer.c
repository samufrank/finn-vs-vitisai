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
    for (int i = 0; i < n; i++) ibuf[i] = (uint8_t)(act[i] & 0x0Fu);
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
    void     *ibuf_virt;   uint64_t  ibuf_phys;
    void     *obuf_virt;   uint64_t  obuf_phys;
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

    int       use_cache_ops;
    int       initialized;
} mlp_runner_state_t;

static mlp_runner_state_t g_mlp = {0};

/* Map precision -> {pack, unpack, input byte/elem, output byte/elem}. */
static int select_dispatch(int precision,
                           pack_fn_t   *pack_out,
                           unpack_fn_t *unpack_out,
                           int *in_bytes_per_elem,
                           int *out_bytes_per_elem)
{
    switch (precision) {
        case 8:
            *pack_out            = finn_mlp_pack_uint8;
            *unpack_out          = finn_mlp_unpack_int24_le;
            *in_bytes_per_elem   = 1;   /* UINT8 */
            *out_bytes_per_elem  = 3;   /* INT24 */
            return 0;
        case 4:
            *pack_out            = finn_mlp_pack_uint4;
            *unpack_out          = finn_mlp_unpack_int16_le;
            *in_bytes_per_elem   = 1;   /* UINT4 with PE=SIMD=1 folding */
            *out_bytes_per_elem  = 2;   /* INT16 */
            return 0;
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
    int       use_cache_ops,
    void     *ibuf_virt, uint64_t ibuf_phys,
    void     *obuf_virt, uint64_t obuf_phys,
    void     *idma_mmio, void   *odma_mmio,
    const float *W0,
    const float *thres,
    float        mul,
    const float *add)
{
    if (in_dim <= 0 || mid_dim <= 0 || num_classes <= 0 || num_thresholds <= 0)
        return -2;
    if (ibuf_virt == NULL || obuf_virt == NULL) return -3;
    if (W0 == NULL || thres == NULL || add == NULL) return -4;

    pack_fn_t   pack;
    unpack_fn_t unpack;
    int         in_be, out_be;
    if (select_dispatch(precision, &pack, &unpack, &in_be, &out_be) != 0)
        return -5;

    /* First-row sort spot check (FINN emits ascending thresholds). Catches
     * the most common misuse; full check over all mid_dim rows is overkill. */
    for (int j = 1; j < num_thresholds; j++) {
        if (thres[j] < thres[j - 1]) return -6;
    }

    g_mlp.ibuf_virt      = ibuf_virt;
    g_mlp.ibuf_phys      = ibuf_phys;
    g_mlp.obuf_virt      = obuf_virt;
    g_mlp.obuf_phys      = obuf_phys;
    g_mlp.idma_mmio      = idma_mmio;
    g_mlp.odma_mmio      = odma_mmio;
    g_mlp.in_dim         = in_dim;
    g_mlp.mid_dim        = mid_dim;
    g_mlp.num_classes    = num_classes;
    g_mlp.num_thresholds = num_thresholds;
    g_mlp.ibuf_bytes     = mid_dim * in_be;
    g_mlp.obuf_bytes     = num_classes * out_be;
    g_mlp.pack           = pack;
    g_mlp.unpack         = unpack;
    g_mlp.W0             = W0;
    g_mlp.thres          = thres;
    g_mlp.mul            = mul;
    g_mlp.add            = add;
    g_mlp.use_cache_ops  = use_cache_ops;
    g_mlp.initialized    = 1;
    return 0;
}

int finn_mlp_runner_destroy(void)
{
    memset(&g_mlp, 0, sizeof(g_mlp));
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

static inline void trigger_dma(void)
{
    volatile uint32_t *idma = (volatile uint32_t *)g_mlp.idma_mmio;
    volatile uint32_t *odma = (volatile uint32_t *)g_mlp.odma_mmio;
    uint64_t ip = g_mlp.ibuf_phys;
    uint64_t op = g_mlp.obuf_phys;

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

    cpu_pre(img, act);
    g_mlp.pack(act, (uint8_t *)g_mlp.ibuf_virt, g_mlp.mid_dim);
    if (g_mlp.use_cache_ops) dcache_clean(g_mlp.ibuf_virt, g_mlp.ibuf_bytes);

    trigger_dma();
    wait_dma();

    if (g_mlp.use_cache_ops) dcache_invalidate(g_mlp.obuf_virt, g_mlp.obuf_bytes);
    g_mlp.unpack((const uint8_t *)g_mlp.obuf_virt, hw, g_mlp.num_classes);

    return cpu_post_argmax(hw);
}

int finn_mlp_infer_batch(
    const uint8_t *images,
    const int32_t *labels,        /* may be NULL */
    int            n_samples,
    int32_t       *predictions_out)
{
    if (!g_mlp.initialized) return -1;
    if (n_samples < 0) return -2;

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

    cpu_pre(img, act);
    if (pack_scratch_out) {
        g_mlp.pack(act, pack_scratch_out, g_mlp.mid_dim);
    }
    g_mlp.unpack(mock_obuf, hw, g_mlp.num_classes);
    return cpu_post_argmax(hw);
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
    for (int c = 0; c < MID; c++) acc[c] = 0.0f;
    for (int k = 0; k < IN; k++) {
        const float v = (float)img[k];
        const float *Wk = W + (size_t)k * MID;
        for (int c = 0; c < MID; c++) acc[c] += v * Wk[c];
    }
    const float inv255 = 1.0f / 255.0f;
    for (int c = 0; c < MID; c++) acc[c] *= inv255;

    uint64_t t1 = mono_ns();
    for (int c = 0; c < MID; c++) {
        const float x = acc[c];
        const float *row = TH + (size_t)c * T;
        int count = 0;
        for (int j = 0; j < T; j++) if (row[j] <= x) count++;
        act[c] = (uint8_t)count;
    }

    uint64_t t2 = mono_ns();
    g_mlp.pack(act, (uint8_t *)g_mlp.ibuf_virt, g_mlp.mid_dim);
    if (g_mlp.use_cache_ops) dcache_clean(g_mlp.ibuf_virt, g_mlp.ibuf_bytes);

    uint64_t t3 = mono_ns();
    trigger_dma();
    wait_dma();

    uint64_t t4 = mono_ns();
    if (g_mlp.use_cache_ops) dcache_invalidate(g_mlp.obuf_virt, g_mlp.obuf_bytes);
    g_mlp.unpack((const uint8_t *)g_mlp.obuf_virt, hw, g_mlp.num_classes);

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
