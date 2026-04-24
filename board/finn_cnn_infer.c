/*
 * finn_cnn_infer.c — board-side hot path for the FINN CNN MNIST deploy.
 *
 * Mirrors finn_mlp_infer.c's structure: Python owns bitstream load, PYNQ
 * buffer allocation, MMIO mapping, dataset I/O, and JSON results; C owns
 * the per-image hot loop (CPU pre-stage im2col + MatMul + MultiThreshold,
 * pack, DMA trigger + poll, GAP + classifier + argmax).
 *
 * INT8-only in this revision. Function-pointer dispatch for pack is kept
 * for consistency with finn_mlp_infer.c and for future INT4 support; in
 * the current hot path pack is `finn_cnn_pack_uint8` = memcpy.
 *
 * MultiThreshold uses the inclusive (>=) convention, matching
 * qonnx.custom_op.general.multithreshold and the patched
 * benchmark.py:multithreshold.
 *
 * MNIST-only scope: init rejects img_c != 1, kernel_size != 3, or pad != 1
 * (those would be CIFAR-10 / a different conv shape; Python fallback is
 * expected for those cases). Rejects any precision != 8.
 *
 * Build on board (ARM64):
 *   gcc -O2 -shared -fPIC -Wall -o libfinn_cnn_infer.so finn_cnn_infer.c
 * Build on host (x86_64) for the correctness harness: same command; the
 * ARM cache-op asm is gated behind __aarch64__ and becomes a no-op.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DMA_REG_ADDR_LO  (0x10 / 4)
#define DMA_REG_ADDR_HI  (0x14 / 4)
#define DMA_REG_COUNT    (0x1C / 4)
#define DMA_REG_CTRL     (0x00 / 4)
#define DMA_AP_DONE_BIT  0x2
#define CACHE_LINE_BYTES 64

/* ---- ARMv8 cache maintenance (gated for host compilation) ---------- */

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
 * Pack / unpack utilities (extern, so the harness can call them
 * directly via ctypes).
 *
 * Both are memcpy for the INT8 deploy: finnpy_to_packed_bytearray on a
 * (1, 28, 28, 1, 8) UINT8 tensor with reverse_endian+reverse_inner
 * hits the fast path (view-as-uint8) so the bytes on the DMA side are
 * the same NHWC-channel-fastest bytes that C produces with memcpy.
 * Same argument for the UINT8 obuf.
 * ============================================================ */

void finn_cnn_pack_uint8(const uint8_t *act, uint8_t *ibuf, int n)
{
    memcpy(ibuf, act, (size_t)n);
}

void finn_cnn_unpack_uint8(const uint8_t *obuf, uint8_t *out, int n)
{
    memcpy(out, obuf, (size_t)n);
}

/* Binary search equivalent of the MultiThreshold linear scan.
 *
 * For a sorted-ascending row of `n` thresholds and activation `x`, the
 * result is the count of values `t` in the row with `t <= x`, i.e. the
 * activation level under the inclusive (>=) MT convention.
 *
 * Returns the first index `lo` in [0, n] such that all row[0..lo) are
 * <= x and all row[lo..n) are > x.  Exact-tie case: row[j] == x falls
 * into the "<= x" half and is counted, matching `row[j] <= x` in the
 * linear scan.
 *
 * Assumes sorted-ascending; runner_init spot-checks row 0 of thres.
 */
static inline int mt_upper_bound(const float *row, int n, float x)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (row[mid] <= x) lo = mid + 1;
        else               hi = mid;
    }
    return lo;
}

/* ============================================================
 * Runner state
 * ============================================================ */

typedef void (*pack_fn_t)(const uint8_t *act, uint8_t *ibuf, int n);

typedef struct {
    /* caller-owned buffers + MMIO */
    void     *ibuf_virt;   uint64_t  ibuf_phys;
    void     *obuf_virt;   uint64_t  obuf_phys;
    void     *idma_mmio;   void     *odma_mmio;

    /* explicit geometry */
    int img_h, img_w, img_c;
    int kernel_size, pad;
    int fpga_in_c;
    int fpga_out_h, fpga_out_w, fpga_out_c;
    int num_classes;
    int num_thresholds;
    int patch_dim;        /* kernel_size * kernel_size * img_c */
    int ibuf_bytes;       /* img_h * img_w * fpga_in_c  (INT8 = 1 B/elem) */
    int obuf_bytes;       /* fpga_out_h * fpga_out_w * fpga_out_c (INT8) */

    /* precision dispatch (unpack isn't used in the hot path — GAP reads
     * obuf_virt directly — so only pack is stored here) */
    pack_fn_t pack;

    /* weights & biases — caller-owned, referenced */
    const float *W_conv;
    const float *thres;
    const float *W_cls;
    float        mul;
    const float *add;

    /* persistent scratch (heap, allocated in init, freed in destroy) */
    float   *img_f32;
    float   *patches_f32;
    float   *acc_f32;
    uint8_t *act_u8;

    int use_cache_ops;
    int initialized;
} cnn_runner_state_t;

static cnn_runner_state_t g_cnn = {0};

/* Map precision -> pack fn + input-byte-per-element. Only INT8 wired. */
static int select_dispatch(int precision, pack_fn_t *pack_out, int *in_bytes_per_elem)
{
    switch (precision) {
        case 8:
            *pack_out          = finn_cnn_pack_uint8;
            *in_bytes_per_elem = 1;
            return 0;
        default:
            return -1;
    }
}

int finn_cnn_runner_init(
    int  precision,
    int  img_h, int img_w, int img_c,
    int  kernel_size, int pad,
    int  fpga_in_c,
    int  fpga_out_h, int fpga_out_w, int fpga_out_c,
    int  num_classes,
    int  num_thresholds,
    int  use_cache_ops,
    void *ibuf_virt,  uint64_t ibuf_phys,
    void *obuf_virt,  uint64_t obuf_phys,
    void *idma_mmio,  void    *odma_mmio,
    const float *W_conv,
    const float *thres,
    const float *W_cls,
    float        mul,
    const float *add)
{
    /* Scope validation: this runner is MNIST-only and INT8-only. */
    if (img_c != 1)                            return -2;
    if (kernel_size != 3 || pad != 1)          return -3;
    if (img_h <= 0 || img_w <= 0)              return -4;
    if (fpga_in_c <= 0 || fpga_out_h <= 0 ||
        fpga_out_w <= 0 || fpga_out_c <= 0)    return -5;
    if (num_classes <= 0 || num_thresholds <= 0) return -6;
    if (ibuf_virt == NULL || obuf_virt == NULL)  return -7;
    if (W_conv == NULL || thres == NULL ||
        W_cls  == NULL || add   == NULL)       return -8;

    pack_fn_t pack;
    int       in_be;
    if (select_dispatch(precision, &pack, &in_be) != 0) return -9;

    /* First-row spot-check of thresholds (FINN emits ascending). */
    for (int j = 1; j < num_thresholds; j++) {
        if (thres[j] < thres[j - 1]) return -10;
    }

    const int patch_dim = kernel_size * kernel_size * img_c;
    const size_t img_n     = (size_t)img_h * img_w * img_c;
    const size_t patches_n = (size_t)img_h * img_w * patch_dim;
    const size_t acc_n     = (size_t)img_h * img_w * fpga_in_c;
    const size_t act_n     = (size_t)img_h * img_w * fpga_in_c;

    float   *img_f32     = (float   *)malloc(img_n     * sizeof(float));
    float   *patches_f32 = (float   *)malloc(patches_n * sizeof(float));
    float   *acc_f32     = (float   *)malloc(acc_n     * sizeof(float));
    uint8_t *act_u8      = (uint8_t *)malloc(act_n);
    if (!img_f32 || !patches_f32 || !acc_f32 || !act_u8) {
        free(img_f32); free(patches_f32); free(acc_f32); free(act_u8);
        return -11;
    }

    memset(&g_cnn, 0, sizeof(g_cnn));
    g_cnn.ibuf_virt      = ibuf_virt;
    g_cnn.ibuf_phys      = ibuf_phys;
    g_cnn.obuf_virt      = obuf_virt;
    g_cnn.obuf_phys      = obuf_phys;
    g_cnn.idma_mmio      = idma_mmio;
    g_cnn.odma_mmio      = odma_mmio;
    g_cnn.img_h          = img_h;
    g_cnn.img_w          = img_w;
    g_cnn.img_c          = img_c;
    g_cnn.kernel_size    = kernel_size;
    g_cnn.pad            = pad;
    g_cnn.fpga_in_c      = fpga_in_c;
    g_cnn.fpga_out_h     = fpga_out_h;
    g_cnn.fpga_out_w     = fpga_out_w;
    g_cnn.fpga_out_c     = fpga_out_c;
    g_cnn.num_classes    = num_classes;
    g_cnn.num_thresholds = num_thresholds;
    g_cnn.patch_dim      = patch_dim;
    g_cnn.ibuf_bytes     = img_h * img_w * fpga_in_c * in_be;
    g_cnn.obuf_bytes     = fpga_out_h * fpga_out_w * fpga_out_c;
    g_cnn.pack           = pack;
    g_cnn.W_conv         = W_conv;
    g_cnn.thres          = thres;
    g_cnn.W_cls          = W_cls;
    g_cnn.mul            = mul;
    g_cnn.add            = add;
    g_cnn.img_f32        = img_f32;
    g_cnn.patches_f32    = patches_f32;
    g_cnn.acc_f32        = acc_f32;
    g_cnn.act_u8         = act_u8;
    g_cnn.use_cache_ops  = use_cache_ops;
    g_cnn.initialized    = 1;
    return 0;
}

int finn_cnn_runner_destroy(void)
{
    free(g_cnn.img_f32);
    free(g_cnn.patches_f32);
    free(g_cnn.acc_f32);
    free(g_cnn.act_u8);
    memset(&g_cnn, 0, sizeof(g_cnn));
    return 0;
}

/* ============================================================
 * CPU pre-stage: normalize + im2col + first MatMul + MultiThreshold.
 * Writes into g_cnn.act_u8 (length img_h*img_w*fpga_in_c).
 * ============================================================ */

static inline void cpu_pre(const uint8_t *img)
{
    const int H = g_cnn.img_h;
    const int W = g_cnn.img_w;
    const int C = g_cnn.img_c;
    const int K = g_cnn.kernel_size;
    const int P = g_cnn.pad;
    const int PD = g_cnn.patch_dim;
    const int Fc = g_cnn.fpga_in_c;
    const int T  = g_cnn.num_thresholds;
    const float *Wc  = g_cnn.W_conv;
    const float *TH  = g_cnn.thres;
    float   *im      = g_cnn.img_f32;
    float   *patches = g_cnn.patches_f32;
    float   *acc     = g_cnn.acc_f32;
    uint8_t *act     = g_cnn.act_u8;

    /* Stage 0: cast + normalize */
    const float inv255 = 1.0f / 255.0f;
    for (int i = 0; i < H * W * C; i++) im[i] = (float)img[i] * inv255;

    /* Stage 1: im2col (zero-padded, C-inner) */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float *pd = patches + (size_t)(y * W + x) * PD;
            int pi = 0;
            for (int ky = 0; ky < K; ky++) {
                int iy = y + ky - P;
                for (int kx = 0; kx < K; kx++) {
                    int ix = x + kx - P;
                    int in_bounds =
                        ((unsigned)iy < (unsigned)H) &&
                        ((unsigned)ix < (unsigned)W);
                    if (in_bounds) {
                        const float *src = im + (size_t)(iy * W + ix) * C;
                        for (int c = 0; c < C; c++) pd[pi++] = src[c];
                    } else {
                        for (int c = 0; c < C; c++) pd[pi++] = 0.0f;
                    }
                }
            }
        }
    }

    /* Stage 2: first MatMul
     *   acc[(y,x), o] = sum_k patches[(y,x), k] * W_conv[k, o]
     * Loop order: pixel outer, k outer within pixel, o inner.  Iterates
     * W_conv row-by-row (cache-friendly; row size = Fc * sizeof(float)). */
    for (int yx = 0; yx < H * W; yx++) {
        const float *pd = patches + (size_t)yx * PD;
        float *ap = acc + (size_t)yx * Fc;
        for (int o = 0; o < Fc; o++) ap[o] = 0.0f;
        for (int k = 0; k < PD; k++) {
            const float v = pd[k];
            const float *wk = Wc + (size_t)k * Fc;
            for (int o = 0; o < Fc; o++) ap[o] += v * wk[o];
        }
    }

    /* Stage 3: MultiThreshold via binary search on sorted-ascending rows.
     * Semantically identical to the linear scan's `count(row[j] <= x)` for
     * sorted input, at log2(T) compares instead of T.  For T=255: ~8 vs 255. */
    for (int yx = 0; yx < H * W; yx++) {
        const float *ap = acc + (size_t)yx * Fc;
        uint8_t *atp = act + (size_t)yx * Fc;
        for (int c = 0; c < Fc; c++) {
            const float *row = TH + (size_t)c * T;
            atp[c] = (uint8_t)mt_upper_bound(row, T, ap[c]);
        }
    }
}

/* ============================================================
 * CPU post-stage: GAP + second MatMul + dequant + bias + argmax.
 * Reads obuf_bytes directly (no separate unpack buffer).
 * ============================================================ */

static inline int cpu_post_argmax(const uint8_t *obuf_bytes)
{
    const int OH = g_cnn.fpga_out_h;
    const int OW = g_cnn.fpga_out_w;
    const int OC = g_cnn.fpga_out_c;
    const int K  = g_cnn.num_classes;
    const float *Wcls = g_cnn.W_cls;
    const float *add  = g_cnn.add;
    const float mul   = g_cnn.mul;

    /* GAP with uint32 accumulator. 49 * 255 = 12,495 << 2^32. */
    uint32_t acc[OC];
    for (int c = 0; c < OC; c++) acc[c] = 0;
    for (int i = 0; i < OH * OW; i++) {
        const uint8_t *row = obuf_bytes + (size_t)i * OC;
        for (int c = 0; c < OC; c++) acc[c] += row[c];
    }
    /* Cast to float for the single divide. OH*OW=49; 12495/49.0f = 255.0f
     * exactly, so the all-255 synthetic case hits exact equality. */
    float feat[OC];
    for (int c = 0; c < OC; c++) feat[c] = (float)acc[c] / (float)(OH * OW);

    /* Second MatMul: logits[o] = sum_c feat[c] * W_cls[c, o] */
    float logits[K];
    for (int o = 0; o < K; o++) logits[o] = 0.0f;
    for (int c = 0; c < OC; c++) {
        const float v = feat[c];
        const float *wr = Wcls + (size_t)c * K;
        for (int o = 0; o < K; o++) logits[o] += v * wr[o];
    }

    /* Dequant + bias + argmax. mul is positive scalar in both deploys, so
     * it's argmax-irrelevant, but applied for parity. */
    int best_idx = 0;
    float best_v = logits[0] * mul + add[0];
    for (int o = 1; o < K; o++) {
        float v = logits[o] * mul + add[o];
        if (v > best_v) { best_v = v; best_idx = o; }
    }
    return best_idx;
}

/* ============================================================
 * DMA helpers (same pattern as finn_t / finn_mlp runners).
 * ============================================================ */

static inline void trigger_dma(void)
{
    volatile uint32_t *idma = (volatile uint32_t *)g_cnn.idma_mmio;
    volatile uint32_t *odma = (volatile uint32_t *)g_cnn.odma_mmio;
    uint64_t ip = g_cnn.ibuf_phys;
    uint64_t op = g_cnn.obuf_phys;

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
    volatile uint32_t *odma = (volatile uint32_t *)g_cnn.odma_mmio;
    while ((mmio_read(odma, DMA_REG_CTRL) & DMA_AP_DONE_BIT) == 0) { }
}

/* ============================================================
 * Real inference entries.
 * ============================================================ */

int finn_cnn_infer_one(const uint8_t *img)
{
    if (!g_cnn.initialized) return -1;

    cpu_pre(img);
    g_cnn.pack(g_cnn.act_u8, (uint8_t *)g_cnn.ibuf_virt, g_cnn.ibuf_bytes);
    if (g_cnn.use_cache_ops) dcache_clean(g_cnn.ibuf_virt, g_cnn.ibuf_bytes);

    trigger_dma();
    wait_dma();

    if (g_cnn.use_cache_ops) dcache_invalidate(g_cnn.obuf_virt, g_cnn.obuf_bytes);
    return cpu_post_argmax((const uint8_t *)g_cnn.obuf_virt);
}

int finn_cnn_infer_batch(
    const uint8_t *images,
    const int32_t *labels,
    int            n_samples,
    int32_t       *predictions_out)
{
    if (!g_cnn.initialized) return -1;
    if (n_samples < 0) return -2;

    const int IN = g_cnn.img_h * g_cnn.img_w * g_cnn.img_c;
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        int pred = finn_cnn_infer_one(images + (size_t)i * IN);
        predictions_out[i] = (int32_t)pred;
        if (labels && pred == labels[i]) correct++;
    }
    return correct;
}

/* ============================================================
 * Mock entry for host-side harness. Skips DMA; runs cpu_pre + pack
 * (into caller-provided scratch, may be NULL) + cpu_post using
 * caller-provided mock_obuf bytes.
 * ============================================================ */

int finn_cnn_infer_one_mock(
    const uint8_t *img,
    const uint8_t *mock_obuf,
    uint8_t       *pack_scratch_out)
{
    if (!g_cnn.initialized) return -1;
    cpu_pre(img);
    if (pack_scratch_out) {
        g_cnn.pack(g_cnn.act_u8, pack_scratch_out, g_cnn.ibuf_bytes);
    }
    return cpu_post_argmax(mock_obuf);
}

/* ============================================================
 * Profiled entry: 10-stage nanosecond breakdown.
 *
 * Stages (must match Sam's table and the harness):
 *   0 = cast + normalize
 *   1 = im2col
 *   2 = first MatMul
 *   3 = MultiThreshold
 *   4 = pack (+ dc cvac when use_cache_ops)
 *   5 = DMA trigger + wait
 *   6 = dc civac (when use_cache_ops) — the "unpack" slot; GAP reads obuf
 *       directly so there's no unpack-to-buffer step
 *   7 = GAP
 *   8 = second MatMul
 *   9 = dequant + bias + argmax
 * ============================================================ */

static inline uint64_t mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int finn_cnn_infer_one_profiled(const uint8_t *img, uint64_t *ns_out)
{
    if (!g_cnn.initialized) return -1;

    const int H  = g_cnn.img_h;
    const int W  = g_cnn.img_w;
    const int C  = g_cnn.img_c;
    const int Kk = g_cnn.kernel_size;
    const int P  = g_cnn.pad;
    const int PD = g_cnn.patch_dim;
    const int Fc = g_cnn.fpga_in_c;
    const int T  = g_cnn.num_thresholds;
    const int OH = g_cnn.fpga_out_h;
    const int OW = g_cnn.fpga_out_w;
    const int OC = g_cnn.fpga_out_c;
    const int NC = g_cnn.num_classes;
    const float *Wc   = g_cnn.W_conv;
    const float *TH   = g_cnn.thres;
    const float *Wcls = g_cnn.W_cls;
    const float *add  = g_cnn.add;
    const float mul   = g_cnn.mul;
    float   *im      = g_cnn.img_f32;
    float   *patches = g_cnn.patches_f32;
    float   *acc     = g_cnn.acc_f32;
    uint8_t *act     = g_cnn.act_u8;

    uint64_t t[11];
    t[0] = mono_ns();

    /* Stage 0: cast + normalize */
    const float inv255 = 1.0f / 255.0f;
    for (int i = 0; i < H * W * C; i++) im[i] = (float)img[i] * inv255;
    t[1] = mono_ns();

    /* Stage 1: im2col */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float *pd = patches + (size_t)(y * W + x) * PD;
            int pi = 0;
            for (int ky = 0; ky < Kk; ky++) {
                int iy = y + ky - P;
                for (int kx = 0; kx < Kk; kx++) {
                    int ix = x + kx - P;
                    int in_bounds =
                        ((unsigned)iy < (unsigned)H) &&
                        ((unsigned)ix < (unsigned)W);
                    if (in_bounds) {
                        const float *src = im + (size_t)(iy * W + ix) * C;
                        for (int c = 0; c < C; c++) pd[pi++] = src[c];
                    } else {
                        for (int c = 0; c < C; c++) pd[pi++] = 0.0f;
                    }
                }
            }
        }
    }
    t[2] = mono_ns();

    /* Stage 2: first MatMul */
    for (int yx = 0; yx < H * W; yx++) {
        const float *pd = patches + (size_t)yx * PD;
        float *ap = acc + (size_t)yx * Fc;
        for (int o = 0; o < Fc; o++) ap[o] = 0.0f;
        for (int k = 0; k < PD; k++) {
            const float v = pd[k];
            const float *wk = Wc + (size_t)k * Fc;
            for (int o = 0; o < Fc; o++) ap[o] += v * wk[o];
        }
    }
    t[3] = mono_ns();

    /* Stage 3: MultiThreshold (binary search; see mt_upper_bound comment) */
    for (int yx = 0; yx < H * W; yx++) {
        const float *ap = acc + (size_t)yx * Fc;
        uint8_t *atp = act + (size_t)yx * Fc;
        for (int c = 0; c < Fc; c++) {
            const float *row = TH + (size_t)c * T;
            atp[c] = (uint8_t)mt_upper_bound(row, T, ap[c]);
        }
    }
    t[4] = mono_ns();

    /* Stage 4: pack (+ cache clean) */
    g_cnn.pack(act, (uint8_t *)g_cnn.ibuf_virt, g_cnn.ibuf_bytes);
    if (g_cnn.use_cache_ops) dcache_clean(g_cnn.ibuf_virt, g_cnn.ibuf_bytes);
    t[5] = mono_ns();

    /* Stage 5: DMA trigger + wait */
    trigger_dma();
    wait_dma();
    t[6] = mono_ns();

    /* Stage 6: cache invalidate ("unpack" slot; GAP consumes obuf directly) */
    if (g_cnn.use_cache_ops) dcache_invalidate(g_cnn.obuf_virt, g_cnn.obuf_bytes);
    t[7] = mono_ns();

    /* Stage 7: GAP */
    const uint8_t *ob = (const uint8_t *)g_cnn.obuf_virt;
    uint32_t gap_acc[OC];
    for (int c = 0; c < OC; c++) gap_acc[c] = 0;
    for (int i = 0; i < OH * OW; i++) {
        const uint8_t *row = ob + (size_t)i * OC;
        for (int c = 0; c < OC; c++) gap_acc[c] += row[c];
    }
    float feat[OC];
    for (int c = 0; c < OC; c++) feat[c] = (float)gap_acc[c] / (float)(OH * OW);
    t[8] = mono_ns();

    /* Stage 8: second MatMul */
    float logits[NC];
    for (int o = 0; o < NC; o++) logits[o] = 0.0f;
    for (int c = 0; c < OC; c++) {
        const float v = feat[c];
        const float *wr = Wcls + (size_t)c * NC;
        for (int o = 0; o < NC; o++) logits[o] += v * wr[o];
    }
    t[9] = mono_ns();

    /* Stage 9: dequant + bias + argmax */
    int best_idx = 0;
    float best_v = logits[0] * mul + add[0];
    for (int o = 1; o < NC; o++) {
        float v = logits[o] * mul + add[o];
        if (v > best_v) { best_v = v; best_idx = o; }
    }
    t[10] = mono_ns();

    for (int i = 0; i < 10; i++) ns_out[i] = t[i + 1] - t[i];
    return best_idx;
}
