/*
 * finn_cnn_infer.c — board-side hot path for the FINN CNN MNIST deploy.
 *
 * Mirrors finn_mlp_infer.c's structure: Python owns bitstream load, PYNQ
 * buffer allocation, MMIO mapping, dataset I/O, and JSON results; C owns
 * the per-image hot loop (CPU pre-stage im2col + MatMul + MultiThreshold,
 * pack, DMA trigger + poll, optional 2x2 stride-2 MaxPool when the FPGA
 * partition does not include the post-Conv-N MaxPool, GAP + classifier +
 * argmax).
 *
 * INT8-only in this revision. Function-pointer dispatch for pack is kept
 * for consistency with finn_mlp_infer.c and for future INT4 support; in
 * the current hot path pack is `finn_cnn_pack_uint8` = memcpy.
 *
 * MultiThreshold uses the inclusive (>=) convention, matching
 * qonnx.custom_op.general.multithreshold and the patched
 * benchmark.py:multithreshold.
 *
 * Supports MNIST (img_c=1) and CIFAR-10 (img_c=3); init still requires
 * kernel_size=3, pad=1 (matches every Brevitas QAT train in this project).
 * INT8 + INT4 (per-side) via select_dispatch.
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
 * directly via ctypes). One pair per FINN dtype — runner_init wires
 * them via select_dispatch.
 *
 * Pack signature: pack(act_u8[n], ibuf[ibuf_bytes], n_elements).
 *   n is element count; INT8 writes n bytes, INT4 writes n/2 bytes.
 * Unpack signature: unpack(obuf_packed[obuf_bytes], out_u8[n], n_elements).
 *   n is element count; obuf_packed has n bytes (INT8) or n/2 bytes (INT4).
 *   out_u8 always receives n bytes (1 channel per byte, value 0..255 INT8
 *   or 0..15 INT4) so the GAP loop is precision-agnostic.
 *
 * INT8: pack/unpack are memcpy. finnpy_to_packed_bytearray on a UINT8
 * tensor with reverse_endian+reverse_inner hits the fast path
 * (view-as-uint8); same on the unpack side.
 *
 * INT4 packing convention (verified against finnpy_to_packed_bytearray
 * with DataType['UINT4'], reverse_endian=True, reverse_inner=True):
 *   even-index element → LOW nibble of byte i/2
 *   odd-index element  → HIGH nibble of byte i/2
 * Equivalent: ibuf[i/2] = (act[i] & 0xF) | ((act[i+1] & 0xF) << 4).
 *
 * NOTE — INT4 CNN is 2-per-byte, *unlike* INT4 MLP which is 1-per-byte
 * (ibuf[i] = act[i] & 0x0F). The MLP uses PE=SIMD=1 so each cycle
 * carries one INT4 in its own byte; the CNN uses FMPadding SIMD=8 so
 * 8 INT4 channels per pixel pack into 4 bytes that stream together.
 * Confirmed by inspecting both deploys' driver/driver.py:
 *   MLP INT4: ishape_packed=(1,64,1)  → 1 byte per element
 *   CNN INT4: ishape_packed=(1,28,28,1,4) → 8 ch/pixel in 4 bytes (2-per-byte)
 *   CNN INT4: oshape_packed=(1, 7, 7,1,8) → 16 ch/pixel in 8 bytes (2-per-byte)
 * Don't reuse finn_mlp_pack_uint4 here. (And vice versa.)
 * ============================================================ */

void finn_cnn_pack_uint8(const uint8_t *act, uint8_t *ibuf, int n)
{
    memcpy(ibuf, act, (size_t)n);
}

void finn_cnn_unpack_uint8(const uint8_t *obuf, uint8_t *out, int n)
{
    memcpy(out, obuf, (size_t)n);
}

void finn_cnn_pack_uint4(const uint8_t *act, uint8_t *ibuf, int n)
{
    /* 2-per-byte INT4: n elements pack into n/2 bytes. Used when the
     * deploy's ishape_packed has SIMD/group last dim ≥ 2 (e.g. tiny CNN
     * INT4 with SIMD=8 → 4 bytes per group at INT4). Caller guarantees n
     * even (runner_init asserts via the pack-bytes ratio). */
    int half = n >> 1;
    for (int i = 0; i < half; i++) {
        uint8_t lo = act[2*i]     & 0x0Fu;
        uint8_t hi = act[2*i + 1] & 0x0Fu;
        ibuf[i] = (uint8_t)(lo | (hi << 4));
    }
}

void finn_cnn_unpack_uint4(const uint8_t *obuf, uint8_t *out, int n)
{
    /* 2-per-byte INT4: n/2 bytes unpack into n elements. */
    int half = n >> 1;
    for (int i = 0; i < half; i++) {
        uint8_t b = obuf[i];
        out[2*i]     = (uint8_t)(b & 0x0Fu);
        out[2*i + 1] = (uint8_t)((b >> 4) & 0x0Fu);
    }
}

void finn_cnn_pack_uint4_1pb(const uint8_t *act, uint8_t *ibuf, int n)
{
    /* 1-per-byte INT4: n elements pack into n bytes (low nibble used,
     * high nibble zero). Used when FINN folds with SIMD=1 (or any group
     * size that doesn't pack neatly at 4 bits) — the byte boundary
     * forces one element per byte even though only 4 bits carry data.
     * Examples: deep_3 CNN INT4 oshape_packed=(1,7,7,64,1) → SIMD=1 →
     * 1 byte per element. */
    for (int i = 0; i < n; i++) {
        ibuf[i] = act[i] & 0x0Fu;
    }
}

void finn_cnn_unpack_uint4_1pb(const uint8_t *obuf, uint8_t *out, int n)
{
    /* 1-per-byte INT4: n bytes unpack to n elements (low nibble of each
     * input byte; high nibble masked off). Mirrors finn_cnn_pack_uint4_1pb. */
    for (int i = 0; i < n; i++) {
        out[i] = obuf[i] & 0x0Fu;
    }
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

/* CPU 2x2 stride-2 valid-pad MaxPool over NHWC, byte-per-channel. Matches
 * PyTorch MaxPool2d(2) with default ceil_mode=False: OH = (IH - 2) / 2 + 1
 * (last input row dropped when IH odd). Used for deep_3-style topologies
 * where FINN puts the post-Conv-N MaxPool on the CPU side. INT8 and INT4
 * data both arrive here as 1 byte/channel after unpack, so this is
 * precision-agnostic. */
static inline void cpu_maxpool_2x2(const uint8_t *src, uint8_t *dst,
                                   int IW, int OH, int OW, int OC)
{
    for (int oy = 0; oy < OH; oy++) {
        const int iy0 = 2 * oy;
        const int iy1 = iy0 + 1;
        for (int ox = 0; ox < OW; ox++) {
            const int ix0 = 2 * ox;
            const int ix1 = ix0 + 1;
            const uint8_t *p00 = src + (size_t)(iy0 * IW + ix0) * OC;
            const uint8_t *p01 = src + (size_t)(iy0 * IW + ix1) * OC;
            const uint8_t *p10 = src + (size_t)(iy1 * IW + ix0) * OC;
            const uint8_t *p11 = src + (size_t)(iy1 * IW + ix1) * OC;
            uint8_t       *dp  = dst + (size_t)(oy  * OW + ox)  * OC;
            for (int c = 0; c < OC; c++) {
                uint8_t a = p00[c], b = p01[c], cc = p10[c], dd = p11[c];
                uint8_t m1 = a  > b  ? a  : b;
                uint8_t m2 = cc > dd ? cc : dd;
                dp[c] = m1 > m2 ? m1 : m2;
            }
        }
    }
}

/* ============================================================
 * Runner state
 * ============================================================ */

typedef void (*pack_fn_t)  (const uint8_t *act, uint8_t *ibuf, int n);
typedef void (*unpack_fn_t)(const uint8_t *obuf, uint8_t *out,  int n);

typedef struct {
    /* caller-owned buffers + MMIO. Double-buffered: slot 0 is set by
     * runner_init; slot 1 is optionally set by finn_cnn_set_second_buffers
     * to enable batch overlap of CPU prep[N+1] with FPGA accel[N]. When
     * unset, slot 1 mirrors slot 0 and n_buffers stays 1 (single-buffer
     * lockstep loop, byte-identical to the pre-double-buffer code path). */
    void     *ibuf_virt[2]; uint64_t ibuf_phys[2];
    void     *obuf_virt[2]; uint64_t obuf_phys[2];
    int       n_buffers;       /* 1 = single-buffered (default), 2 = double */
    void     *idma_mmio;   void     *odma_mmio;

    /* explicit geometry */
    int img_h, img_w, img_c;
    int kernel_size, pad;
    int fpga_in_c;
    int fpga_out_h, fpga_out_w, fpga_out_c;
    int num_classes;
    int num_thresholds;
    int patch_dim;        /* kernel_size * kernel_size * img_c */

    /* CPU-side post-FPGA MaxPool. 0 = none (tiny 2-conv: FPGA includes the
     * final MaxPool, GAP reads obuf_unpacked directly). 2 = 2x2 stride-2
     * valid-pad (deep_3 3-conv: FPGA ends after Conv3+MT; the post-Conv3
     * MaxPool runs on CPU between unpack and GAP). gap_h/gap_w == fpga_out_h/w
     * when k=0; (fpga_out_h - 2)/2 + 1 when k=2 (matches PyTorch
     * MaxPool2d(2)). */
    int cpu_post_maxpool_k;
    int gap_h, gap_w;

    /* Packed buffer layout. The byte-per-pixel count and pack/unpack
     * function are determined by the deploy's actual ishape_packed /
     * oshape_packed shapes — NOT derived from precision alone. INT4
     * deploys vary: tiny CNN INT4 has SIMD=8 → 2-per-byte (4 bytes/pixel
     * out for 16 channels); deep_3 CNN INT4 has SIMD=1 → 1-per-byte (64
     * bytes/pixel out for 64 channels, only low nibble carries data).
     * Caller passes ibuf_packed_bytes / obuf_packed_bytes per image so
     * we don't have to reverse-engineer the folding. */
    /* Partition layout. 0 = classic (Conv1 on CPU: im2col + MatMul +
     * MultiThreshold). 1 = qi (input QuantIdentity moved Conv1 onto FPGA;
     * CPU only does input MultiThreshold on raw image). For partition=1
     * W_conv may be NULL and the input MT thresholds are in `thres` with
     * shape (img_c, num_thresholds). All post-FPGA stages (MaxPool, GAP,
     * cls MatMul, dequant, argmax) are identical between partitions. */
    int partition;             /* 0 or 1 */
    /* Signedness of the FPGA-input dtype. 0 = UINT (classic CNN: post-ReLU
     * uint8 [0, 255]). 1 = INT (QI with QuantIdentity(Int8...): signed
     * int8 [-128, 127]). For signed FPGA inputs cpu_pre_qi must subtract
     * 2^(precision-1) from the threshold count to map [0, 2^N-1] -> the
     * signed two's-complement byte the FPGA expects. Classic deploys are
     * always unsigned in this project; init enforces idt_signed=0 there. */
    int idt_signed;            /* 0 or 1 */
    /* Per-side bitwidths. QI INT4 deploys are mixed-precision (INT8 input
     * from QuantIdentity, INT4 output from final QuantReLU), so the runner
     * stores in/out independently. cpu_pre_qi's signed-IDT bias formula
     * uses in_precision (the FPGA-input bitwidth). */
    int in_precision;          /* 4 or 8 */
    int out_precision;         /* 4 or 8 */
    int ibuf_bpp;              /* bytes per pixel into FPGA */
    int obuf_bpp;              /* bytes per pixel out of FPGA */
    int ibuf_bytes;            /* img_h * img_w * ibuf_bpp */
    int obuf_bytes;            /* fpga_out_h * fpga_out_w * obuf_bpp */

    pack_fn_t   pack;
    unpack_fn_t unpack;

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
    uint8_t *obuf_unpacked;    /* OH*OW*fpga_out_c bytes (1 ch/byte, 0..15 or 0..255) */
    uint8_t *gap_in_u8;        /* gap_h*gap_w*fpga_out_c bytes; allocated only when
                                * cpu_post_maxpool_k > 0. NULL when k=0 (the GAP
                                * reads obuf_unpacked directly). */

    int use_cache_ops;
    int initialized;
} cnn_runner_state_t;

static cnn_runner_state_t g_cnn = {0};

/* Pick pack/unpack functions and per-pixel byte counts from the actual
 * packed buffer sizes (Python computes these from the deploy's
 * ishape_packed / oshape_packed and passes them in). Input pack is
 * chosen by in_precision + ibuf byte ratio; output unpack by
 * out_precision + obuf byte ratio — independently, so mixed-precision
 * deploys (e.g. QI INT4: INT8 in, INT4 out) work. Supported ratios:
 *   precision=8: bytes == elements (memcpy)
 *   precision=4: bytes == elements/2 (2-per-byte) OR bytes == elements (1-per-byte)
 * Anything else returns -1. img_h*img_w must divide ibuf_packed_bytes;
 * fpga_out_h*fpga_out_w must divide obuf_packed_bytes. */
static int select_dispatch(int in_precision, int out_precision,
                           int n_in_elements,  int ibuf_packed_bytes,
                           int n_out_elements, int obuf_packed_bytes,
                           int img_pixels, int fpga_out_pixels,
                           pack_fn_t   *pack_out,
                           unpack_fn_t *unpack_out,
                           int *ibuf_bpp_out,
                           int *obuf_bpp_out)
{
    /* Input side */
    if (in_precision == 8) {
        if (ibuf_packed_bytes != n_in_elements) return -1;
        *pack_out = finn_cnn_pack_uint8;
    } else if (in_precision == 4) {
        if (2 * ibuf_packed_bytes == n_in_elements) {
            *pack_out = finn_cnn_pack_uint4;       /* 2-per-byte */
        } else if (ibuf_packed_bytes == n_in_elements) {
            *pack_out = finn_cnn_pack_uint4_1pb;   /* 1-per-byte */
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    /* Output side — chosen independently from input. */
    if (out_precision == 8) {
        if (obuf_packed_bytes != n_out_elements) return -1;
        *unpack_out = finn_cnn_unpack_uint8;
    } else if (out_precision == 4) {
        if (2 * obuf_packed_bytes == n_out_elements) {
            *unpack_out = finn_cnn_unpack_uint4;
        } else if (obuf_packed_bytes == n_out_elements) {
            *unpack_out = finn_cnn_unpack_uint4_1pb;
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    if (img_pixels <= 0 || fpga_out_pixels <= 0)               return -1;
    if (ibuf_packed_bytes % img_pixels != 0)                   return -1;
    if (obuf_packed_bytes % fpga_out_pixels != 0)              return -1;
    *ibuf_bpp_out = ibuf_packed_bytes / img_pixels;
    *obuf_bpp_out = obuf_packed_bytes / fpga_out_pixels;
    return 0;
}

int finn_cnn_runner_init(
    int  in_precision,              /* FPGA-input bitwidth (4 or 8) */
    int  out_precision,             /* FPGA-output bitwidth (4 or 8) */
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
    const float *add,
    int          cpu_post_maxpool_k,
    int          ibuf_packed_bytes,
    int          obuf_packed_bytes,
    int          partition,         /* 0 = classic, 1 = qi */
    int          idt_signed)        /* 0 = unsigned IDT, 1 = signed (QI only) */
{
    /* Scope validation: img_c ∈ {1, 3} (MNIST, CIFAR-10), 3x3 kernel,
     * pad=1. cpu_pre / cpu_pre_qi loops are generic over img_c. */
    if (img_c != 1 && img_c != 3)              return -2;
    if (kernel_size != 3 || pad != 1)          return -3;
    if (img_h <= 0 || img_w <= 0)              return -4;
    if (fpga_in_c <= 0 || fpga_out_h <= 0 ||
        fpga_out_w <= 0 || fpga_out_c <= 0)    return -5;
    if (num_classes <= 0 || num_thresholds <= 0) return -6;
    if (ibuf_virt == NULL || obuf_virt == NULL)  return -7;
    if (partition != 0 && partition != 1)      return -15;
    if (idt_signed != 0 && idt_signed != 1)    return -16;
    /* Classic deploys in this project always have UINT IDT (post-ReLU
     * uint8). QI IDT may be signed (QuantIdentity(Int8ActPerTensorFloat))
     * or unsigned (Uint8ActPerTensorFloat). */
    if (partition == 0 && idt_signed != 0)     return -16;
    /* W_conv is unused on the QI path (Conv1 is on FPGA). All other
     * weight pointers are required for both partitions. */
    if (thres == NULL || W_cls == NULL || add == NULL)  return -8;
    if (partition == 0 && W_conv == NULL)               return -8;

    /* CPU post-FPGA MaxPool: 0 (none) or 2 (2x2 stride-2 valid-pad). */
    if (cpu_post_maxpool_k != 0 && cpu_post_maxpool_k != 2) return -12;
    if (cpu_post_maxpool_k == 2 &&
        (fpga_out_h < 2 || fpga_out_w < 2))                 return -13;
    int gap_h, gap_w;
    if (cpu_post_maxpool_k == 0) {
        gap_h = fpga_out_h;
        gap_w = fpga_out_w;
    } else {
        gap_h = (fpga_out_h - 2) / 2 + 1;
        gap_w = (fpga_out_w - 2) / 2 + 1;
    }

    pack_fn_t   pack;
    unpack_fn_t unpack;
    int         ibuf_bpp, obuf_bpp;
    const int   n_in_elements  = img_h * img_w * fpga_in_c;
    const int   n_out_elements = fpga_out_h * fpga_out_w * fpga_out_c;
    const int   img_pixels     = img_h * img_w;
    const int   fpga_out_pixels = fpga_out_h * fpga_out_w;
    if (ibuf_packed_bytes <= 0 || obuf_packed_bytes <= 0)      return -14;
    if (select_dispatch(in_precision, out_precision,
                        n_in_elements,  ibuf_packed_bytes,
                        n_out_elements, obuf_packed_bytes,
                        img_pixels,     fpga_out_pixels,
                        &pack, &unpack,
                        &ibuf_bpp, &obuf_bpp) != 0) return -9;

    /* First-row spot-check of thresholds (FINN emits ascending). */
    for (int j = 1; j < num_thresholds; j++) {
        if (thres[j] < thres[j - 1]) return -10;
    }

    const int patch_dim = kernel_size * kernel_size * img_c;
    const size_t img_n      = (size_t)img_h * img_w * img_c;
    const size_t patches_n  = (size_t)img_h * img_w * patch_dim;
    const size_t acc_n      = (size_t)img_h * img_w * fpga_in_c;
    const size_t act_n      = (size_t)img_h * img_w * fpga_in_c;
    const size_t obuf_un_n  = (size_t)fpga_out_h * fpga_out_w * fpga_out_c;

    float   *img_f32       = (float   *)malloc(img_n      * sizeof(float));
    float   *patches_f32   = (float   *)malloc(patches_n  * sizeof(float));
    float   *acc_f32       = (float   *)malloc(acc_n      * sizeof(float));
    uint8_t *act_u8        = (uint8_t *)malloc(act_n);
    uint8_t *obuf_unpacked = (uint8_t *)malloc(obuf_un_n);
    uint8_t *gap_in_u8     = NULL;
    if (cpu_post_maxpool_k > 0) {
        gap_in_u8 = (uint8_t *)malloc((size_t)gap_h * gap_w * fpga_out_c);
    }
    if (!img_f32 || !patches_f32 || !acc_f32 || !act_u8 || !obuf_unpacked ||
        (cpu_post_maxpool_k > 0 && !gap_in_u8)) {
        free(img_f32); free(patches_f32); free(acc_f32);
        free(act_u8); free(obuf_unpacked); free(gap_in_u8);
        return -11;
    }

    memset(&g_cnn, 0, sizeof(g_cnn));
    g_cnn.ibuf_virt[0]   = ibuf_virt;
    g_cnn.ibuf_phys[0]   = ibuf_phys;
    g_cnn.obuf_virt[0]   = obuf_virt;
    g_cnn.obuf_phys[0]   = obuf_phys;
    /* Slot 1 mirrors slot 0 until set_second_buffers bumps n_buffers. */
    g_cnn.ibuf_virt[1]   = ibuf_virt;
    g_cnn.ibuf_phys[1]   = ibuf_phys;
    g_cnn.obuf_virt[1]   = obuf_virt;
    g_cnn.obuf_phys[1]   = obuf_phys;
    g_cnn.n_buffers      = 1;
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
    g_cnn.partition      = partition;
    g_cnn.idt_signed     = idt_signed;
    g_cnn.in_precision   = in_precision;
    g_cnn.out_precision  = out_precision;
    g_cnn.ibuf_bpp       = ibuf_bpp;
    g_cnn.obuf_bpp       = obuf_bpp;
    g_cnn.ibuf_bytes     = img_h * img_w * ibuf_bpp;
    g_cnn.obuf_bytes     = fpga_out_h * fpga_out_w * obuf_bpp;
    g_cnn.pack           = pack;
    g_cnn.unpack         = unpack;
    g_cnn.W_conv         = W_conv;
    g_cnn.thres          = thres;
    g_cnn.W_cls          = W_cls;
    g_cnn.mul            = mul;
    g_cnn.add            = add;
    g_cnn.img_f32        = img_f32;
    g_cnn.patches_f32    = patches_f32;
    g_cnn.acc_f32        = acc_f32;
    g_cnn.act_u8         = act_u8;
    g_cnn.obuf_unpacked  = obuf_unpacked;
    g_cnn.gap_in_u8      = gap_in_u8;
    g_cnn.cpu_post_maxpool_k = cpu_post_maxpool_k;
    g_cnn.gap_h          = gap_h;
    g_cnn.gap_w          = gap_w;
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
    free(g_cnn.obuf_unpacked);
    free(g_cnn.gap_in_u8);    /* NULL when k=0 — free(NULL) is a no-op */
    memset(&g_cnn, 0, sizeof(g_cnn));
    return 0;
}

/* Optional second-buffer setter for double-buffered batch inference.
 * Caller (Python) allocates the second ibuf/obuf pair (same shape +
 * cacheable as slot 0) and registers them here. After this returns
 * success, finn_cnn_infer_batch overlaps cpu_pre+pack[N+1] with
 * accel[N] and unpack+cpu_post[N-1]. Single-image entries
 * (finn_cnn_infer_one, _profiled) always use slot 0 and are
 * unaffected. Returns 0 on success, negative on misuse. */
int finn_cnn_set_second_buffers(
    void *ibuf_b_virt, uint64_t ibuf_b_phys,
    void *obuf_b_virt, uint64_t obuf_b_phys)
{
    if (!g_cnn.initialized)                              return -1;
    if (ibuf_b_virt == NULL || obuf_b_virt == NULL)      return -2;
    g_cnn.ibuf_virt[1] = ibuf_b_virt;
    g_cnn.ibuf_phys[1] = ibuf_b_phys;
    g_cnn.obuf_virt[1] = obuf_b_virt;
    g_cnn.obuf_phys[1] = obuf_b_phys;
    g_cnn.n_buffers    = 2;
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
 * CPU pre-stage (QI partition): cast + per-channel input MultiThreshold
 * on raw image. No im2col, no first MatMul — the QuantIdentity moved
 * Conv1 onto the FPGA, so the CPU just quantizes the float-normalized
 * pixel and packs the same byte the post-Conv1 ReLU would have produced.
 * Thresholds layout: g_cnn.thres is (img_c, num_thresholds), broadcast
 * across all (y,x) positions.
 * ============================================================ */

static inline void cpu_pre_qi(const uint8_t *img)
{
    const int H = g_cnn.img_h;
    const int W = g_cnn.img_w;
    const int C = g_cnn.img_c;
    const int Fc = g_cnn.fpga_in_c;     /* equals C for the QI path */
    const int T  = g_cnn.num_thresholds;
    const float *TH  = g_cnn.thres;
    uint8_t *act     = g_cnn.act_u8;
    const float inv255 = 1.0f / 255.0f;
    /* For signed IDT: shift the raw count [0, 2^N-1] down by 2^(N-1)
     * so the resulting byte, reinterpreted as int8 by the FPGA, matches
     * the MultiThreshold's signed output [-2^(N-1), 2^(N-1) - 1]. The
     * cast keeps the byte representation correct for both signs. */
    const int bias = g_cnn.idt_signed ? (1 << (g_cnn.in_precision - 1)) : 0;

    for (int yx = 0; yx < H * W; yx++) {
        for (int c = 0; c < C; c++) {
            float v = (float)img[yx * C + c] * inv255;
            const float *row = TH + (size_t)c * T;
            int count = mt_upper_bound(row, T, v);
            act[yx * Fc + c] = (uint8_t)(count - bias);
        }
    }
}

/* ============================================================
 * CPU post-stage: GAP + second MatMul + dequant + bias + argmax.
 * Consumes a 1-byte-per-channel buffer (g_cnn.obuf_unpacked when
 * cpu_post_maxpool_k=0; g_cnn.gap_in_u8 when k=2 — the caller has already
 * applied the CPU MaxPool and reduced the spatial dims to gap_h/gap_w).
 * At INT4 the unpack expanded packed nibbles into the same layout, so
 * this loop is precision-agnostic — values are 0..255 (INT8) or 0..15 (INT4).
 * ============================================================ */

static inline int cpu_post_argmax(const uint8_t *obuf_u8)
{
    const int OH = g_cnn.gap_h;
    const int OW = g_cnn.gap_w;
    const int OC = g_cnn.fpga_out_c;
    const int K  = g_cnn.num_classes;
    const float *Wcls = g_cnn.W_cls;
    const float *add  = g_cnn.add;
    const float mul   = g_cnn.mul;

    /* GAP with uint32 accumulator. INT8: 49*255=12,495; INT4: 49*15=735.
     * Both << 2^32, so the accumulator can't overflow. */
    uint32_t acc[OC];
    for (int c = 0; c < OC; c++) acc[c] = 0;
    for (int i = 0; i < OH * OW; i++) {
        const uint8_t *row = obuf_u8 + (size_t)i * OC;
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

static inline void trigger_dma(int slot)
{
    volatile uint32_t *idma = (volatile uint32_t *)g_cnn.idma_mmio;
    volatile uint32_t *odma = (volatile uint32_t *)g_cnn.odma_mmio;
    uint64_t ip = g_cnn.ibuf_phys[slot];
    uint64_t op = g_cnn.obuf_phys[slot];

    /* Same order as v1 / driver.py: arm output DMA before input. */
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

    /* Element counts (independent of precision). */
    const int n_in  = g_cnn.img_h     * g_cnn.img_w     * g_cnn.fpga_in_c;
    const int n_out = g_cnn.fpga_out_h * g_cnn.fpga_out_w * g_cnn.fpga_out_c;

    if (g_cnn.partition == 1) cpu_pre_qi(img); else cpu_pre(img);
    g_cnn.pack(g_cnn.act_u8, (uint8_t *)g_cnn.ibuf_virt[0], n_in);
    if (g_cnn.use_cache_ops) dcache_clean(g_cnn.ibuf_virt[0], g_cnn.ibuf_bytes);

    trigger_dma(0);
    wait_dma();

    if (g_cnn.use_cache_ops) dcache_invalidate(g_cnn.obuf_virt[0], g_cnn.obuf_bytes);
    g_cnn.unpack((const uint8_t *)g_cnn.obuf_virt[0], g_cnn.obuf_unpacked, n_out);

    const uint8_t *gap_input;
    if (g_cnn.cpu_post_maxpool_k > 0) {
        cpu_maxpool_2x2(g_cnn.obuf_unpacked, g_cnn.gap_in_u8,
                        g_cnn.fpga_out_w, g_cnn.gap_h, g_cnn.gap_w,
                        g_cnn.fpga_out_c);
        gap_input = g_cnn.gap_in_u8;
    } else {
        gap_input = g_cnn.obuf_unpacked;
    }
    return cpu_post_argmax(gap_input);
}

/* Double-buffered batch loop. Pipelines:
 *
 *   slot   = current FPGA's slot
 *   nslot  = other slot (1-slot), used to stage next input
 *
 *   prologue: cpu_pre+pack[0] -> ibuf[0], trigger DMA(0)
 *   loop i=1..N-1:
 *     cpu_pre+pack[i]  -> ibuf[nslot]   (overlaps with current DMA)
 *     wait_dma()                          (current DMA done)
 *     trigger DMA(nslot)                  (next FPGA work starts)
 *     unpack+cpu_post[slot] -> pred[i-1]  (overlaps with the new DMA)
 *     swap
 *   epilogue: wait final DMA, unpack+cpu_post[slot] -> pred[N-1]
 *
 * obuf_unpacked / gap_in_u8 are CPU-only scratch and used sequentially
 * in the cpu_post step — single instances are fine. The act_u8/patches
 * scratch is overwritten per cpu_pre call, then immediately consumed
 * by pack — also fine with one instance. Only the FPGA-touchable
 * ibuf/obuf need to be doubled.
 */
static int infer_batch_double(
    const uint8_t *images, const int32_t *labels,
    int n_samples, int32_t *predictions_out)
{
    if (n_samples == 0) return 0;
    const int IN    = g_cnn.img_h * g_cnn.img_w * g_cnn.img_c;
    const int n_in  = g_cnn.img_h * g_cnn.img_w * g_cnn.fpga_in_c;
    const int n_out = g_cnn.fpga_out_h * g_cnn.fpga_out_w * g_cnn.fpga_out_c;
    int correct = 0;

    /* For batch-of-1 the optimization has nothing to overlap; use the
     * lockstep single path so timing is comparable. */
    if (n_samples == 1) {
        int pred = finn_cnn_infer_one(images);
        predictions_out[0] = (int32_t)pred;
        return (labels && pred == labels[0]) ? 1 : 0;
    }

    /* Helper: cpu_pre + pack into slot's ibuf + dcache clean. */
    #define STAGE_INTO(slot_, img_ptr_) do { \
        if (g_cnn.partition == 1) cpu_pre_qi(img_ptr_); else cpu_pre(img_ptr_); \
        g_cnn.pack(g_cnn.act_u8, (uint8_t *)g_cnn.ibuf_virt[(slot_)], n_in); \
        if (g_cnn.use_cache_ops) \
            dcache_clean(g_cnn.ibuf_virt[(slot_)], g_cnn.ibuf_bytes); \
    } while (0)

    /* Helper: cache invalidate + unpack(slot's obuf) + cpu_post -> pred. */
    #define TAIL_FROM(slot_, pred_lvalue_) do { \
        if (g_cnn.use_cache_ops) \
            dcache_invalidate(g_cnn.obuf_virt[(slot_)], g_cnn.obuf_bytes); \
        g_cnn.unpack((const uint8_t *)g_cnn.obuf_virt[(slot_)], \
                      g_cnn.obuf_unpacked, n_out); \
        const uint8_t *gap_input_; \
        if (g_cnn.cpu_post_maxpool_k > 0) { \
            cpu_maxpool_2x2(g_cnn.obuf_unpacked, g_cnn.gap_in_u8, \
                            g_cnn.fpga_out_w, g_cnn.gap_h, g_cnn.gap_w, \
                            g_cnn.fpga_out_c); \
            gap_input_ = g_cnn.gap_in_u8; \
        } else { \
            gap_input_ = g_cnn.obuf_unpacked; \
        } \
        (pred_lvalue_) = cpu_post_argmax(gap_input_); \
    } while (0)

    /* Prologue: stage sample 0, kick off the first DMA. */
    int slot = 0;
    STAGE_INTO(slot, images);
    trigger_dma(slot);

    /* Steady state. At loop top, `slot` holds the in-flight DMA's slot. */
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

    /* Epilogue: wait for the last DMA and tail it. */
    wait_dma();
    int pred;
    TAIL_FROM(slot, pred);
    predictions_out[n_samples - 1] = (int32_t)pred;
    if (labels && pred == labels[n_samples - 1]) correct++;

    #undef STAGE_INTO
    #undef TAIL_FROM
    return correct;
}

int finn_cnn_infer_batch(
    const uint8_t *images,
    const int32_t *labels,
    int            n_samples,
    int32_t       *predictions_out)
{
    if (!g_cnn.initialized) return -1;
    if (n_samples < 0) return -2;

    /* Dispatch on the second-buffer registration. Single-buffer path is
     * byte-identical to the pre-double-buffer code (lockstep loop). */
    if (g_cnn.n_buffers == 2) {
        return infer_batch_double(images, labels, n_samples, predictions_out);
    }
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
 * (into caller-provided scratch, may be NULL) + unpack(mock_obuf) +
 * cpu_post.
 *
 * mock_obuf is the *packed* FPGA output bytes the caller would have
 * read off DMA — INT8: ohw*OC bytes, INT4: ohw*OC/2 bytes. The mock
 * runs the same unpack the real path runs so test_end_to_end and the
 * INT4 GAP tests exercise the same code path.
 * ============================================================ */

int finn_cnn_infer_one_mock(
    const uint8_t *img,
    const uint8_t *mock_obuf,
    uint8_t       *pack_scratch_out)
{
    if (!g_cnn.initialized) return -1;

    const int n_in  = g_cnn.img_h     * g_cnn.img_w     * g_cnn.fpga_in_c;
    const int n_out = g_cnn.fpga_out_h * g_cnn.fpga_out_w * g_cnn.fpga_out_c;

    if (g_cnn.partition == 1) cpu_pre_qi(img); else cpu_pre(img);
    if (pack_scratch_out) {
        g_cnn.pack(g_cnn.act_u8, pack_scratch_out, n_in);
    }
    g_cnn.unpack(mock_obuf, g_cnn.obuf_unpacked, n_out);

    const uint8_t *gap_input;
    if (g_cnn.cpu_post_maxpool_k > 0) {
        cpu_maxpool_2x2(g_cnn.obuf_unpacked, g_cnn.gap_in_u8,
                        g_cnn.fpga_out_w, g_cnn.gap_h, g_cnn.gap_w,
                        g_cnn.fpga_out_c);
        gap_input = g_cnn.gap_in_u8;
    } else {
        gap_input = g_cnn.obuf_unpacked;
    }
    return cpu_post_argmax(gap_input);
}

/* ============================================================
 * Profiled entry: 11-stage nanosecond breakdown.
 *
 * Stages (must match the harness):
 *   0 = cast + normalize
 *   1 = im2col
 *   2 = first MatMul
 *   3 = MultiThreshold
 *   4 = pack (+ dc cvac when use_cache_ops)
 *   5 = DMA trigger + wait
 *   6 = dc civac (when use_cache_ops) + unpack (memcpy at INT8,
 *       nibble-extract at INT4; result lives in g_cnn.obuf_unpacked)
 *   7 = CPU 2x2 MaxPool (only when cpu_post_maxpool_k > 0; otherwise this
 *       stage's elapsed time is a few ns of pointer setup — leave the slot
 *       in place so the harness's stage_names list has a stable length)
 *   8 = GAP (reads obuf_unpacked when k=0, gap_in_u8 when k>0; precision-agnostic)
 *   9 = second MatMul
 *  10 = dequant + bias + argmax
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
    const int GH = g_cnn.gap_h;        /* GAP input H: == OH when k=0, OH/2 when k=2 */
    const int GW = g_cnn.gap_w;        /* GAP input W: == OW when k=0, OW/2 when k=2 */
    const int K  = g_cnn.cpu_post_maxpool_k;
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
    /* PD/Wc/im/patches/acc/T are unused on the QI path; reference
     * them so -Wunused-variable doesn't fire at the QI builds. */
    (void)PD; (void)Wc; (void)im; (void)patches; (void)acc;

    uint64_t t[12];
    t[0] = mono_ns();

    if (g_cnn.partition == 1) {
        /* QI partition: stages 0-2 don't exist (no float-image scratch,
         * no im2col, no MatMul1). All input-quant work happens in stage 3.
         * Zero-pad the early stages so the harness's stage_names list
         * keeps a stable 11-slot layout — benchmark.py supplies a
         * partition-aware stage_names that labels 0-2 as "(qi noop)". */
        t[1] = t[0];
        t[2] = t[0];
        t[3] = t[0];
        cpu_pre_qi(img);
        t[4] = mono_ns();
    } else {
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
    }

    /* Stage 4: pack (+ cache clean) */
    const int n_in  = H * W * Fc;
    const int n_out = OH * OW * OC;
    g_cnn.pack(act, (uint8_t *)g_cnn.ibuf_virt[0], n_in);
    if (g_cnn.use_cache_ops) dcache_clean(g_cnn.ibuf_virt[0], g_cnn.ibuf_bytes);
    t[5] = mono_ns();

    /* Stage 5: DMA trigger + wait */
    trigger_dma(0);
    wait_dma();
    t[6] = mono_ns();

    /* Stage 6: cache invalidate + unpack (INT8: memcpy; INT4: nibble extract) */
    if (g_cnn.use_cache_ops) dcache_invalidate(g_cnn.obuf_virt[0], g_cnn.obuf_bytes);
    g_cnn.unpack((const uint8_t *)g_cnn.obuf_virt[0], g_cnn.obuf_unpacked, n_out);
    t[7] = mono_ns();

    /* Stage 7: CPU 2x2 MaxPool (only when k>0). When k=0 this is a few ns of
     * pointer setup so the harness sees a stable 11-slot stage layout. */
    const uint8_t *ob;
    if (K > 0) {
        cpu_maxpool_2x2(g_cnn.obuf_unpacked, g_cnn.gap_in_u8, OW, GH, GW, OC);
        ob = g_cnn.gap_in_u8;
    } else {
        ob = g_cnn.obuf_unpacked;
    }
    t[8] = mono_ns();

    /* Stage 8: GAP (reads ob, 1 byte/channel, value 0..255 or 0..15) */
    uint32_t gap_acc[OC];
    for (int c = 0; c < OC; c++) gap_acc[c] = 0;
    for (int i = 0; i < GH * GW; i++) {
        const uint8_t *row = ob + (size_t)i * OC;
        for (int c = 0; c < OC; c++) gap_acc[c] += row[c];
    }
    float feat[OC];
    for (int c = 0; c < OC; c++) feat[c] = (float)gap_acc[c] / (float)(GH * GW);
    t[9] = mono_ns();

    /* Stage 9: second MatMul */
    float logits[NC];
    for (int o = 0; o < NC; o++) logits[o] = 0.0f;
    for (int c = 0; c < OC; c++) {
        const float v = feat[c];
        const float *wr = Wcls + (size_t)c * NC;
        for (int o = 0; o < NC; o++) logits[o] += v * wr[o];
    }
    t[10] = mono_ns();

    /* Stage 10: dequant + bias + argmax */
    int best_idx = 0;
    float best_v = logits[0] * mul + add[0];
    for (int o = 1; o < NC; o++) {
        float v = logits[o] * mul + add[o];
        if (v > best_v) { best_v = v; best_idx = o; }
    }
    t[11] = mono_ns();

    for (int i = 0; i < 11; i++) ns_out[i] = t[i + 1] - t[i];
    return best_idx;
}
