/*
 * vta_infer.c — Board-side VTA inference in C (MLP and CNN).
 *
 * Removes all Python overhead from the inference loop.
 * Uses TVM's C runtime API for module loading and buffer management.
 * CPU-side quantization, im2col, maxpool, bias, ReLU in C with -O2.
 *
 * Supports:
 *   - MLP: flat input -> GEMM layers -> argmax
 *   - CNN: im2col -> GEMM (with o-tiling) -> maxpool -> avgpool -> dense
 *
 * Build on board:
 *   gcc -O2 -o vta_infer vta_infer.c \
 *       -I/home/xilinx/tvm-src/include \
 *       -I/home/xilinx/tvm-src/3rdparty/dlpack/include \
 *       -L/home/xilinx/tvm-src/build -ltvm_runtime \
 *       -ldl -lm -lpthread
 *
 * Usage:
 *   export LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH
 *   ./vta_infer <model_dir> <mnist_dir> [num_images] [num_runs] [output.json]
 *
 * Date: April 1, 2026 (updated from March 31 MLP-only version)
 */

#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdint.h>
#include <assert.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#define MAX_LAYERS 12
#define BLOCK_IN 16
#define BLOCK_OUT 16

/* Buffer sizes — sized for the size-sweep CNN topologies (cnn.py:get_cnn_config)
 * AND for the recognized-benchmark deployments (ResNet-8 on CIFAR-10).
 *
 *   Largest in_f  (ResNet-8 stage3 conv2, kernel=3, in_C=64): 3*3*64 = 576,
 *                 padded to 576 (multiple of BLOCK_IN=16). 1024 has margin.
 *   Largest out_f (cnn_large conv3, out_C=128 / ResNet-8 stage3 64): 128.
 *   Largest spatial (CIFAR-10 ResNet-8 stem output): 32*32 = 1024.
 *
 * Heap allocations scale linearly with these constants. ZU3 has 8 GB DDR;
 * total runtime memory still well under 20 MB.
 *
 * MAX_LAYERS=12 covers ResNet-8 (10 layers: 9 conv + 1 dense).
 * MAX_SPATIAL=1024 covers CIFAR-10 32*32 input (was 800 for MNIST 784).
 */
#define MAX_SPATIAL 1024  /* >= 32*32 = 1024 (CIFAR-10) */
#define MAX_FEATURES 1024 /* >= max padded in_f (cnn_large conv3 = 576) */
#define MAX_OUT_F 128     /* >= max padded out_f (cnn_large conv3 = 128) */
#define MAX_FLAT 1024     /* max flattened input for MLP */
#define MAX_LAYERS_P1 (MAX_LAYERS + 1)  /* act_scales has num_layers+1 entries sometimes */

/* ---- Simple .npy loader (handles int8 and float32) ---- */

typedef struct {
    void *data;
    int ndim;
    int64_t shape[8];
    char dtype[16];
    size_t elem_size;
    size_t total_elems;
} NpyArray;

static int npy_load(const char *path, NpyArray *arr) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); return -1; }
    if (magic[0] != 0x93 || magic[1] != 'N') {
        fprintf(stderr, "%s: not a .npy file\n", path);
        fclose(f); return -1;
    }

    int major = magic[6];
    uint32_t header_len;
    if (major == 1) {
        uint16_t hl;
        fread(&hl, 2, 1, f);
        header_len = hl;
    } else {
        fread(&header_len, 4, 1, f);
    }

    char *header = (char *)malloc(header_len + 1);
    fread(header, 1, header_len, f);
    header[header_len] = '\0';

    char *dp = strstr(header, "'descr': '");
    if (dp) {
        dp += 10;
        int i = 0;
        while (dp[i] && dp[i] != '\'' && i < 15) { arr->dtype[i] = dp[i]; i++; }
        arr->dtype[i] = '\0';
    }

    if (strcmp(arr->dtype, "<i1") == 0 || strcmp(arr->dtype, "|i1") == 0)
        arr->elem_size = 1;
    else if (strcmp(arr->dtype, "<f4") == 0)
        arr->elem_size = 4;
    else if (strcmp(arr->dtype, "<i4") == 0)
        arr->elem_size = 4;
    else if (strcmp(arr->dtype, "<f8") == 0)
        arr->elem_size = 8;
    else {
        fprintf(stderr, "%s: unsupported dtype '%s'\n", path, arr->dtype);
        free(header); fclose(f); return -1;
    }

    char *sp = strstr(header, "'shape': (");
    arr->ndim = 0;
    arr->total_elems = 1;
    if (sp) {
        sp += 10;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            int64_t dim = strtoll(sp, &sp, 10);
            arr->shape[arr->ndim++] = dim;
            arr->total_elems *= dim;
        }
    }

    free(header);
    arr->data = malloc(arr->total_elems * arr->elem_size);
    size_t rd = fread(arr->data, arr->elem_size, arr->total_elems, f);
    fclose(f);

    if (rd != arr->total_elems) {
        fprintf(stderr, "%s: expected %zu elems, got %zu\n", path, arr->total_elems, rd);
        free(arr->data); return -1;
    }
    return 0;
}

static void npy_free(NpyArray *arr) {
    if (arr->data) { free(arr->data); arr->data = NULL; }
}

/* ---- MNIST loader ---- */

static uint8_t *load_gz_file(const char *path, size_t *out_len) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "gunzip -c '%s' 2>/dev/null", path);
    FILE *p = popen(cmd, "r");
    if (!p) return NULL;
    size_t cap = 1 << 20;
    uint8_t *buf = (uint8_t *)malloc(cap);
    size_t len = 0, n;
    while ((n = fread(buf + len, 1, cap - len, p)) > 0) {
        len += n;
        if (len >= cap) { cap *= 2; buf = (uint8_t *)realloc(buf, cap); }
    }
    pclose(p);
    *out_len = len;
    return buf;
}

static int load_mnist_images(const char *path, float **images, int *n_images) {
    size_t len;
    uint8_t *raw = load_gz_file(path, &len);
    if (!raw || len < 16) { free(raw); return -1; }
    int num = (raw[4]<<24) | (raw[5]<<16) | (raw[6]<<8) | raw[7];
    int rows = (raw[8]<<24) | (raw[9]<<16) | (raw[10]<<8) | raw[11];
    int cols = (raw[12]<<24) | (raw[13]<<16) | (raw[14]<<8) | raw[15];
    int pixels = rows * cols;
    *n_images = num;
    *images = (float *)malloc(num * pixels * sizeof(float));
    for (int i = 0; i < num * pixels; i++)
        (*images)[i] = raw[16 + i] / 255.0f;
    free(raw);
    return 0;
}

static int load_mnist_labels(const char *path, uint8_t **labels, int *n_labels) {
    size_t len;
    uint8_t *raw = load_gz_file(path, &len);
    if (!raw || len < 8) { free(raw); return -1; }
    int num = (raw[4]<<24) | (raw[5]<<16) | (raw[6]<<8) | raw[7];
    *n_labels = num;
    *labels = (uint8_t *)malloc(num);
    memcpy(*labels, raw + 8, num);
    free(raw);
    return 0;
}

/* ---- CIFAR-10 loaders ----
 * Format produced by board/prepare_cifar10_for_board.py:
 *   cifar10_test_images.bin: N x H x W x C uint8, HWC (matches the layout
 *     im2col reads: x[si*W*C + sj*C + c]).
 *   cifar10_test_labels.bin: N uint8 labels in [0, 9].
 * Sizes are determined from file size + the H,W,C from the model config
 * (cnn_input_h/w/c, populated from input_shape in the JSON).
 */
static int load_cifar10_images(const char *path, int H, int W, int C,
                               float **images, int *n_images) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int per = H * W * C;
    if (sz <= 0 || sz % per != 0) {
        fprintf(stderr, "%s: size %ld not a multiple of %d (H*W*C)\n",
                path, sz, per);
        fclose(f);
        return -1;
    }
    int num = (int)(sz / per);
    uint8_t *raw = (uint8_t *)malloc(sz);
    if (fread(raw, 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "Short read on %s\n", path);
        free(raw); fclose(f);
        return -1;
    }
    fclose(f);
    *n_images = num;
    *images = (float *)malloc(num * per * sizeof(float));
    for (long i = 0; i < (long)num * per; i++)
        (*images)[i] = raw[i] / 255.0f;
    free(raw);
    return 0;
}

static int load_cifar10_labels(const char *path,
                               uint8_t **labels, int *n_labels) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    *n_labels = (int)sz;
    *labels = (uint8_t *)malloc(sz);
    if (fread(*labels, 1, sz, f) != (size_t)sz) {
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

/* ---- TVM helpers ---- */

#define TVM_CHECK(x) do { \
    int ret = (x); \
    if (ret != 0) { \
        fprintf(stderr, "TVM error at %s:%d: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
        exit(1); \
    } \
} while(0)

static DLTensor *alloc_vta_tensor(int64_t *shape, int ndim, DLDataType dtype) {
    DLTensor *t = NULL;
    DLDevice dev = {kDLExtDev, 0};
    TVM_CHECK(TVMArrayAlloc(shape, ndim, dtype.code, dtype.bits, dtype.lanes,
                            dev.device_type, dev.device_id, &t));
    return t;
}

static DLTensor *alloc_cpu_tensor(int64_t *shape, int ndim, DLDataType dtype) {
    DLTensor *t = NULL;
    DLDevice dev = {kDLCPU, 0};
    TVM_CHECK(TVMArrayAlloc(shape, ndim, dtype.code, dtype.bits, dtype.lanes,
                            dev.device_type, dev.device_id, &t));
    return t;
}

/* ---- CNN helper functions ---- */

/* ============================================================
 * Per-stage profiling — enabled on the first warmup image. Mirrors
 * FINN's runner pattern: one inference timed stage-by-stage, breakdown
 * printed once before the measured runs start. Stage names follow
 * "L<i>.<sub>" for per-layer stages (im2col, quant, gemm, dequant,
 * maxpool) plus "input_scale" / "GAP" / "dense_*" / "argmax" for
 * pre/post stages. Total wall time at the bottom doubles as a
 * steady-state FPS upper bound.
 * ============================================================ */
#define MAX_PROFILE_STAGES 48

static struct {
    char     name[32];
    uint64_t ns;
} _prof_stages[MAX_PROFILE_STAGES];
static int      _prof_n        = 0;
static int      _prof_active   = 0;
static uint64_t _prof_last_ns  = 0;

static inline uint64_t mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline void prof_init(void) {
    _prof_n        = 0;
    _prof_last_ns  = mono_ns();
}

static inline void prof_mark(const char *name) {
    if (!_prof_active) return;
    uint64_t now = mono_ns();
    if (_prof_n < MAX_PROFILE_STAGES) {
        snprintf(_prof_stages[_prof_n].name, sizeof(_prof_stages[_prof_n].name),
                 "%s", name);
        _prof_stages[_prof_n].ns = now - _prof_last_ns;
        _prof_n++;
    }
    _prof_last_ns = now;
}

static inline void prof_mark_layer(int layer_idx, const char *suffix) {
    if (!_prof_active) return;
    char buf[32];
    snprintf(buf, sizeof(buf), "L%d.%s", layer_idx, suffix);
    prof_mark(buf);
}

static void prof_print(void) {
    if (_prof_n == 0) return;
    uint64_t total = 0;
    for (int i = 0; i < _prof_n; i++) total += _prof_stages[i].ns;
    printf("\n[VTA timing breakdown — one inference]:\n");
    for (int i = 0; i < _prof_n; i++) {
        double us  = _prof_stages[i].ns / 1000.0;
        double pct = total > 0 ? 100.0 * _prof_stages[i].ns / total : 0;
        printf("  %-22s: %9.2f µs   (%5.1f %%)\n",
               _prof_stages[i].name, us, pct);
    }
    double total_us = total / 1000.0;
    double fps      = total > 0 ? 1e9 / total : 0;
    printf("  %-22s: %9.2f µs   (steady-state upper bound ~ %.0f FPS)\n",
           "total", total_us, fps);
    printf("\n");
}


static void im2col(const float *x_hwc, int H, int W, int C,
                   int kH, int kW, int pad, int stride,
                   float *patches, int *Ho_out, int *Wo_out)
{
    int Ho = (H + 2 * pad - kH) / stride + 1;
    int Wo = (W + 2 * pad - kW) / stride + 1;
    *Ho_out = Ho;
    *Wo_out = Wo;
    int patch_len = kH * kW * C;

    int idx = 0;
    for (int i = 0; i < Ho; i++) {
        for (int j = 0; j < Wo; j++) {
            int pidx = 0;
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int si = i * stride + ki - pad;
                    int sj = j * stride + kj - pad;
                    for (int c = 0; c < C; c++) {
                        if (si >= 0 && si < H && sj >= 0 && sj < W)
                            patches[idx * patch_len + pidx] = x_hwc[si * W * C + sj * C + c];
                        else
                            patches[idx * patch_len + pidx] = 0.0f;
                        pidx++;
                    }
                }
            }
            idx++;
        }
    }
}

static void maxpool2d(const float *x, int H, int W, int C, int ps, float *out) {
    int Ho = H / ps, Wo = W / ps;
    for (int i = 0; i < Ho; i++) {
        for (int j = 0; j < Wo; j++) {
            for (int c = 0; c < C; c++) {
                float mx = -1e30f;
                for (int pi = 0; pi < ps; pi++) {
                    for (int pj = 0; pj < ps; pj++) {
                        int si = i * ps + pi, sj = j * ps + pj;
                        float v = x[si * W * C + sj * C + c];
                        if (v > mx) mx = v;
                    }
                }
                out[i * Wo * C + j * C + c] = mx;
            }
        }
    }
}

static void global_avg_pool(const float *x, int H, int W, int C, float *out) {
    float scale = 1.0f / (H * W);
    for (int c = 0; c < C; c++) out[c] = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            for (int c = 0; c < C; c++)
                out[c] += x[i * W * C + j * C + c];
    for (int c = 0; c < C; c++) out[c] *= scale;
}

/* ---- CNN INT4-o8: CHW helpers -----------------------------------------
 * The INT4-o8 CNN pipeline uses CHW activation layout (matches Brevitas
 * and test_vta_cnn_int4_o8.py). im2col_chw produces patches in the same
 * (ki, kj, c) row order as the existing HWC im2col so the weight layout
 * is unchanged; only the SOURCE indexing differs.
 */
static void im2col_chw(const int8_t *x_chw, int C, int H, int W,
                       int kH, int kW, int pad, int pad_value,
                       int8_t *patches_out, int *Ho_out, int *Wo_out)
{
    int Ho = H + 2 * pad - kH + 1;
    int Wo = W + 2 * pad - kW + 1;
    int patch_len = kH * kW * C;
    *Ho_out = Ho;
    *Wo_out = Wo;
    int idx = 0;
    for (int i = 0; i < Ho; i++) {
        for (int j = 0; j < Wo; j++) {
            int pidx = 0;
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int si = i + ki - pad;
                    int sj = j + kj - pad;
                    for (int c = 0; c < C; c++) {
                        if (si >= 0 && si < H && sj >= 0 && sj < W)
                            patches_out[idx * patch_len + pidx] =
                                x_chw[c * H * W + si * W + sj];
                        else
                            patches_out[idx * patch_len + pidx] = (int8_t)pad_value;
                        pidx++;
                    }
                }
            }
            idx++;
        }
    }
}

static void maxpool2d_chw(const double *x, int C, int H, int W, int ps, double *out) {
    int Ho = H / ps, Wo = W / ps;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < Ho; i++) {
            for (int j = 0; j < Wo; j++) {
                double mx = -1e300;
                for (int pi = 0; pi < ps; pi++) {
                    for (int pj = 0; pj < ps; pj++) {
                        int si = i * ps + pi, sj = j * ps + pj;
                        double v = x[c * H * W + si * W + sj];
                        if (v > mx) mx = v;
                    }
                }
                out[c * Ho * Wo + i * Wo + j] = mx;
            }
        }
    }
}

/* ---- Layer config ---- */

typedef struct {
    int in_f, out_f, real_in, real_out;
    int n_tiles, m_tiles;
    /* m_compiled: m used at TVM/HLS compile time. Default == m_tiles (legacy
     * single-call). When < m_tiles (--force-m1 path), the runtime loops
     * (m_tiles / m_compiled) times per o-chunk, each call using one slice
     * of the tiled weights. Fixes the m>1 + n_chunks>1 zero-output bug
     * observed on ResNet-8 INT8 (same as session 23 INT4-o8 transformer
     * Issue 4). */
    int m_compiled;
    int o_total, o_tile, n_chunks;
    int shift;
    float w_scale;        /* scalar w_scale (INT8 CNN / MLP paths) */
    /* Per-channel w_scale (INT4-o8 CNN). NULL unless this layer has an
     * array-valued "w_scale" in config. Length = w_scale_n (= real_out). */
    float *w_scale_arr;
    int w_scale_n;
    char type[16];  /* "conv", "dense", or "mlp" */
    int kernel_size, padding, in_channels, out_channels, pool_size;
    int stride;           /* conv stride. Default 1 (cnn arch); ResNet-8
                           * uses stride 2 in stage2/stage3 first convs. */

    /* Skip-connection metadata (resnet8 arch; defaults preserve cnn behavior).
     *   consume_input_from: layer index whose post-dequant float output is the
     *                       input for this layer. -1 = use chain spatial_a.
     *   skip_add_from:      layer index whose post-dequant float output is
     *                       added to this layer's post-dequant output BEFORE
     *                       ReLU. -1 = no skip-add.
     *   branch_only:        1 = this layer's output feeds only a future
     *                           skip-add; do not update chain spatial_a.
     *   apply_relu:         1 = ReLU after dequant (and after skip-add).
     *                       Defaults to 1 for conv, 0 for dense.
     *   save_to_slot:       index into saved_acts[] where this layer's output
     *                       is stored (-1 if not saved). Computed at init from
     *                       which layers are referenced by future layers.
     */
    int consume_input_from;
    int skip_add_from;
    int branch_only;
    int apply_relu;
    int save_to_slot;

    /* Loaded data */
    int8_t *W_tiled;
    float *bias_float;    /* float bias for CPU-side (last layer or INT8 path) */
    int32_t *bias_int;    /* int32 bias for VTA ALU ADD (hidden layers, vta_native) */
    int has_vta_bias;     /* 1 = 4-arg module (bias in VTA), 0 = 3-arg */
    float in_scale;       /* learned activation scale (vta_native path) */

    /* TVM handles */
    TVMModuleHandle mod;
    TVMFunctionHandle func;
    DLTensor *A_dl, *B_dl, *C_dl, *D_dl;
    /* B_dl_chunks[k] is the weight tensor for m-chunk k (shape m_compiled,
     * n, BO, BI). Length = m_tiles / m_compiled. Legacy: single-element
     * array, B_dl_chunks[0] aliases B_dl. */
    DLTensor **B_dl_chunks;
} Layer;

/* Saved activation slot for residual / fork paths.
 * Each saved slot holds one layer's post-dequant (and post-ReLU if apply_relu)
 * spatial activation, plus its (H, W, C) shape and per-tensor scale.
 * Allocated lazily during init only for layers that appear in another layer's
 * consume_input_from or skip_add_from.
 */
typedef struct {
    float *data;   /* H*W*C floats; NULL if slot unused */
    int H, W, C;
    float scale;   /* max-abs / 127 of this saved tensor */
    int valid;     /* set per-image when populated; cleared per inference */
} SavedAct;

/* ---- INT4 nibble pack/unpack (flat across entire tensor) ---- */

static void pack_int4(const int8_t *vals, int8_t *out, int n) {
    int half = n / 2;
    for (int k = 0; k < half; k++) {
        uint8_t lo = (uint8_t)vals[2*k]   & 0x0F;
        uint8_t hi = (uint8_t)vals[2*k+1] & 0x0F;
        out[k] = (int8_t)((hi << 4) | lo);
    }
    memset(out + half, 0, n - half);
}

static void unpack_int4(const int8_t *packed, int8_t *out, int n) {
    int half = n / 2;
    for (int k = 0; k < half; k++) {
        uint8_t byte = (uint8_t)packed[k];
        int8_t lo = (int8_t)(byte & 0x0F);
        int8_t hi = (int8_t)((byte >> 4) & 0x0F);
        if (lo > 7) lo -= 16;
        if (hi > 7) hi -= 16;
        out[2*k]   = lo;
        out[2*k+1] = hi;
    }
    for (int k = half * 2; k < n; k++) out[k] = 0;
}

/* ---- Minimal JSON string field extraction ---- */

static int json_find_str(const char *json, const char *key, char *out, int max_len) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '"') {
        p++;
        int i = 0;
        while (*p && *p != '"' && i < max_len - 1) { out[i++] = *p++; }
        out[i] = '\0';
        return 0;
    }
    return -1;
}

static int json_find_int(const char *json, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return (int)strtol(p, NULL, 10);
}

static int json_find_bool(const char *json, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    /* Fall back to integer (handles 1/0) */
    return (int)strtol(p, NULL, 10);
}

static float json_find_float(const char *json, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char *p = strstr(json, pattern);
    if (!p) return 0.0f;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return strtof(p, NULL);
}

/* Parse a JSON float array: "key": [v1, v2, ...]. Returns count written
 * to out[], or -1 if key not found / not an array. Stops at max_len. */
static int json_find_float_array(const char *json, const char *key,
                                 float *out, int max_len)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    if (*p != '[') return -1;
    p++;
    int n = 0;
    while (*p && *p != ']' && n < max_len) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n') p++;
        if (*p == ']' || *p == '\0') break;
        char *end = NULL;
        float v = strtof(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
    }
    return n;
}

static int json_find_int_array(const char *json, const char *key,
                               int *out, int max_len)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    if (*p != '[') return -1;
    p++;
    int n = 0;
    while (*p && *p != ']' && n < max_len) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n') p++;
        if (*p == ']' || *p == '\0') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
    }
    return n;
}

/* Find the start of the Nth layer object in the "layers" array */
static const char *json_find_layer(const char *json, int layer_idx) {
    const char *p = strstr(json, "\"layers\"");
    if (!p) return NULL;
    p = strchr(p, '[');
    if (!p) return NULL;
    p++;
    for (int i = 0; i <= layer_idx; i++) {
        p = strchr(p, '{');
        if (!p) return NULL;
        if (i < layer_idx) {
            /* Skip to matching closing brace */
            int depth = 1;
            p++;
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                if (*p == '}') depth--;
                p++;
            }
        }
    }
    return p;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <mnist_dir> [num_images] [num_runs] [output.json]\n", argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];
    const char *mnist_dir = argv[2];
    int max_images = argc > 3 ? atoi(argv[3]) : 10000;
    int num_runs = argc > 4 ? atoi(argv[4]) : 3;
    const char *json_out = argc > 5 ? argv[5] : NULL;

    /* ---- Load VTA runtime ---- */
    void *vta_lib = dlopen("/home/xilinx/tvm-src/build/libvta.so", RTLD_NOW | RTLD_GLOBAL);
    if (!vta_lib) {
        fprintf(stderr, "Cannot load libvta.so: %s\n", dlerror());
        return 1;
    }
    printf("VTA runtime loaded\n");

    /* ---- Parse config.json ---- */
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *cf = fopen(path, "r");
    if (!cf) { fprintf(stderr, "Cannot open config.json\n"); return 1; }
    fseek(cf, 0, SEEK_END);
    long fsize = ftell(cf);
    fseek(cf, 0, SEEK_SET);
    char *json = (char *)malloc(fsize + 1);
    fread(json, 1, fsize, cf);
    json[fsize] = '\0';
    fclose(cf);

    /* Detect model type and requant mode */
    char model_type[32] = "mlp";
    json_find_str(json, "model_type", model_type, sizeof(model_type));
    int is_cnn = (strcmp(model_type, "cnn") == 0)
                 || (strcmp(model_type, "cnn_perchan_o8") == 0);
    int is_int4_o8 = (strcmp(model_type, "cnn_perchan_o8") == 0);

    char requant_mode[32] = "cpu_per_image";
    json_find_str(json, "requant_mode", requant_mode, sizeof(requant_mode));
    int is_vta_native = (strcmp(requant_mode, "vta_native") == 0);
    /* INT4-o8 CNN reuses the int4 weight pack + int32 VTA bias pipeline
     * originally added for MLP INT4. Flag covers both for the three
     * per-layer conditionals below (weight pack, bias load, D_dl alloc). */
    int use_int4_weights = is_vta_native || is_int4_o8;
    printf("Config: model_type='%s' requant_mode='%s' is_vta_native=%d "
           "is_int4_o8=%d is_cnn=%d\n",
           model_type, requant_mode, is_vta_native, is_int4_o8, is_cnn);

    /* INT4-o8 CNN: parse zero_point and the global activation-scale array. */
    int zero_point = 8;
    float act_scales[MAX_LAYERS_P1] = {0};
    int act_scales_n = 0;
    if (is_int4_o8) {
        int zp_parsed = json_find_int(json, "zero_point");
        if (zp_parsed > 0) zero_point = zp_parsed;
        act_scales_n = json_find_float_array(json, "act_scales_brevitas",
                                             act_scales, MAX_LAYERS_P1);
        if (act_scales_n <= 0) {
            fprintf(stderr, "FATAL: INT4-o8 CNN config missing act_scales_brevitas\n");
            return 1;
        }
        printf("INT4-o8: zero_point=%d act_scales_n=%d [", zero_point, act_scales_n);
        for (int k = 0; k < act_scales_n; k++) printf(" %.6f", act_scales[k]);
        printf(" ]\n");
    }

    float input_scale = 0;
    int input_clip_max = 127;
    if (is_vta_native) {
        input_scale = json_find_float(json, "input_scale");
        input_clip_max = json_find_int(json, "input_clip_max");
        if (input_scale <= 0) {
            /* Fall back to board_act_scales[0] if input_scale not present */
            /* For MLP INT4 v2: input_scale = 1/7 = 0.142857... */
            fprintf(stderr, "Warning: input_scale not found in config, "
                    "trying board_act_scales\n");
            /* Minimal: just use a known default */
            input_scale = 1.0f / 7.0f;
        }
        if (input_clip_max <= 0) input_clip_max = 7;
    }

    int num_layers = json_find_int(json, "num_layers");
    if (num_layers <= 0 || num_layers > MAX_LAYERS) {
        fprintf(stderr, "Invalid num_layers: %d\n", num_layers);
        return 1;
    }
    printf("Model type: %s, requant: %s, layers: %d\n",
           model_type, requant_mode, num_layers);

    /* ---- Parse and load layers ---- */
    Layer layers[MAX_LAYERS];
    DLDataType dtype_int8 = {kDLInt, 8, 1};

    for (int i = 0; i < num_layers; i++) {
        const char *lj = json_find_layer(json, i);
        if (!lj) { fprintf(stderr, "Cannot find layer %d in config\n", i); return 1; }

        /* Extract a substring for this layer (up to closing brace).
         *
         * Buffer must hold the largest formatted layer dict. INT4-o8 conv
         * layers carry per-channel w_scale + combined_scale arrays whose
         * size grows with out_f; cnn_large layer 2 (out_f=128) is ~7 KB
         * with indent=2, vs <500 B for the smallest INT8 layer. The
         * trailing keys (n_tiles, m_tiles) get truncated if the buffer
         * is too small, which silently produces -1 from json_find_int
         * and then -nan biases downstream — fail loud instead. */
        char layer_json[16384];
        int depth = 0;
        size_t lj_len = 0;
        const char *lp = lj;
        do {
            if (*lp == '{') depth++;
            if (*lp == '}') depth--;
            layer_json[lj_len++] = *lp++;
        } while (depth > 0 && lj_len < sizeof(layer_json) - 1);
        layer_json[lj_len] = '\0';
        if (depth != 0) {
            fprintf(stderr,
                    "FATAL: layer %d JSON truncated (%zu bytes hit buffer "
                    "limit before closing brace). Bump layer_json[] in "
                    "vta_infer.c.\n", i, lj_len);
            return 1;
        }

        layers[i].in_f = json_find_int(layer_json, "in_f");
        layers[i].out_f = json_find_int(layer_json, "out_f");
        layers[i].real_out = json_find_int(layer_json, "real_out");
        layers[i].n_tiles = json_find_int(layer_json, "n_tiles");
        layers[i].m_tiles = json_find_int(layer_json, "m_tiles");
        /* m_compiled defaults to m_tiles (legacy single-call). Present only
         * when --force-m1 was used at export time AND m_tiles > 1. */
        if (strstr(layer_json, "\"m_compiled\"")) {
            layers[i].m_compiled = json_find_int(layer_json, "m_compiled");
        } else {
            layers[i].m_compiled = layers[i].m_tiles;
        }
        if (layers[i].m_compiled < 1 || layers[i].m_tiles % layers[i].m_compiled != 0) {
            fprintf(stderr, "FATAL layer %d: m_tiles=%d not divisible by m_compiled=%d\n",
                    i, layers[i].m_tiles, layers[i].m_compiled);
            return 1;
        }
        layers[i].shift = json_find_int(layer_json, "shift");
        layers[i].w_scale = json_find_float(layer_json, "w_scale");
        layers[i].w_scale_arr = NULL;
        layers[i].w_scale_n = 0;
        layers[i].has_vta_bias = json_find_bool(layer_json, "has_vta_bias");
        layers[i].in_scale = json_find_float(layer_json, "in_scale");
        layers[i].D_dl = NULL;
        layers[i].bias_int = NULL;
        layers[i].bias_float = NULL;

        /* Skip-connection metadata. json_find_int returns 0 for missing keys,
         * so we use a sentinel-aware helper inline: read raw, then check via
         * json_find_bool-style markers. For simplicity we pre-init to defaults
         * and overwrite only when the field exists in the JSON. */
        layers[i].consume_input_from = -1;
        layers[i].skip_add_from = -1;
        layers[i].branch_only = 0;
        /* apply_relu defaults: 1 for conv, 0 for dense (matches legacy where
         * ReLU is hardcoded inside the conv branch and absent from dense). */
        layers[i].apply_relu = -1;  /* -1 = use type-based default later */
        layers[i].save_to_slot = -1;
        layers[i].stride = 1;
        if (strstr(layer_json, "\"consume_input_from\"")) {
            layers[i].consume_input_from = json_find_int(layer_json, "consume_input_from");
        }
        if (strstr(layer_json, "\"skip_add_from\"")) {
            layers[i].skip_add_from = json_find_int(layer_json, "skip_add_from");
        }
        if (strstr(layer_json, "\"branch_only\"")) {
            layers[i].branch_only = json_find_bool(layer_json, "branch_only");
        }
        if (strstr(layer_json, "\"apply_relu\"")) {
            layers[i].apply_relu = json_find_bool(layer_json, "apply_relu");
        }
        if (strstr(layer_json, "\"stride\"")) {
            layers[i].stride = json_find_int(layer_json, "stride");
            if (layers[i].stride < 1) layers[i].stride = 1;
        }

        /* INT4-o8 CNN: parse per-channel w_scale array. */
        if (is_int4_o8) {
            float tmp_w[MAX_OUT_F];
            int n = json_find_float_array(layer_json, "w_scale", tmp_w, MAX_OUT_F);
            if (n > 0) {
                layers[i].w_scale_arr = (float *)malloc(n * sizeof(float));
                memcpy(layers[i].w_scale_arr, tmp_w, n * sizeof(float));
                layers[i].w_scale_n = n;
            } else {
                fprintf(stderr, "FATAL: INT4-o8 layer %d missing array w_scale\n", i);
                return 1;
            }
        }

        /* CNN-specific fields */
        if (is_cnn) {
            json_find_str(layer_json, "type", layers[i].type, sizeof(layers[i].type));
            layers[i].o_total = json_find_int(layer_json, "o_total");
            layers[i].o_tile = json_find_int(layer_json, "o_tile");
            layers[i].n_chunks = json_find_int(layer_json, "n_chunks");
            layers[i].real_in = json_find_int(layer_json, "real_in");
            if (strcmp(layers[i].type, "conv") == 0) {
                layers[i].kernel_size = json_find_int(layer_json, "kernel_size");
                layers[i].padding = json_find_int(layer_json, "padding");
                layers[i].in_channels = json_find_int(layer_json, "in_channels");
                layers[i].out_channels = json_find_int(layer_json, "out_channels");
                layers[i].pool_size = json_find_int(layer_json, "pool");
            }
        } else {
            /* MLP: o=1, no tiling */
            strcpy(layers[i].type, "mlp");
            layers[i].o_total = 1;
            layers[i].o_tile = 1;
            layers[i].n_chunks = 1;
            layers[i].real_in = layers[i].in_f;
        }

        /* Apply type-based default for apply_relu when not explicitly set in
         * config. Legacy: conv always ReLU'd, dense never. ResNet-8 downsample
         * 1x1 layers explicitly set apply_relu=false. */
        if (layers[i].apply_relu < 0) {
            layers[i].apply_relu = (strcmp(layers[i].type, "conv") == 0) ? 1 : 0;
        }

        /* Load module */
        char mod_file[128];
        json_find_str(layer_json, "module_file", mod_file, sizeof(mod_file));
        /* Try .so first (linked), then .o.so (legacy), then .o */
        char mod_path[512], so_file[128];
        /* Replace .o with .so */
        strcpy(so_file, mod_file);
        char *dot_o = strstr(so_file, ".o");
        if (dot_o && dot_o[2] == '\0') { strcpy(dot_o, ".so"); }
        snprintf(mod_path, sizeof(mod_path), "%s/%s", model_dir, so_file);
        if (access(mod_path, F_OK) != 0) {
            snprintf(mod_path, sizeof(mod_path), "%s/%s.so", model_dir, mod_file);
            if (access(mod_path, F_OK) != 0) {
                snprintf(mod_path, sizeof(mod_path), "%s/%s", model_dir, mod_file);
            }
        }
        printf("  Layer %d (%s): %s  o=%d (tile=%d x%d) n=%d m=%d shift=%d\n",
               i, layers[i].type, so_file,
               layers[i].o_total, layers[i].o_tile, layers[i].n_chunks,
               layers[i].n_tiles, layers[i].m_tiles, layers[i].shift);

        TVM_CHECK(TVMModLoadFromFile(mod_path, "so", &layers[i].mod));
        TVM_CHECK(TVMModGetFunction(layers[i].mod, "my_gemm", 0, &layers[i].func));

        /* Load weights */
        char wfile[128], bfile[128];
        json_find_str(layer_json, "weight_file", wfile, sizeof(wfile));
        json_find_str(layer_json, "bias_file", bfile, sizeof(bfile));

        NpyArray warr, barr;
        snprintf(path, sizeof(path), "%s/%s", model_dir, wfile);
        if (npy_load(path, &warr) != 0) return 1;
        layers[i].W_tiled = (int8_t *)warr.data;

        snprintf(path, sizeof(path), "%s/%s", model_dir, bfile);
        if (npy_load(path, &barr) != 0) return 1;
        if (use_int4_weights && layers[i].has_vta_bias) {
            layers[i].bias_int = (int32_t *)barr.data;
            printf("    bias: int32 (%zu elems), first3=[%d, %d, %d]\n",
                   barr.total_elems,
                   layers[i].bias_int[0], layers[i].bias_int[1], layers[i].bias_int[2]);
        } else {
            layers[i].bias_float = (float *)barr.data;
            printf("    bias: float32 (%zu elems), first3=[%.4f, %.4f, %.4f]\n",
                   barr.total_elems,
                   layers[i].bias_float[0], layers[i].bias_float[1], layers[i].bias_float[2]);
        }

        /* Allocate VTA tensors (sized for o_tile).
         *
         * Layout: weights are tiled at full m_tiles; each "m-chunk" is a
         * (m_compiled, n, BO, BI) slice. For legacy single-call layers
         * (m_compiled == m_tiles) there is exactly one chunk equal to the
         * full tensor. For multi-call layers (m_compiled < m_tiles) we
         * allocate (m_tiles / m_compiled) separate B tensors.
         *
         * A/C tensors are sized for ONE m-chunk (m_compiled, not m_tiles)
         * because each VTA call processes only that chunk's outputs. */
        int ot = layers[i].o_tile;
        int nt = layers[i].n_tiles;
        int mt = layers[i].m_tiles;
        int mc = layers[i].m_compiled;
        int n_b_chunks = mt / mc;

        int64_t a_shape[] = {ot, nt, 1, BLOCK_IN};
        int64_t b_shape[] = {mc, nt, BLOCK_OUT, BLOCK_IN};
        int64_t c_shape[] = {ot, mc, 1, BLOCK_OUT};

        layers[i].A_dl = alloc_vta_tensor(a_shape, 4, dtype_int8);
        layers[i].C_dl = alloc_vta_tensor(c_shape, 4, dtype_int8);

        /* Per-chunk B tensors. */
        layers[i].B_dl_chunks = (DLTensor **)malloc(
            n_b_chunks * sizeof(DLTensor *));
        int chunk_b_elems = mc * nt * BLOCK_OUT * BLOCK_IN;
        for (int bc = 0; bc < n_b_chunks; bc++) {
            layers[i].B_dl_chunks[bc] = alloc_vta_tensor(b_shape, 4, dtype_int8);
            DLTensor *B_cpu = alloc_cpu_tensor(b_shape, 4, dtype_int8);
            int8_t *src_slice = layers[i].W_tiled + bc * chunk_b_elems;
            if (use_int4_weights) {
                int8_t *w_packed = (int8_t *)malloc(chunk_b_elems);
                pack_int4(src_slice, w_packed, chunk_b_elems);
                memcpy(B_cpu->data, w_packed, chunk_b_elems);
                free(w_packed);
            } else {
                memcpy(B_cpu->data, src_slice, chunk_b_elems);
            }
            TVM_CHECK(TVMArrayCopyFromTo(B_cpu, layers[i].B_dl_chunks[bc], NULL));
            TVMArrayFree(B_cpu);
        }
        /* Legacy alias: B_dl points to chunk 0. The MLP VTA_CALL macro reads
         * _vl->B_dl directly; for single-chunk layers (m_compiled==m_tiles)
         * this is the full tensor as before. */
        layers[i].B_dl = layers[i].B_dl_chunks[0];
        if (n_b_chunks == 1) {
            printf("    weights: %s (%d bytes, single B chunk)\n",
                   use_int4_weights ? "int4-packed" : "raw int8",
                   chunk_b_elems);
        } else {
            printf("    weights: %s (%d B chunks × %d bytes — m_compiled=%d, m_tiles=%d)\n",
                   use_int4_weights ? "int4-packed" : "raw int8",
                   n_b_chunks, chunk_b_elems, mc, mt);
        }

        /* Allocate and load int32 bias to VTA (INT4 hidden layers + all INT4-o8 CNN layers) */
        if (use_int4_weights && layers[i].has_vta_bias) {
            DLDataType dtype_int32 = {kDLInt, 32, 1};
            int64_t d_shape[] = {ot, mt, 1, BLOCK_OUT};
            layers[i].D_dl = alloc_vta_tensor(d_shape, 4, dtype_int32);
            DLTensor *D_cpu = alloc_cpu_tensor(d_shape, 4, dtype_int32);
            /* Broadcast bias (mt * BLOCK_OUT int32s) across o_tile rows */
            int32_t *d_data = (int32_t *)D_cpu->data;
            for (int r = 0; r < ot; r++) {
                memcpy(d_data + r * mt * BLOCK_OUT,
                       layers[i].bias_int,
                       mt * BLOCK_OUT * sizeof(int32_t));
            }
            TVM_CHECK(TVMArrayCopyFromTo(D_cpu, layers[i].D_dl, NULL));
            TVMArrayFree(D_cpu);
        }
    }

    /* ---- Compute save_to_slot per layer + allocate saved-activation pool ----
     * A layer needs saving if any later layer references it via
     * consume_input_from or skip_add_from. Slot index assignment is dense:
     * the i-th layer that needs saving gets slot i.
     * For legacy MNIST CNN configs (no skip metadata), no slots are needed
     * and the pool stays empty — preserving zero overhead on that path. */
    SavedAct saved_acts[MAX_LAYERS];
    for (int s = 0; s < MAX_LAYERS; s++) {
        saved_acts[s].data = NULL;
        saved_acts[s].H = saved_acts[s].W = saved_acts[s].C = 0;
        saved_acts[s].scale = 0;
        saved_acts[s].valid = 0;
    }
    int n_saved_slots = 0;
    for (int target = 0; target < num_layers; target++) {
        int referenced = 0;
        for (int j = 0; j < num_layers; j++) {
            if (layers[j].consume_input_from == target ||
                layers[j].skip_add_from == target) {
                referenced = 1;
                break;
            }
        }
        if (referenced) {
            layers[target].save_to_slot = n_saved_slots;
            /* Allocate buffer sized for this layer's full spatial output:
             * H*W*real_out for conv. The H,W are determined per-image at
             * runtime so allocate the upper bound (MAX_SPATIAL * MAX_OUT_F). */
            saved_acts[n_saved_slots].data =
                (float *)malloc(MAX_SPATIAL * MAX_OUT_F * sizeof(float));
            if (!saved_acts[n_saved_slots].data) {
                fprintf(stderr, "FATAL: saved_acts slot %d alloc failed\n",
                        n_saved_slots);
                return 1;
            }
            n_saved_slots++;
        }
    }
    if (n_saved_slots > 0) {
        printf("Saved-activation pool: %d slot(s) allocated\n", n_saved_slots);
    }

    /* ---- Parse input shape from config (defaults to MNIST 1x28x28) ----
     * Generalizes the hardcoded 28*28*1 input. The config "input_shape" is
     * [C, H, W] (PyTorch convention). For the legacy cnn-tiny-mnist case
     * with [1, 28, 28], cnn_input_h=28, cnn_input_w=28, cnn_input_c=1 →
     * identical loop bounds to the prior hardcoded path. */
    int cnn_input_h = 28, cnn_input_w = 28, cnn_input_c = 1;
    {
        const char *isj = strstr(json, "\"input_shape\"");
        if (isj) {
            int vals[4] = {0,0,0,0};
            int n = json_find_int_array(isj, "input_shape", vals, 4);
            if (n >= 3) {
                cnn_input_c = vals[0];
                cnn_input_h = vals[1];
                cnn_input_w = vals[2];
            }
        }
    }
    if (is_cnn) {
        printf("Input shape: %dx%dx%d (HWC)\n", cnn_input_h, cnn_input_w, cnn_input_c);
    }

    free(json);
    printf("All modules loaded\n");

    /* ---- Load test set: MNIST (1x28x28) or CIFAR-10 (32x32x3) ----
     * Dispatch on the model's input_shape (cnn_input_c). The data dir
     * argument is treated as MNIST-format when cnn_input_c==1 and as
     * the CIFAR-10 board binary directory (board/prepare_cifar10_for_board.py
     * output) when cnn_input_c==3. MLP path always uses MNIST. */
    float *images = NULL;
    uint8_t *labels = NULL;
    int n_images, n_labels;
    int use_cifar10 = is_cnn && cnn_input_c == 3;
    if (use_cifar10) {
        snprintf(path, sizeof(path), "%s/cifar10_test_images.bin", mnist_dir);
        if (load_cifar10_images(path, cnn_input_h, cnn_input_w, cnn_input_c,
                                &images, &n_images) != 0) return 1;
        snprintf(path, sizeof(path), "%s/cifar10_test_labels.bin", mnist_dir);
        if (load_cifar10_labels(path, &labels, &n_labels) != 0) return 1;
    } else {
        snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte.gz", mnist_dir);
        if (load_mnist_images(path, &images, &n_images) != 0) return 1;
        snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte.gz", mnist_dir);
        if (load_mnist_labels(path, &labels, &n_labels) != 0) return 1;
    }
    if (max_images < n_images) n_images = max_images;
    printf("Loaded %d %s test images\n", n_images,
           use_cifar10 ? "CIFAR-10" : "MNIST");

    /* ---- Allocate CPU-side transfer tensors ----
     * We need one pair (A_cpu, C_cpu) per layer since shapes differ.
     * Reuse approach: allocate largest needed.
     */
    int max_a_elems = 0, max_c_elems = 0;
    for (int i = 0; i < num_layers; i++) {
        int ae = layers[i].o_tile * layers[i].n_tiles * BLOCK_IN;
        int ce = layers[i].o_tile * layers[i].m_tiles * BLOCK_OUT;
        if (ae > max_a_elems) max_a_elems = ae;
        if (ce > max_c_elems) max_c_elems = ce;
    }
    int64_t a_max_shape[] = {max_a_elems, 1, 1, 1};  /* flat, reshape via shape pointer */
    int64_t c_max_shape[] = {max_c_elems, 1, 1, 1};
    DLDataType dt_i8 = {kDLInt, 8, 1};
    DLTensor *A_cpu = alloc_cpu_tensor(a_max_shape, 4, dt_i8);
    DLTensor *C_cpu = alloc_cpu_tensor(c_max_shape, 4, dt_i8);

    /* ---- Working buffers ---- */
    /* For CNN: im2col patches, spatial activations, etc. */
    float *patches_f = (float *)malloc(MAX_SPATIAL * MAX_FEATURES * sizeof(float));
    int8_t *patches_i8 = (int8_t *)malloc(MAX_SPATIAL * MAX_FEATURES);
    int8_t *vta_out_i8 = (int8_t *)malloc(MAX_SPATIAL * MAX_OUT_F);
    float *y_float = (float *)malloc(MAX_SPATIAL * MAX_OUT_F * sizeof(float));
    /* Spatial buffers — sized to hold the largest possible per-image
     * activation. For MNIST CNN that's 28*28*MAX_OUT_F; for CIFAR-10
     * ResNet-8 the stem output is 32*32*16 = 16K floats, well within
     * MAX_SPATIAL*MAX_OUT_F = 1024*128 = 128K floats = 512KB each. */
    float *spatial_a = (float *)malloc(MAX_SPATIAL * MAX_OUT_F * sizeof(float));
    float *spatial_b = (float *)malloc(MAX_SPATIAL * MAX_OUT_F * sizeof(float));

    /* INT4-o8 CNN CHW buffers (offset-encoded int8 activations and float64
     * per-channel dequant/pool workspace). Only allocated for the INT4-o8
     * path to avoid wasting memory on INT8 CNN / MLP runs. */
    int8_t *chw_buf_a = NULL;
    int8_t *chw_buf_b = NULL;
    double *chw_float_a = NULL;
    double *chw_float_b = NULL;
    if (is_int4_o8) {
        chw_buf_a   = (int8_t *)malloc(28 * 28 * MAX_OUT_F);
        chw_buf_b   = (int8_t *)malloc(28 * 28 * MAX_OUT_F);
        chw_float_a = (double *)malloc(28 * 28 * MAX_OUT_F * sizeof(double));
        chw_float_b = (double *)malloc(28 * 28 * MAX_OUT_F * sizeof(double));
        if (!chw_buf_a || !chw_buf_b || !chw_float_a || !chw_float_b) {
            fprintf(stderr, "FATAL: failed to allocate CHW buffers\n");
            return 1;
        }
    }

    /* For MLP */
    int8_t h_int8[MAX_FLAT];
    float h_float_mlp[MAX_FLAT];

    /* ---- VTA call helper ---- */
    #define VTA_CALL(layer_ptr, a_data, a_bytes, c_data, c_bytes) do { \
        Layer *_vl = (layer_ptr); \
        memcpy(A_cpu->data, (a_data), (a_bytes)); \
        A_cpu->shape[0] = _vl->o_tile; \
        A_cpu->shape[1] = _vl->n_tiles; \
        A_cpu->shape[2] = 1; \
        A_cpu->shape[3] = BLOCK_IN; \
        A_cpu->strides = NULL; \
        TVM_CHECK(TVMArrayCopyFromTo(A_cpu, _vl->A_dl, NULL)); \
        memset(C_cpu->data, 0, (c_bytes)); \
        C_cpu->shape[0] = _vl->o_tile; \
        C_cpu->shape[1] = _vl->m_tiles; \
        C_cpu->shape[2] = 1; \
        C_cpu->shape[3] = BLOCK_OUT; \
        C_cpu->strides = NULL; \
        TVM_CHECK(TVMArrayCopyFromTo(C_cpu, _vl->C_dl, NULL)); \
        TVMValue args[3]; \
        int type_codes[3] = {kTVMDLTensorHandle, kTVMDLTensorHandle, kTVMDLTensorHandle}; \
        args[0].v_handle = _vl->A_dl; \
        args[1].v_handle = _vl->B_dl; \
        args[2].v_handle = _vl->C_dl; \
        TVMValue rv; int rt; \
        TVM_CHECK(TVMFuncCall(_vl->func, args, type_codes, 3, &rv, &rt)); \
        TVM_CHECK(TVMArrayCopyFromTo(_vl->C_dl, C_cpu, NULL)); \
        memcpy((c_data), C_cpu->data, (c_bytes)); \
    } while(0)

    /* ---- 4-arg VTA call (GEMM + bias + SHR + CLIP) for vta_native ---- */
    #define VTA_CALL_BIAS(layer_ptr, a_data, a_bytes, c_data, c_bytes) do { \
        Layer *_vl = (layer_ptr); \
        memcpy(A_cpu->data, (a_data), (a_bytes)); \
        A_cpu->shape[0] = _vl->o_tile; \
        A_cpu->shape[1] = _vl->n_tiles; \
        A_cpu->shape[2] = 1; \
        A_cpu->shape[3] = BLOCK_IN; \
        A_cpu->strides = NULL; \
        TVM_CHECK(TVMArrayCopyFromTo(A_cpu, _vl->A_dl, NULL)); \
        memset(C_cpu->data, 0, (c_bytes)); \
        C_cpu->shape[0] = _vl->o_tile; \
        C_cpu->shape[1] = _vl->m_tiles; \
        C_cpu->shape[2] = 1; \
        C_cpu->shape[3] = BLOCK_OUT; \
        C_cpu->strides = NULL; \
        TVM_CHECK(TVMArrayCopyFromTo(C_cpu, _vl->C_dl, NULL)); \
        TVMValue args[4]; \
        int type_codes[4] = {kTVMDLTensorHandle, kTVMDLTensorHandle, \
                             kTVMDLTensorHandle, kTVMDLTensorHandle}; \
        args[0].v_handle = _vl->A_dl; \
        args[1].v_handle = _vl->B_dl; \
        args[2].v_handle = _vl->D_dl; \
        args[3].v_handle = _vl->C_dl; \
        TVMValue rv; int rt; \
        TVM_CHECK(TVMFuncCall(_vl->func, args, type_codes, 4, &rv, &rt)); \
        TVM_CHECK(TVMArrayCopyFromTo(_vl->C_dl, C_cpu, NULL)); \
        memcpy((c_data), C_cpu->data, (c_bytes)); \
    } while(0)

    /* ---- Inference functions ---- */

    /* MLP inference: returns predicted class */
    #define MLP_INFER(img_ptr, prediction) do { \
        prof_init(); \
        float *_img = (img_ptr); \
        float x_max = 0; \
        for (int _k = 0; _k < 784; _k++) { \
            float av = fabsf(_img[_k]); \
            if (av > x_max) x_max = av; \
        } \
        float x_s = (x_max > 0) ? x_max / 127.0f : 1e-10f; \
        for (int _k = 0; _k < 784; _k++) { \
            float v = roundf(_img[_k] / x_s); \
            h_int8[_k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); \
        } \
        prof_mark("input_quant"); \
        float current_scale = x_s; \
        for (int _li = 0; _li < num_layers; _li++) { \
            Layer *_l = &layers[_li]; \
            int _a_bytes = _l->n_tiles * BLOCK_IN; \
            int _c_bytes = _l->m_tiles * BLOCK_OUT; \
            int8_t _c_out[MAX_FLAT]; \
            VTA_CALL(_l, h_int8, _a_bytes, _c_out, _c_bytes); \
            prof_mark_layer(_li, "gemm"); \
            float combined = current_scale * _l->w_scale * (float)(1 << _l->shift); \
            for (int _j = 0; _j < _l->out_f; _j++) { \
                h_float_mlp[_j] = (float)_c_out[_j] * combined + _l->bias_float[_j]; \
            } \
            prof_mark_layer(_li, "dequant"); \
            if (_li < num_layers - 1) { \
                float y_max = 0; \
                for (int _j = 0; _j < _l->out_f; _j++) { \
                    if (h_float_mlp[_j] < 0) h_float_mlp[_j] = 0; \
                    float av = fabsf(h_float_mlp[_j]); \
                    if (av > y_max) y_max = av; \
                } \
                float ns = (y_max > 0) ? y_max / 127.0f : 1e-10f; \
                for (int _j = 0; _j < _l->out_f; _j++) { \
                    float v = roundf(h_float_mlp[_j] / ns); \
                    h_int8[_j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); \
                } \
                current_scale = ns; \
                prof_mark_layer(_li, "relu_requant"); \
            } else { \
                float best = h_float_mlp[0]; int best_idx = 0; \
                for (int _j = 1; _j < _l->real_out; _j++) { \
                    if (h_float_mlp[_j] > best) { best = h_float_mlp[_j]; best_idx = _j; } \
                } \
                (prediction) = best_idx; \
                prof_mark("argmax"); \
            } \
        } \
    } while(0)

    /* MLP inference: vta_native INT4 path */
    int8_t _packed_buf[MAX_FLAT];  /* reusable pack buffer */
    int8_t _unpacked_buf[MAX_FLAT];

    #define MLP_INFER_VTA_NATIVE(img_ptr, prediction) do { \
        prof_init(); \
        float *_img = (img_ptr); \
        /* Input quantize: fixed scale, clip [0, input_clip_max] */ \
        for (int _k = 0; _k < 784; _k++) { \
            float v = roundf(_img[_k] / input_scale); \
            h_int8[_k] = (int8_t)(v < 0 ? 0 : (v > input_clip_max ? input_clip_max : (int)v)); \
        } \
        prof_mark("input_quant"); \
        for (int _li = 0; _li < num_layers; _li++) { \
            Layer *_ln = &layers[_li]; \
            int _a_elems = _ln->n_tiles * BLOCK_IN; \
            int _c_elems = _ln->m_tiles * BLOCK_OUT; \
            /* Pack int4 input into nibble format */ \
            pack_int4(h_int8, _packed_buf, _a_elems); \
            prof_mark_layer(_li, "pack"); \
            int8_t _c_packed[MAX_FLAT]; \
            if (_ln->has_vta_bias) { \
                /* Hidden layer: 4-arg call. VTA does GEMM+bias+SHR+CLIP. */ \
                /* Output is packed int4 — unpack for next layer. */ \
                VTA_CALL_BIAS(_ln, _packed_buf, _a_elems, _c_packed, _c_elems); \
                prof_mark_layer(_li, "gemm"); \
                unpack_int4(_c_packed, _unpacked_buf, _c_elems); \
                memcpy(h_int8, _unpacked_buf, _ln->real_out); \
                prof_mark_layer(_li, "unpack"); \
            } else { \
                /* Last layer: 3-arg call. CPU dequant + float bias + argmax. */ \
                VTA_CALL(_ln, _packed_buf, _a_elems, _c_packed, _c_elems); \
                prof_mark_layer(_li, "gemm"); \
                unpack_int4(_c_packed, _unpacked_buf, _c_elems); \
                float combined = _ln->in_scale * _ln->w_scale \
                                 * (float)(1 << _ln->shift); \
                float best = -1e30f; int best_idx = 0; \
                for (int _j = 0; _j < _ln->real_out; _j++) { \
                    float val = (float)_unpacked_buf[_j] * combined \
                                + _ln->bias_float[_j]; \
                    if (val > best) { best = val; best_idx = _j; } \
                } \
                (prediction) = best_idx; \
                prof_mark("argmax"); \
            } \
        } \
    } while(0)

    /* CNN inference: returns predicted class.
     * img_input: cnn_input_h * cnn_input_w * cnn_input_c floats in HWC layout.
     * Legacy MNIST: 28*28*1 = 784. CIFAR-10 ResNet-8: 32*32*3 = 3072. */
    int cnn_infer(const float *img_input) {
        prof_init();
        int n_input = cnn_input_h * cnn_input_w * cnn_input_c;
        float x_max = 0;
        for (int k = 0; k < n_input; k++) {
            float av = fabsf(img_input[k]);
            if (av > x_max) x_max = av;
        }
        float current_scale = (x_max > 0) ? x_max / 127.0f : 1e-10f;

        /* spatial_a holds current chain activation in HWC format. */
        int cur_H = cnn_input_h, cur_W = cnn_input_w, cur_C = cnn_input_c;
        for (int k = 0; k < n_input; k++) spatial_a[k] = img_input[k];

        /* Reset per-image saved activation validity. */
        for (int s = 0; s < n_saved_slots; s++) saved_acts[s].valid = 0;

        prof_mark("input_scale");

        for (int li = 0; li < num_layers; li++) {
            Layer *l = &layers[li];

            /* Resolve input source: chain (default) or saved fork (resnet8). */
            const float *in_buf;
            int in_H, in_W, in_C;
            float in_scale;
            if (l->consume_input_from >= 0) {
                int from = l->consume_input_from;
                int slot = layers[from].save_to_slot;
                if (slot < 0 || !saved_acts[slot].valid) {
                    fprintf(stderr,
                            "FATAL L%d: consume_input_from=%d but slot=%d valid=%d\n",
                            li, from, slot,
                            slot >= 0 ? saved_acts[slot].valid : -1);
                    return -1;
                }
                in_buf = saved_acts[slot].data;
                in_H = saved_acts[slot].H;
                in_W = saved_acts[slot].W;
                in_C = saved_acts[slot].C;
                in_scale = saved_acts[slot].scale;
            } else {
                in_buf = spatial_a;
                in_H = cur_H;
                in_W = cur_W;
                in_C = cur_C;
                in_scale = current_scale;
            }

            if (strcmp(l->type, "conv") == 0) {
                int kk = l->kernel_size;
                int pad = l->padding;
                int Ho, Wo;

                /* im2col: in_buf (H, W, C) -> patches_f (Ho*Wo, kk*kk*C). */
                im2col(in_buf, in_H, in_W, in_C, kk, kk, pad, l->stride,
                       patches_f, &Ho, &Wo);
                int n_pixels = Ho * Wo;
                int patch_dim = kk * kk * in_C;
                prof_mark_layer(li, "im2col");

                /* Pad patches to in_f (BLOCK_IN alignment) */
                if (patch_dim < l->in_f) {
                    /* Zero-pad each row from patch_dim to in_f */
                    /* Work backwards to avoid overwriting */
                    for (int r = n_pixels - 1; r >= 0; r--) {
                        /* Move row r from offset r*patch_dim to r*in_f, then zero-pad */
                        if (r > 0)
                            memmove(patches_f + r * l->in_f, patches_f + r * patch_dim,
                                    patch_dim * sizeof(float));
                        memset(patches_f + r * l->in_f + patch_dim, 0,
                               (l->in_f - patch_dim) * sizeof(float));
                    }
                }

                /* Quantize patches using in_scale (chain or fork-resolved).
                 * rintf for banker's rounding to match Python's np.round. */
                for (int k = 0; k < n_pixels * l->in_f; k++) {
                    float v = rintf(patches_f[k] / in_scale);
                    patches_i8[k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
                prof_mark_layer(li, "quant");

                /* DEBUG: input quant stats for the profiled layer. */
                if (_prof_active) {
                    int n_p = n_pixels * l->in_f;
                    int8_t pmin = 127, pmax = -128;
                    long psum = 0;
                    int pnz = 0;
                    for (int k = 0; k < n_p; k++) {
                        int8_t v = patches_i8[k];
                        if (v < pmin) pmin = v;
                        if (v > pmax) pmax = v;
                        psum += v;
                        if (v != 0) pnz++;
                    }
                    fprintf(stderr,
                            "DEBUG L%d patches_i8 (in_scale=%.6f): n=%d  "
                            "min=%d max=%d mean=%.2f nonzero=%d/%d\n",
                            li, in_scale, n_p,
                            (int)pmin, (int)pmax, (double)psum / n_p, pnz, n_p);
                }

                /* Run VTA GEMM. Outer loop: o-tile chunks (existing tiling
                 * for o>96 hardware limit). Inner loop: m-tile chunks. For
                 * single-call legacy layers (m_compiled==m_tiles), the inner
                 * loop runs once with the full tensor (identical to before).
                 * For --force-m1 layers, the inner loop runs m_tiles times,
                 * each call using one weight slice and stitching the BLOCK_OUT
                 * output channels into vta_out_i8 at the right offset. */
                int ot = l->o_tile;
                int chunk_a_bytes = ot * l->n_tiles * BLOCK_IN;
                int n_b_chunks = l->m_tiles / l->m_compiled;
                int per_call_c_bytes = ot * l->m_compiled * BLOCK_OUT;
                /* Per-call output buffer: receives one m-chunk's worth of
                 * output, then we transpose into vta_out_i8. */
                int8_t *c_chunk_buf = (int8_t *)alloca(per_call_c_bytes);

                for (int chunk = 0; chunk < l->n_chunks; chunk++) {
                    int start = chunk * ot;
                    for (int bc = 0; bc < n_b_chunks; bc++) {
                        /* Set up VTA call: A from patches, B from this
                         * m-chunk's pre-loaded weight tensor, C to scratch. */
                        A_cpu->shape[0] = ot;
                        A_cpu->shape[1] = l->n_tiles;
                        A_cpu->shape[2] = 1;
                        A_cpu->shape[3] = BLOCK_IN;
                        memcpy(A_cpu->data, patches_i8 + start * l->in_f,
                               chunk_a_bytes);
                        TVM_CHECK(TVMArrayCopyFromTo(A_cpu, l->A_dl, NULL));

                        TVMValue args[3];
                        int type_codes[3] = {kTVMDLTensorHandle,
                                             kTVMDLTensorHandle,
                                             kTVMDLTensorHandle};
                        args[0].v_handle = l->A_dl;
                        args[1].v_handle = l->B_dl_chunks[bc];
                        args[2].v_handle = l->C_dl;
                        int ret_type;
                        TVMValue ret_value;
                        TVM_CHECK(TVMFuncCall(l->func, args, type_codes, 3,
                                              &ret_value, &ret_type));

                        C_cpu->shape[0] = ot;
                        C_cpu->shape[1] = l->m_compiled;
                        C_cpu->shape[2] = 1;
                        C_cpu->shape[3] = BLOCK_OUT;
                        TVM_CHECK(TVMArrayCopyFromTo(l->C_dl, C_cpu, NULL));
                        memcpy(c_chunk_buf, C_cpu->data, per_call_c_bytes);

                        /* Stitch this m-chunk's BLOCK_OUT*m_compiled channels
                         * into vta_out_i8 at the right offset per row.
                         * vta_out_i8 layout: (o_total, out_f) where
                         *   out_f = m_tiles * BLOCK_OUT
                         * and channel index for this m-chunk's c-th
                         * (m_compiled, BLOCK_OUT) elem is
                         *   bc * m_compiled * BLOCK_OUT + c. */
                        int chans_per_chunk = l->m_compiled * BLOCK_OUT;
                        for (int r = 0; r < ot; r++) {
                            memcpy(vta_out_i8 + (start + r) * l->out_f
                                              + bc * chans_per_chunk,
                                   c_chunk_buf + r * chans_per_chunk,
                                   chans_per_chunk);
                        }
                    }
                }
                prof_mark_layer(li, "gemm");

                /* DEBUG: dump GEMM output for the layer being profiled.
                 * Per-pixel dump splits across all m_tiles so per-tile
                 * corruption shows up. Pixels 100/200/350/400 picked to
                 * land within typical activation regions for both 28x28
                 * (n_pixels=784) and 14x14 (n_pixels=196). */
                if (_prof_active) {
                    int total_out = n_pixels * l->out_f;
                    int8_t vmin = 127, vmax = -128;
                    long sum = 0;
                    int nonzero = 0;
                    for (int k = 0; k < total_out; k++) {
                        int8_t v = vta_out_i8[k];
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                        sum += v;
                        if (v != 0) nonzero++;
                    }
                    fprintf(stderr,
                            "DEBUG L%d vta_out_i8: n=%d (=%d pixels x out_f=%d=%d*BLOCK_OUT)  "
                            "min=%d max=%d mean=%.2f nonzero=%d/%d\n",
                            li, total_out, n_pixels, l->out_f, l->m_tiles,
                            (int)vmin, (int)vmax, (double)sum / total_out,
                            nonzero, total_out);
                    int probe_pixels[] = {100, 200, 350, 400, 500};
                    for (int pp = 0; pp < (int)(sizeof(probe_pixels)/sizeof(int)); pp++) {
                        int p = probe_pixels[pp];
                        if (p >= n_pixels) continue;
                        fprintf(stderr, "DEBUG L%d pixel=%3d", li, p);
                        for (int t = 0; t < l->m_tiles; t++) {
                            fprintf(stderr, "  tile%d[c=%d..%d]:",
                                    t, t * BLOCK_OUT, t * BLOCK_OUT + BLOCK_OUT - 1);
                            for (int c = 0; c < BLOCK_OUT; c++)
                                fprintf(stderr, " %4d",
                                        (int)vta_out_i8[p * l->out_f + t * BLOCK_OUT + c]);
                        }
                        fprintf(stderr, "\n");
                    }
                }

                /* Dequantize + bias. ReLU deferred to after optional skip-add
                 * so that residual sums match Brevitas's QAT semantics
                 * (post-add ReLU). For legacy CNN layers (no skip_add_from,
                 * apply_relu=1) the math is bit-identical to the prior
                 * inline-ReLU version: the output's real_out channels go
                 * through the same dequant + bias + ReLU sequence; padded
                 * channels are discarded by the [:real_out] slice either
                 * way. */
                float combined = in_scale * l->w_scale * (float)(1 << l->shift);
                int out_c = l->real_out;
                for (int r = 0; r < n_pixels; r++) {
                    for (int c = 0; c < out_c; c++) {
                        spatial_b[r * out_c + c] =
                            (float)vta_out_i8[r * l->out_f + c] * combined
                            + l->bias_float[c];
                    }
                }

                /* Skip-add (residual): post-dequant, pre-ReLU. */
                if (l->skip_add_from >= 0) {
                    int from = l->skip_add_from;
                    int slot = layers[from].save_to_slot;
                    if (slot < 0 || !saved_acts[slot].valid) {
                        fprintf(stderr,
                                "FATAL L%d: skip_add_from=%d but slot=%d valid=%d\n",
                                li, from, slot,
                                slot >= 0 ? saved_acts[slot].valid : -1);
                        return -1;
                    }
                    if (saved_acts[slot].H != Ho || saved_acts[slot].W != Wo
                        || saved_acts[slot].C != out_c) {
                        fprintf(stderr,
                                "FATAL L%d: skip-add shape mismatch: main=(%d,%d,%d) "
                                "saved=(%d,%d,%d)\n",
                                li, Ho, Wo, out_c,
                                saved_acts[slot].H, saved_acts[slot].W,
                                saved_acts[slot].C);
                        return -1;
                    }
                    const float *skip_h = saved_acts[slot].data;
                    int n_elem = Ho * Wo * out_c;
                    for (int k = 0; k < n_elem; k++) spatial_b[k] += skip_h[k];
                }

                /* Apply ReLU per layer config (default: 1 for conv = legacy). */
                if (l->apply_relu) {
                    int n_elem = Ho * Wo * out_c;
                    for (int k = 0; k < n_elem; k++) {
                        if (spatial_b[k] < 0) spatial_b[k] = 0;
                    }
                }
                prof_mark_layer(li, "dequant");

                /* MaxPool (legacy CNN) OR memcpy. For branch_only layers we
                 * write to a scratch path so the chain spatial_a is preserved
                 * for the next chain-consuming layer. spatial_b already holds
                 * post-dequant + post-skip-add + post-ReLU values. */
                int new_H, new_W;
                float *post_pool_buf;  /* points to where new activations land */
                if (l->branch_only) {
                    /* Branch-only: do NOT clobber spatial_a. The data we need
                     * is in spatial_b; save_to_slot will copy from there. */
                    post_pool_buf = spatial_b;
                    new_H = Ho;
                    new_W = Wo;
                    /* ResNet-8 downsamples have pool_size=0; no pool here. If
                     * a future topology had a pool on a branch_only layer, it
                     * would need to be applied here into a scratch buffer. */
                } else if (l->pool_size > 0) {
                    maxpool2d(spatial_b, Ho, Wo, out_c, l->pool_size, spatial_a);
                    new_H = Ho / l->pool_size;
                    new_W = Wo / l->pool_size;
                    post_pool_buf = spatial_a;
                } else {
                    memcpy(spatial_a, spatial_b, Ho * Wo * out_c * sizeof(float));
                    new_H = Ho;
                    new_W = Wo;
                    post_pool_buf = spatial_a;
                }
                prof_mark_layer(li, "maxpool");

                /* Compute scale of this layer's output for downstream use. */
                int n_act = new_H * new_W * out_c;
                float next_max = 0;
                for (int k = 0; k < n_act; k++) {
                    float av = fabsf(post_pool_buf[k]);
                    if (av > next_max) next_max = av;
                }
                float layer_out_scale = (next_max > 0) ? next_max / 127.0f : 1e-10f;

                /* Save activation if any future layer references this index. */
                if (l->save_to_slot >= 0) {
                    int slot = l->save_to_slot;
                    memcpy(saved_acts[slot].data, post_pool_buf,
                           n_act * sizeof(float));
                    saved_acts[slot].H = new_H;
                    saved_acts[slot].W = new_W;
                    saved_acts[slot].C = out_c;
                    saved_acts[slot].scale = layer_out_scale;
                    saved_acts[slot].valid = 1;
                }

                /* Update chain unless this layer is a side-branch. For legacy
                 * CNN, branch_only=0 → chain always updates from spatial_a. */
                if (!l->branch_only) {
                    cur_H = new_H;
                    cur_W = new_W;
                    cur_C = out_c;
                    current_scale = layer_out_scale;
                }

                /* DEBUG: post-pool/post-scale snapshot for inter-layer
                 * handoff. Helps detect MaxPool/dequant bugs and scale
                 * mismatches that would corrupt L1's input quant. */
                if (_prof_active && li == 0) {
                    int n_act = cur_H * cur_W * cur_C;
                    float amin = 1e30f, amax = -1e30f, asum = 0;
                    int nz = 0;
                    for (int k = 0; k < n_act; k++) {
                        float v = spatial_a[k];
                        if (v < amin) amin = v;
                        if (v > amax) amax = v;
                        asum += v;
                        if (v != 0) nz++;
                    }
                    fprintf(stderr,
                            "DEBUG L0 post-pool spatial_a: shape=(%d,%d,%d) n=%d  "
                            "min=%.4f max=%.4f mean=%.4f nonzero=%d/%d  "
                            "next_scale=%.6f\n",
                            cur_H, cur_W, cur_C, n_act,
                            amin, amax, (double)asum / n_act, nz, n_act,
                            current_scale);
                }

            } else if (strcmp(l->type, "dense") == 0) {
                /* Global average pool: in_buf (H, W, C) -> feat (C).
                 * Legacy: in_buf=spatial_a, in_H/W/C=cur_H/W/C, in_scale=
                 * current_scale → identical to before. ResNet-8: FC reads
                 * its consume_input_from saved activation (post-stage3). */
                float feat[MAX_OUT_F];
                global_avg_pool(in_buf, in_H, in_W, in_C, feat);
                prof_mark("GAP");

                /* Quantize post-GAP features using current_scale (carried from
                 * end of conv2). Earlier versions recomputed feat_s from the
                 * post-GAP vector itself, which gave tighter ±1 LSB granularity
                 * but differed from Python — and on real VTA hardware this
                 * produced a 5-point accuracy drop when combined with hardware
                 * jitter. The shifts baked into the VTA modules were calibrated
                 * against Python's current_scale behavior (export_vta_cnn.py
                 * calibrate_cnn); this restores that contract.
                 * Also: rintf (banker's half-to-even) instead of roundf, to
                 * match np.round at the Python quant site. */
                int8_t feat_i8[MAX_OUT_F];
                memset(feat_i8, 0, l->in_f);
                for (int k = 0; k < in_C; k++) {
                    float v = rintf(feat[k] / in_scale);
                    feat_i8[k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
                prof_mark("dense_quant");

                if (_prof_active) {
                    fprintf(stderr,
                            "DEBUG dense feat (in_C=%d, in_scale=%.6f): ",
                            in_C, in_scale);
                    for (int k = 0; k < in_C; k++)
                        fprintf(stderr, "%.3f ", feat[k]);
                    fprintf(stderr, "\n");
                    fprintf(stderr, "DEBUG dense feat_i8 (in_f=%d): ", l->in_f);
                    for (int k = 0; k < l->in_f; k++)
                        fprintf(stderr, "%d ", (int)feat_i8[k]);
                    fprintf(stderr, "\n");
                }

                int8_t dense_out[MAX_OUT_F];
                VTA_CALL(l, feat_i8, l->n_tiles * BLOCK_IN,
                         dense_out, l->m_tiles * BLOCK_OUT);
                prof_mark("dense_gemm");

                if (_prof_active) {
                    fprintf(stderr, "DEBUG dense vta_out_i8 (out_f=%d): ", l->out_f);
                    for (int k = 0; k < l->out_f; k++)
                        fprintf(stderr, "%4d ", (int)dense_out[k]);
                    fprintf(stderr, "\n");
                }

                float combined = in_scale * l->w_scale * (float)(1 << l->shift);
                float logits[MAX_OUT_F];
                for (int k = 0; k < l->out_f; k++) {
                    logits[k] = (float)dense_out[k] * combined + l->bias_float[k];
                }

                float best = logits[0];
                int best_idx = 0;
                for (int k = 1; k < l->real_out; k++) {
                    if (logits[k] > best) { best = logits[k]; best_idx = k; }
                }
                if (_prof_active) {
                    fprintf(stderr, "DEBUG dense logits (real_out=%d): ", l->real_out);
                    for (int k = 0; k < l->real_out; k++)
                        fprintf(stderr, "%.3f ", logits[k]);
                    fprintf(stderr, "  -> argmax=%d\n", best_idx);
                }
                prof_mark("dequant+argmax");
                return best_idx;
            }
        }
        return -1;  /* should not reach */
    }

    /* ==============================================================
     * CNN INT4-o8 inference (Mode G pipeline, per-channel dequant)
     * Byte-for-byte functional port of test_vta_cnn_int4_o8.py:infer_one.
     *
     * Per conv layer (2 layers):
     *   CPU: im2col CHW with pad_value = -ZP
     *   CPU: zero-pad each patch row to n_tiles*BLOCK_IN (tail zeros,
     *        matching Python np.pad default — weights pad to zeros too)
     *   CPU: pack two int4 values per byte (pack_int4)
     *   VTA: 4-arg GEMM + corrected_int32_bias + SHR + CLIP[-128,127] -> int8
     *   CPU: per-channel dequant (w_scale[c] * act_scale_in * 2^shift) -> float64 CHW
     *   CPU: ReLU (clip at 0)
     *   CPU: MaxPool 2x2 stride 2
     *   CPU: requant round(pooled / act_scales[ci+1]) clip [0,15], subtract ZP
     *
     * Dense:
     *   CPU: AdaptiveAvgPool via dequant + mean + requant (last_scale cancels,
     *        but we compute literally to match the Python trace)
     *   CPU: pad to BLOCK_IN, pack int4
     *   VTA: 4-arg GEMM + corrected_bias + SHR + CLIP -> int8
     *   CPU: per-channel dequant + argmax (no CPU bias — it's in VTA)
     */
    int cnn_infer_int4_o8(const float *img_28x28) {
        prof_init();
        const int ZP = zero_point;

        /* Count conv layers */
        int num_convs = 0;
        for (int i = 0; i < num_layers; i++) {
            if (strcmp(layers[i].type, "conv") == 0) num_convs++;
        }

        /* ---- Input quantize: img [0,1] -> Brevitas [0,15] -> VTA [-8,7] ---- */
        double in_s = (double)act_scales[0];
        int cur_C = 1, cur_H = 28, cur_W = 28;
        for (int k = 0; k < 784; k++) {
            double v = rint((double)img_28x28[k] / in_s);
            if (v < 0) v = 0;
            if (v > 15) v = 15;
            chw_buf_a[k] = (int8_t)((int)v - ZP);
        }
        int8_t *cur_in = chw_buf_a;
        int8_t *cur_out = chw_buf_b;
        prof_mark("input_quant");

        /* ---- Conv layers ---- */
        for (int ci = 0; ci < num_convs; ci++) {
            Layer *l = &layers[ci];
            int kk = l->kernel_size;
            int pad = l->padding;
            int Ho, Wo;

            /* im2col CHW with pad_value = -ZP */
            im2col_chw(cur_in, cur_C, cur_H, cur_W, kk, kk, pad, -ZP,
                       patches_i8, &Ho, &Wo);
            int n_pixels = Ho * Wo;
            int patch_dim = kk * kk * cur_C;

            /* Zero-pad each row from patch_dim to in_f (tail zeros) */
            if (patch_dim < l->in_f) {
                for (int r = n_pixels - 1; r >= 0; r--) {
                    if (r > 0)
                        memmove(patches_i8 + r * l->in_f, patches_i8 + r * patch_dim,
                                patch_dim);
                    memset(patches_i8 + r * l->in_f + patch_dim, 0,
                           l->in_f - patch_dim);
                }
            }
            prof_mark_layer(ci, "im2col");

            /* VTA chunks: pack int4, call 4-arg module, int8 output */
            int ot = l->o_tile;
            int chunk_a_bytes = ot * l->n_tiles * BLOCK_IN;  /* bytes after pack (int4 -> half) */
            int chunk_c_bytes = ot * l->m_tiles * BLOCK_OUT;
            int chunk_a_bytes_unpacked = ot * l->in_f;       /* pre-pack byte count */

            for (int chunk = 0; chunk < l->n_chunks; chunk++) {
                int start = chunk * ot;
                /* In-place pack: pack_int4 writes output[k] from vals[2k] and
                 * vals[2k+1], so source and dest can overlap safely. After
                 * packing, the tail of the buffer (bytes [chunk_a_bytes/2..])
                 * is zeroed by pack_int4, which the VTA load skips anyway. */
                int8_t *src = patches_i8 + start * l->in_f;
                pack_int4(src, src, chunk_a_bytes_unpacked);
                VTA_CALL_BIAS(l, src, chunk_a_bytes,
                              vta_out_i8 + start * l->out_f, chunk_c_bytes);
            }
            prof_mark_layer(ci, "gemm");

            /* Per-channel dequant (float64) + ReLU into chw_float_a[C, Ho, Wo] */
            int C_out_valid = l->real_out;
            double shift_scale = (double)(1u << l->shift);
            double act_scale_in = (double)act_scales[ci];
            double cs[MAX_OUT_F];
            for (int c = 0; c < C_out_valid; c++) {
                cs[c] = (double)l->w_scale_arr[c] * act_scale_in * shift_scale;
            }
            for (int c = 0; c < C_out_valid; c++) {
                double cs_c = cs[c];
                for (int i = 0; i < Ho; i++) {
                    for (int j = 0; j < Wo; j++) {
                        double v = (double)vta_out_i8[(i * Wo + j) * l->out_f + c] * cs_c;
                        if (v < 0.0) v = 0.0;  /* ReLU */
                        chw_float_a[c * Ho * Wo + i * Wo + j] = v;
                    }
                }
            }
            prof_mark_layer(ci, "dequant");

            /* MaxPool 2x2 stride 2 -> chw_float_b[C, Ho/2, Wo/2] */
            int pool_H = Ho / 2, pool_W = Wo / 2;
            maxpool2d_chw(chw_float_a, C_out_valid, Ho, Wo, 2, chw_float_b);
            prof_mark_layer(ci, "maxpool");

            /* Requant: round(pooled / act_scales[ci+1]) clip [0,15], subtract ZP */
            double out_s = (double)act_scales[ci + 1];
            for (int c = 0; c < C_out_valid; c++) {
                for (int i = 0; i < pool_H; i++) {
                    for (int j = 0; j < pool_W; j++) {
                        double v = rint(chw_float_b[c * pool_H * pool_W + i * pool_W + j] / out_s);
                        if (v < 0) v = 0;
                        if (v > 15) v = 15;
                        cur_out[c * pool_H * pool_W + i * pool_W + j] = (int8_t)((int)v - ZP);
                    }
                }
            }
            prof_mark_layer(ci, "requant");

            /* Swap cur_in <- cur_out for next layer */
            int8_t *tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            cur_H = pool_H; cur_W = pool_W; cur_C = C_out_valid;
        }

        /* ---- Dense layer (AdaptiveAvgPool + 4-arg VTA + argmax) ---- */
        Layer *dl = &layers[num_convs];
        double last_s = (double)act_scales[num_convs];

        /* AdaptiveAvgPool: mean over (H, W) of (cur_in + ZP). The last_scale
         * multiplication in Python cancels against the divide in the requant,
         * so we compute the integer-domain mean directly and then clip/offset. */
        int8_t x_d_vta[MAX_OUT_F];
        memset(x_d_vta, 0, dl->in_f);
        for (int c = 0; c < cur_C; c++) {
            long sum = 0;
            for (int i = 0; i < cur_H; i++) {
                for (int j = 0; j < cur_W; j++) {
                    sum += (int)(cur_in[c * cur_H * cur_W + i * cur_W + j]) + ZP;
                }
            }
            double mean = (double)sum / (double)(cur_H * cur_W);
            double v = rint(mean);
            if (v < 0) v = 0;
            if (v > 15) v = 15;
            x_d_vta[c] = (int8_t)((int)v - ZP);
        }
        prof_mark("GAP+quant");

        /* Pack int4 + 4-arg VTA call */
        int8_t a_dense_packed[MAX_OUT_F];
        pack_int4(x_d_vta, a_dense_packed, dl->in_f);
        int8_t dense_out[MAX_OUT_F];
        VTA_CALL_BIAS(dl, a_dense_packed, dl->n_tiles * BLOCK_IN,
                      dense_out, dl->m_tiles * BLOCK_OUT);
        prof_mark("dense_gemm");

        /* Per-channel dequant + argmax. No CPU bias — corrected bias is in VTA. */
        int C_d_valid = dl->real_out;
        double shift_scale_d = (double)(1u << dl->shift);
        double best = 0.0;
        int best_idx = 0;
        for (int c = 0; c < C_d_valid; c++) {
            double logit = (double)dense_out[c] *
                           (double)dl->w_scale_arr[c] * last_s * shift_scale_d;
            if (c == 0 || logit > best) { best = logit; best_idx = c; }
        }
        prof_mark("dequant+argmax");
        return best_idx;
    }

    /* ---- Clock sanity check ---- */
    {
        time_t now = time(NULL);
        struct tm *t = gmtime(&now);
        if (t->tm_year + 1900 > 2030) {
            fprintf(stderr, "ERROR: Board clock not synced (year %d).\n", t->tm_year + 1900);
            fprintf(stderr, "  From host: ssh -t xilinx@192.168.3.1 \"sudo date -s '$(date -u +%%Y-%%m-%%d\\ %%H:%%M:%%S)'\"\n");
            return 1;
        }
    }

    /* ---- Dispatch macro: selects INT8 or INT4 vta_native MLP path ---- */
    #define MLP_DISPATCH(img_ptr, prediction) do { \
        if (is_vta_native) { \
            MLP_INFER_VTA_NATIVE(img_ptr, prediction); \
        } else { \
            MLP_INFER(img_ptr, prediction); \
        } \
    } while(0)

    /* ---- Top-level dispatch: CNN INT4-o8 / CNN INT8 / MLP ---- */
    #define INFER_DISPATCH(img_ptr, prediction) do { \
        if (is_cnn && is_int4_o8) { \
            (prediction) = cnn_infer_int4_o8((img_ptr)); \
        } else if (is_cnn) { \
            (prediction) = cnn_infer((img_ptr)); \
        } else { \
            MLP_DISPATCH((img_ptr), prediction); \
        } \
    } while(0)

    /* ---- Pre-inference sanity checks ---- */
    printf("Sanity checks:\n");
    printf("  is_vta_native=%d, is_cnn=%d\n", is_vta_native, is_cnn);
    for (int i = 0; i < num_layers; i++) {
        printf("  layer %d: has_vta_bias=%d bias_float=%p bias_int=%p D_dl=%p\n",
               i, layers[i].has_vta_bias,
               (void*)layers[i].bias_float, (void*)layers[i].bias_int,
               (void*)layers[i].D_dl);
        if (!use_int4_weights && !layers[i].bias_float) {
            fprintf(stderr, "FATAL: INT8 layer %d has NULL bias_float!\n", i);
            return 1;
        }
    }

    /* Per-image stride: MLP/MNIST=784; CNN reads cnn_input_*. */
    const int per_image_floats = is_cnn
        ? cnn_input_h * cnn_input_w * cnn_input_c
        : 784;

    /* ---- Warmup + one profiled inference (last warmup image) ----
     * Profile on the last image of the warmup so JIT/cache/PCIe init
     * overhead is excluded — by image 10 the steady-state costs of each
     * stage (im2col, gemm, dequant, etc.) are what we want to see.
     * Profiling i=0 instead would charge L0.gemm with one-shot VTA
     * driver setup (~50 ms vs ~3.8 ms steady-state). */
    printf("Warmup (10 images)...\n");
    int n_warm = (n_images < 10) ? n_images : 10;
    int profile_idx = n_warm - 1;
    for (int i = 0; i < n_warm; i++) {
        int pred;
        if (i == profile_idx) _prof_active = 1;
        INFER_DISPATCH(images + i * per_image_floats, pred);
        if (i == profile_idx) {
            _prof_active = 0;
            prof_print();
        }
        (void)pred;
    }

    /* ---- Verification ---- */
    printf("Verification (100 images)...\n");
    int verify_correct = 0;
    for (int i = 0; i < 100 && i < n_images; i++) {
        int pred;
        INFER_DISPATCH(images + i * per_image_floats, pred);
        if (pred == labels[i]) verify_correct++;
    }
    printf("  Accuracy: %d/100\n", verify_correct);
    if (!is_cnn && verify_correct < 90)
        printf("  WARNING: suspiciously low accuracy\n");
    if (is_cnn && verify_correct < 80)
        printf("  WARNING: suspiciously low accuracy\n");

    /* ---- Stabilization ---- */
    printf("Thermal stabilization (5s)...\n");
    sleep(5);

    /* ---- Idle measurement ---- */
    printf("Idle measurement (5s)...\n");
    struct timespec ts_tmp;
    clock_gettime(CLOCK_REALTIME, &ts_tmp);
    double idle_t_start = ts_tmp.tv_sec + ts_tmp.tv_nsec / 1e9;
    sleep(5);
    clock_gettime(CLOCK_REALTIME, &ts_tmp);
    double idle_t_end = ts_tmp.tv_sec + ts_tmp.tv_nsec / 1e9;

    /* ---- Benchmark runs ---- */
    printf("Running %d benchmark runs (%d images each)...\n", num_runs, n_images);

    double run_t_start[16], run_t_end[16], run_elapsed[16], run_acc[16];
    int run_correct[16];

    for (int run = 0; run < num_runs && run < 16; run++) {
        int correct = 0;
        struct timespec ts_start, ts_end, ts_real;

        clock_gettime(CLOCK_REALTIME, &ts_real);
        run_t_start[run] = ts_real.tv_sec + ts_real.tv_nsec / 1e9;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        for (int i = 0; i < n_images; i++) {
            int pred;
            INFER_DISPATCH(images + i * per_image_floats, pred);
            if (pred == labels[i]) correct++;
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        clock_gettime(CLOCK_REALTIME, &ts_real);
        run_t_end[run] = ts_real.tv_sec + ts_real.tv_nsec / 1e9;

        double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                         (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
        double fps = n_images / elapsed;
        double ms_per = 1000.0 * elapsed / n_images;
        double acc = 100.0 * correct / n_images;

        run_elapsed[run] = elapsed;
        run_acc[run] = acc;
        run_correct[run] = correct;

        printf("  Run %d: %.1f FPS, %.3f ms/img, accuracy=%.2f%% (%d/%d)\n",
               run + 1, fps, ms_per, acc, correct, n_images);
    }

    /* ---- Write JSON ---- */
    if (json_out) {
        FILE *jf = fopen(json_out, "w");
        if (!jf) { fprintf(stderr, "Cannot write %s\n", json_out); }
        else {
            fprintf(jf, "{\n  \"config\": {\n");
            fprintf(jf, "    \"toolchain\": \"vta\",\n");
            fprintf(jf, "    \"runtime\": \"c\",\n");
            fprintf(jf, "    \"model_type\": \"%s\",\n", model_type);
            fprintf(jf, "    \"model_dir\": \"%s\",\n", model_dir);
            fprintf(jf, "    \"dataset\": \"mnist\",\n");
            fprintf(jf, "    \"batch_size\": 1,\n");
            fprintf(jf, "    \"num_runs\": %d,\n", num_runs);
            fprintf(jf, "    \"num_images\": %d,\n", n_images);
            fprintf(jf, "    \"num_layers\": %d,\n", num_layers);
            fprintf(jf, "    \"vta_clock_mhz\": 250,\n");
            fprintf(jf, "    \"board\": \"AUP-ZU3\",\n");
            fprintf(jf, "    \"power_method\": \"none\"\n");
            fprintf(jf, "  },\n");

            fprintf(jf, "  \"idle\": {\n");
            fprintf(jf, "    \"t_start\": %.3f,\n", idle_t_start);
            fprintf(jf, "    \"t_end\": %.3f,\n", idle_t_end);
            fprintf(jf, "    \"power\": {\"mean\": null, \"std\": null, \"n_samples\": 0},\n");
            fprintf(jf, "    \"sysmon\": {\"temp_ps_c\": null, \"temp_pl_c\": null, \"vccint_v\": null, \"n_samples\": 0}\n");
            fprintf(jf, "  },\n");

            fprintf(jf, "  \"runs\": [\n");
            for (int r = 0; r < num_runs; r++) {
                double fps = n_images / run_elapsed[r];
                double ms = 1000.0 * run_elapsed[r] / n_images;
                fprintf(jf, "    {\n");
                fprintf(jf, "      \"run\": %d,\n", r + 1);
                fprintf(jf, "      \"t_start\": %.3f,\n", run_t_start[r]);
                fprintf(jf, "      \"t_end\": %.3f,\n", run_t_end[r]);
                fprintf(jf, "      \"accuracy\": %.2f,\n", run_acc[r]);
                fprintf(jf, "      \"time_s\": %.6f,\n", run_elapsed[r]);
                fprintf(jf, "      \"throughput_fps\": %.1f,\n", fps);
                fprintf(jf, "      \"latency_ms\": %.4f,\n", ms);
                fprintf(jf, "      \"avg_power_w\": null,\n");
                fprintf(jf, "      \"energy_total_j\": null,\n");
                fprintf(jf, "      \"energy_per_image_mj\": null,\n");
                fprintf(jf, "      \"power_samples\": 0,\n");
                fprintf(jf, "      \"sysmon\": null\n");
                fprintf(jf, "    }%s\n", r < num_runs - 1 ? "," : "");
            }
            fprintf(jf, "  ],\n");

            double sum_fps = 0, sum_acc = 0, sum_lat = 0;
            for (int r = 0; r < num_runs; r++) {
                sum_fps += n_images / run_elapsed[r];
                sum_acc += run_acc[r];
                sum_lat += 1000.0 * run_elapsed[r] / n_images;
            }
            double mean_fps = sum_fps / num_runs;
            double mean_acc = sum_acc / num_runs;
            double mean_lat = sum_lat / num_runs;
            double var_fps = 0, var_lat = 0;
            for (int r = 0; r < num_runs; r++) {
                double f = n_images / run_elapsed[r];
                double l_val = 1000.0 * run_elapsed[r] / n_images;
                var_fps += (f - mean_fps) * (f - mean_fps);
                var_lat += (l_val - mean_lat) * (l_val - mean_lat);
            }

            fprintf(jf, "  \"summary\": {\n");
            fprintf(jf, "    \"accuracy\": %.2f,\n", mean_acc);
            fprintf(jf, "    \"throughput_fps_mean\": %.1f,\n", mean_fps);
            fprintf(jf, "    \"throughput_fps_std\": %.1f,\n", sqrt(var_fps / num_runs));
            fprintf(jf, "    \"latency_ms_mean\": %.4f,\n", mean_lat);
            fprintf(jf, "    \"latency_ms_std\": %.4f,\n", sqrt(var_lat / num_runs));
            fprintf(jf, "    \"idle_power_w\": null,\n");
            fprintf(jf, "    \"idle_power_std\": null,\n");
            fprintf(jf, "    \"idle_temp_pl_c\": null,\n");
            fprintf(jf, "    \"avg_power_w_mean\": null,\n");
            fprintf(jf, "    \"avg_power_w_std\": null,\n");
            fprintf(jf, "    \"dynamic_power_w\": null,\n");
            fprintf(jf, "    \"energy_per_image_mj_mean\": null,\n");
            fprintf(jf, "    \"energy_per_image_mj_std\": null\n");
            fprintf(jf, "  }\n");
            fprintf(jf, "}\n");
            fclose(jf);
            printf("Results saved to: %s\n", json_out);
        }
    }

    /* ---- Cleanup ---- */
    for (int i = 0; i < num_layers; i++) {
        TVMArrayFree(layers[i].A_dl);
        TVMArrayFree(layers[i].B_dl);
        TVMArrayFree(layers[i].C_dl);
        free(layers[i].W_tiled);
        if (layers[i].bias_float) free(layers[i].bias_float);
        if (layers[i].bias_int) free(layers[i].bias_int);
        if (layers[i].w_scale_arr) free(layers[i].w_scale_arr);
    }
    TVMArrayFree(A_cpu);
    TVMArrayFree(C_cpu);
    free(patches_f);
    free(patches_i8);
    free(vta_out_i8);
    free(y_float);
    free(spatial_a);
    free(spatial_b);
    if (chw_buf_a)   free(chw_buf_a);
    if (chw_buf_b)   free(chw_buf_b);
    if (chw_float_a) free(chw_float_a);
    if (chw_float_b) free(chw_float_b);
    free(images);
    free(labels);

    printf("Done.\n");
    return 0;
}
