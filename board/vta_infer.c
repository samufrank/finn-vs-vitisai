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
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdint.h>
#include <assert.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#define MAX_LAYERS 8
#define BLOCK_IN 16
#define BLOCK_OUT 16

/* Buffer sizes for CNN (largest case: conv1 = 784 output pixels, conv2 = 80 input features) */
#define MAX_SPATIAL 800   /* >= 28*28 = 784 */
#define MAX_FEATURES 128  /* >= max padded in_f */
#define MAX_OUT_F 16      /* max padded out_f per layer */
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

static void im2col(const float *x_hwc, int H, int W, int C,
                   int kH, int kW, int pad,
                   float *patches, int *Ho_out, int *Wo_out)
{
    int Ho = H + 2 * pad - kH + 1;
    int Wo = W + 2 * pad - kW + 1;
    *Ho_out = Ho;
    *Wo_out = Wo;
    int patch_len = kH * kW * C;

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
    int o_total, o_tile, n_chunks;
    int shift;
    float w_scale;        /* scalar w_scale (INT8 CNN / MLP paths) */
    /* Per-channel w_scale (INT4-o8 CNN). NULL unless this layer has an
     * array-valued "w_scale" in config. Length = w_scale_n (= real_out). */
    float *w_scale_arr;
    int w_scale_n;
    char type[16];  /* "conv", "dense", or "mlp" */
    int kernel_size, padding, in_channels, out_channels, pool_size;

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
} Layer;

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

        /* Extract a substring for this layer (up to closing brace) */
        char layer_json[2048];
        int depth = 0;
        int lj_len = 0;
        const char *lp = lj;
        do {
            if (*lp == '{') depth++;
            if (*lp == '}') depth--;
            layer_json[lj_len++] = *lp++;
        } while (depth > 0 && lj_len < 2047);
        layer_json[lj_len] = '\0';

        layers[i].in_f = json_find_int(layer_json, "in_f");
        layers[i].out_f = json_find_int(layer_json, "out_f");
        layers[i].real_out = json_find_int(layer_json, "real_out");
        layers[i].n_tiles = json_find_int(layer_json, "n_tiles");
        layers[i].m_tiles = json_find_int(layer_json, "m_tiles");
        layers[i].shift = json_find_int(layer_json, "shift");
        layers[i].w_scale = json_find_float(layer_json, "w_scale");
        layers[i].w_scale_arr = NULL;
        layers[i].w_scale_n = 0;
        layers[i].has_vta_bias = json_find_bool(layer_json, "has_vta_bias");
        layers[i].in_scale = json_find_float(layer_json, "in_scale");
        layers[i].D_dl = NULL;
        layers[i].bias_int = NULL;
        layers[i].bias_float = NULL;

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

        /* Allocate VTA tensors (sized for o_tile) */
        int ot = layers[i].o_tile;
        int nt = layers[i].n_tiles;
        int mt = layers[i].m_tiles;

        int64_t a_shape[] = {ot, nt, 1, BLOCK_IN};
        int64_t b_shape[] = {mt, nt, BLOCK_OUT, BLOCK_IN};
        int64_t c_shape[] = {ot, mt, 1, BLOCK_OUT};

        layers[i].A_dl = alloc_vta_tensor(a_shape, 4, dtype_int8);
        layers[i].B_dl = alloc_vta_tensor(b_shape, 4, dtype_int8);
        layers[i].C_dl = alloc_vta_tensor(c_shape, 4, dtype_int8);

        /* Copy weights to VTA (pack int4 for any INT4 path: MLP vta_native or CNN INT4-o8) */
        DLTensor *B_cpu = alloc_cpu_tensor(b_shape, 4, dtype_int8);
        if (use_int4_weights) {
            int w_elems = mt * nt * BLOCK_OUT * BLOCK_IN;
            int8_t *w_packed = (int8_t *)malloc(w_elems);
            pack_int4(layers[i].W_tiled, w_packed, w_elems);
            memcpy(B_cpu->data, w_packed, w_elems);
            free(w_packed);
            printf("    weights: int4-packed (%d elems)\n", w_elems);
        } else {
            memcpy(B_cpu->data, layers[i].W_tiled, mt * nt * BLOCK_OUT * BLOCK_IN);
            printf("    weights: raw int8 (%d bytes)\n", mt * nt * BLOCK_OUT * BLOCK_IN);
        }
        TVM_CHECK(TVMArrayCopyFromTo(B_cpu, layers[i].B_dl, NULL));
        TVMArrayFree(B_cpu);

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

    free(json);
    printf("All modules loaded\n");

    /* ---- Load MNIST ---- */
    float *images = NULL;
    uint8_t *labels = NULL;
    int n_images, n_labels;
    {
        snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte.gz", mnist_dir);
        if (load_mnist_images(path, &images, &n_images) != 0) return 1;
        snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte.gz", mnist_dir);
        if (load_mnist_labels(path, &labels, &n_labels) != 0) return 1;
    }
    if (max_images < n_images) n_images = max_images;
    printf("Loaded %d MNIST test images\n", n_images);

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
    float *spatial_a = (float *)malloc(28 * 28 * MAX_OUT_F * sizeof(float));  /* activation buffer A */
    float *spatial_b = (float *)malloc(28 * 28 * MAX_OUT_F * sizeof(float));  /* activation buffer B */

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
        float current_scale = x_s; \
        for (int _li = 0; _li < num_layers; _li++) { \
            Layer *_l = &layers[_li]; \
            int _a_bytes = _l->n_tiles * BLOCK_IN; \
            int _c_bytes = _l->m_tiles * BLOCK_OUT; \
            int8_t _c_out[MAX_FLAT]; \
            VTA_CALL(_l, h_int8, _a_bytes, _c_out, _c_bytes); \
            float combined = current_scale * _l->w_scale * (float)(1 << _l->shift); \
            for (int _j = 0; _j < _l->out_f; _j++) { \
                h_float_mlp[_j] = (float)_c_out[_j] * combined + _l->bias_float[_j]; \
            } \
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
            } else { \
                float best = h_float_mlp[0]; int best_idx = 0; \
                for (int _j = 1; _j < _l->real_out; _j++) { \
                    if (h_float_mlp[_j] > best) { best = h_float_mlp[_j]; best_idx = _j; } \
                } \
                (prediction) = best_idx; \
            } \
        } \
    } while(0)

    /* MLP inference: vta_native INT4 path */
    int8_t _packed_buf[MAX_FLAT];  /* reusable pack buffer */
    int8_t _unpacked_buf[MAX_FLAT];

    #define MLP_INFER_VTA_NATIVE(img_ptr, prediction) do { \
        float *_img = (img_ptr); \
        /* Input quantize: fixed scale, clip [0, input_clip_max] */ \
        for (int _k = 0; _k < 784; _k++) { \
            float v = roundf(_img[_k] / input_scale); \
            h_int8[_k] = (int8_t)(v < 0 ? 0 : (v > input_clip_max ? input_clip_max : (int)v)); \
        } \
        for (int _li = 0; _li < num_layers; _li++) { \
            Layer *_ln = &layers[_li]; \
            int _a_elems = _ln->n_tiles * BLOCK_IN; \
            int _c_elems = _ln->m_tiles * BLOCK_OUT; \
            /* Pack int4 input into nibble format */ \
            pack_int4(h_int8, _packed_buf, _a_elems); \
            int8_t _c_packed[MAX_FLAT]; \
            if (_ln->has_vta_bias) { \
                /* Hidden layer: 4-arg call. VTA does GEMM+bias+SHR+CLIP. */ \
                /* Output is packed int4 — unpack for next layer. */ \
                VTA_CALL_BIAS(_ln, _packed_buf, _a_elems, _c_packed, _c_elems); \
                unpack_int4(_c_packed, _unpacked_buf, _c_elems); \
                memcpy(h_int8, _unpacked_buf, _ln->real_out); \
            } else { \
                /* Last layer: 3-arg call. CPU dequant + float bias + argmax. */ \
                VTA_CALL(_ln, _packed_buf, _a_elems, _c_packed, _c_elems); \
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
            } \
        } \
    } while(0)

    /* CNN inference: returns predicted class */
    int cnn_infer(const float *img_28x28) {
        /* img_28x28: 784 floats in row-major (H, W) */
        float x_max = 0;
        for (int k = 0; k < 784; k++) {
            float av = fabsf(img_28x28[k]);
            if (av > x_max) x_max = av;
        }
        float current_scale = (x_max > 0) ? x_max / 127.0f : 1e-10f;

        /* spatial_a holds current activation in HWC format */
        /* For first layer, input is (28, 28, 1) */
        int cur_H = 28, cur_W = 28, cur_C = 1;
        for (int k = 0; k < 784; k++) spatial_a[k] = img_28x28[k];

        for (int li = 0; li < num_layers; li++) {
            Layer *l = &layers[li];

            if (strcmp(l->type, "conv") == 0) {
                int kk = l->kernel_size;
                int pad = l->padding;
                int Ho, Wo;

                /* im2col: spatial_a (H, W, C) -> patches_f (Ho*Wo, kk*kk*C) */
                im2col(spatial_a, cur_H, cur_W, cur_C, kk, kk, pad, patches_f, &Ho, &Wo);
                int n_pixels = Ho * Wo;
                int patch_dim = kk * kk * cur_C;

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

                /* Quantize patches. rintf (honors FE_TONEAREST = half-to-even
                 * banker's) instead of roundf (half-away-from-zero) to match
                 * Python's np.round, which the export-time shift calibration
                 * assumed. Differences of ±1 LSB at this site cascade through
                 * two conv layers and can flip argmax on VTA hardware. */
                for (int k = 0; k < n_pixels * l->in_f; k++) {
                    float v = rintf(patches_f[k] / current_scale);
                    patches_i8[k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }

                /* Run VTA GEMM in chunks */
                int ot = l->o_tile;
                int chunk_a_bytes = ot * l->n_tiles * BLOCK_IN;
                int chunk_c_bytes = ot * l->m_tiles * BLOCK_OUT;

                for (int chunk = 0; chunk < l->n_chunks; chunk++) {
                    int start = chunk * ot;
                    /* Input: patches_i8[start*in_f ... (start+ot)*in_f]
                     * Already in (rows, in_f) layout = (ot, n_tiles, 1, BLOCK_IN) when
                     * in_f = n_tiles * BLOCK_IN, which it is by construction.
                     */
                    VTA_CALL(l, patches_i8 + start * l->in_f, chunk_a_bytes,
                             vta_out_i8 + start * l->out_f, chunk_c_bytes);
                }

                /* Dequantize + bias + ReLU */
                float combined = current_scale * l->w_scale * (float)(1 << l->shift);
                for (int r = 0; r < n_pixels; r++) {
                    for (int c = 0; c < l->out_f; c++) {
                        float val = (float)vta_out_i8[r * l->out_f + c] * combined + l->bias_float[c];
                        if (val < 0) val = 0;  /* ReLU */
                        y_float[r * l->out_f + c] = val;
                    }
                }

                /* Reshape to spatial (Ho, Wo, real_out) into spatial_b */
                int out_c = l->real_out;
                for (int r = 0; r < n_pixels; r++) {
                    for (int c = 0; c < out_c; c++) {
                        spatial_b[r * out_c + c] = y_float[r * l->out_f + c];
                    }
                }

                /* MaxPool */
                if (l->pool_size > 0) {
                    maxpool2d(spatial_b, Ho, Wo, out_c, l->pool_size, spatial_a);
                    cur_H = Ho / l->pool_size;
                    cur_W = Wo / l->pool_size;
                    cur_C = out_c;
                } else {
                    memcpy(spatial_a, spatial_b, Ho * Wo * out_c * sizeof(float));
                    cur_H = Ho;
                    cur_W = Wo;
                    cur_C = out_c;
                }

                /* Update scale for next layer */
                float next_max = 0;
                for (int k = 0; k < cur_H * cur_W * cur_C; k++) {
                    float av = fabsf(spatial_a[k]);
                    if (av > next_max) next_max = av;
                }
                current_scale = (next_max > 0) ? next_max / 127.0f : 1e-10f;

            } else if (strcmp(l->type, "dense") == 0) {
                /* Global average pool: spatial_a (H, W, C) -> feat (C) */
                float feat[MAX_OUT_F];
                global_avg_pool(spatial_a, cur_H, cur_W, cur_C, feat);

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
                for (int k = 0; k < cur_C; k++) {
                    float v = rintf(feat[k] / current_scale);
                    feat_i8[k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }

                int8_t dense_out[MAX_OUT_F];
                VTA_CALL(l, feat_i8, l->n_tiles * BLOCK_IN,
                         dense_out, l->m_tiles * BLOCK_OUT);

                float combined = current_scale * l->w_scale * (float)(1 << l->shift);
                float logits[MAX_OUT_F];
                for (int k = 0; k < l->out_f; k++) {
                    logits[k] = (float)dense_out[k] * combined + l->bias_float[k];
                }

                float best = logits[0];
                int best_idx = 0;
                for (int k = 1; k < l->real_out; k++) {
                    if (logits[k] > best) { best = logits[k]; best_idx = k; }
                }
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

            /* MaxPool 2x2 stride 2 -> chw_float_b[C, Ho/2, Wo/2] */
            int pool_H = Ho / 2, pool_W = Wo / 2;
            maxpool2d_chw(chw_float_a, C_out_valid, Ho, Wo, 2, chw_float_b);

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

        /* Pack int4 + 4-arg VTA call */
        int8_t a_dense_packed[MAX_OUT_F];
        pack_int4(x_d_vta, a_dense_packed, dl->in_f);
        int8_t dense_out[MAX_OUT_F];
        VTA_CALL_BIAS(dl, a_dense_packed, dl->n_tiles * BLOCK_IN,
                      dense_out, dl->m_tiles * BLOCK_OUT);

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

    /* ---- Warmup ---- */
    printf("Warmup (10 images)...\n");
    for (int i = 0; i < 10 && i < n_images; i++) {
        int pred;
        INFER_DISPATCH(images + i * 784, pred);
        (void)pred;
    }

    /* ---- Verification ---- */
    printf("Verification (100 images)...\n");
    int verify_correct = 0;
    for (int i = 0; i < 100 && i < n_images; i++) {
        int pred;
        INFER_DISPATCH(images + i * 784, pred);
        if (pred == labels[i]) verify_correct++;
    }
    printf("  Accuracy: %d/100\n", verify_correct);
    if (!is_cnn && verify_correct < 90)
        printf("  WARNING: suspiciously low accuracy\n");
    if (is_cnn && verify_correct < 80)
        printf("  WARNING: suspiciously low accuracy\n");

    /* ---- Stabilization ---- */
    printf("Thermal stabilization (10s)...\n");
    sleep(10);

    /* ---- Idle measurement ---- */
    printf("Idle measurement (10s)...\n");
    struct timespec ts_tmp;
    clock_gettime(CLOCK_REALTIME, &ts_tmp);
    double idle_t_start = ts_tmp.tv_sec + ts_tmp.tv_nsec / 1e9;
    sleep(10);
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
            INFER_DISPATCH(images + i * 784, pred);
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
