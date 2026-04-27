/* vta_transformer_infer.c — INT4-o8 RadioML Transformer C runner.
 *
 * Goal: bit-exact mirror of benchmark_vta_transformer.py's infer() in C, to
 * eliminate the Python overhead that dominates ~408 VTA calls/inference.
 *
 * Architecture: standalone executable (vta_infer.c pattern). The user loads
 * the bitstream + libvta separately (e.g. `python -c "from pynq import
 * Overlay; Overlay('vta.bit')"`); this executable then loads the 6 cross-
 * compiled VTA .so modules via TVMModLoadFromFile, loads .npy weights /
 * biases / CPU params, and runs the per-sample hot loop.
 *
 * The VTA call path follows vta_infer.c's VTA_CALL_BIAS macro (lines
 * 841-868) — TVMArrayCopyFromTo handles cache coherency, no explicit
 * dc_cvac/dc_civac needed.
 *
 * Phase 1 (this turn): standalone refactor — main() with --flags CLI,
 * build-doc updated, CPU ops are STUBS, weight loaders are STUBS.
 * The executable compiles + links. Running it without Phase 2 will
 * either fail at module-load (if .so files are missing) or produce
 * garbage predictions (if it gets past load).
 *
 * Build (on board):
 *   gcc -O2 -o vta_transformer_infer vta_transformer_infer.c \
 *       -I/home/xilinx/tvm-src/include \
 *       -I/home/xilinx/tvm-src/3rdparty/dlpack/include \
 *       -L/home/xilinx/tvm-src/build -ltvm_runtime -ldl -lm
 *
 * Use:
 *   # 1) Load bitstream once per boot
 *   python3 -c "from pynq import Overlay; Overlay('/home/xilinx/vta.bit')"
 *
 *   # 2) Run
 *   sudo LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH \
 *        ./vta_transformer_infer \
 *            --weights /home/xilinx/transformer_export \
 *            --data    /home/xilinx/data/radioml2018_eval_snr_filtered.npz \
 *            --n       1000
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>


/* ====================== Constants & helper macros ====================== */
#define BLOCK     16
#define M_FULL    64
#define D_MODEL   96
#define D_FF      384
#define N_HEADS   3
#define D_HEAD    32
#define N_CLASSES 24
#define SQRT_96   0.10206207261596577      /* 1 / sqrt(96), double precision */

/* Compile-time scales — copied verbatim from
 * finn-vs-vitisai/vta/transformer_export/scales.json. These are constant
 * across all inferences for this checkpoint; no JSON parsing needed. */
#define SCALE_EMB_IN          0.01308732945472002
#define SCALE_EMB_W           0.005670934449881315
#define SCALE_EMB_OUT         0.0062857321463525295
#define SCALE_POS_IN          0.012570053339004517
#define SCALE_POS_OUT         0.012025801464915276
#define SCALE_ATTN_PRE_OUT    0.28512388467788696
#define SCALE_Q_W             0.07275883108377457
#define SCALE_Q_OUT           0.3750433921813965
#define SCALE_K_W             0.0737423524260521
#define SCALE_K_OUT           0.2912832796573639
#define SCALE_V_W             0.08723586797714233
#define SCALE_V_OUT           0.13940021395683289
#define SCALE_SOFTMAX_IN      0.6048948764801025
#define SCALE_SOFTMAX_OUT     0.010451252572238445
#define SCALE_O_IN            0.14017151296138763
#define SCALE_O_W             0.0674523338675499
#define SCALE_ATTN_RESIDUAL   0.17021943628787994
#define SCALE_MLP_BN_OUT      0.24027536809444427
#define SCALE_FC1_W           0.037845320999622345
#define SCALE_FC1_OUT         0.17932777106761932
#define SCALE_FC2_W           0.04726411774754524
#define SCALE_MLP_RESIDUAL    0.8087873458862305
#define SCALE_CLS_W           0.00672035152092576
#define SCALE_CLS_OUT         0.09700276702642441

/* Tuned coarse shifts — from config.json["tuned_shifts"]. */
#define SH_Q   3
#define SH_K   3
#define SH_V   2
#define SH_QK  3
#define SH_AV  3
#define SH_O   3
#define SH_FC1 3
#define SH_FC2 4

/* TVM_CHECK aborts on failure — acceptable for Phase 1 (kills the Python
 * caller, but at least the error reaches stderr). Phase 2 should refactor
 * to a return-code-based error path so the Python wrapper can recover. */
#define TVM_CHECK(call) do { \
    int _r = (call); \
    if (_r != 0) { \
        fprintf(stderr, "TVM error at %s:%d: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
        exit(1); \
    } \
} while (0)


/* ============================ Module state ============================= */
/* One VtaModule per compiled .so. Buffer shapes are read from
 * config.json's "modules" entry at init. */
typedef struct {
    const char *name;          /* "proj_k96_s3_m1" etc. */
    char        so_path[256];

    /* Geometry (from config.json) */
    int M_full;                /* GEMM full M = 64 */
    int K;                     /* GEMM K (96, 32, 64, 96, 384) */
    int N_full;                /* GEMM full N (varies per call site) */
    int o_tile;                /* per-call M (8 or 16) */
    int n_tiles;               /* K / BLOCK */
    int n_m_chunks;            /* M_full / o_tile */
    int n_calls_per_gemm;      /* N_full / BLOCK (per output stripe) */
    int has_real_bias;         /* fc1, fc2 only */

    /* TVM handles */
    TVMModuleHandle   mod;
    TVMFunctionHandle func;

    /* DMA buffers (kDLExtDev, device 0) — reused across all calls of this module */
    DLTensor *A_dl, *W_dl, *D_dl, *C_dl;
    /* CPU staging (kDLCPU) — same shapes, used as the H2D/D2H source/dest */
    DLTensor *A_cpu, *W_cpu, *D_cpu, *C_cpu;
} VtaModule;


/* ============================ Global state ============================= */
typedef struct {
    /* The 6 modules (mirrors transformer_export/config.json["modules"]) */
    VtaModule proj_s3;          /* shift=3: q_proj, k_proj, o_proj */
    VtaModule proj_s2;          /* shift=2: v_proj */
    VtaModule qkt;
    VtaModule av;
    VtaModule fc1;
    VtaModule fc2;

    /* Static weights, pre-packed as a flat list of n_calls_per_gemm slices.
     * Each slice is (n_tiles * BLOCK * BLOCK) bytes int8 (int4 nibbles). */
    int8_t **W_q_slices, **W_k_slices, **W_v_slices, **W_o_slices;
    int8_t **W_fc1_slices, **W_fc2_slices;

    /* Per-call int32 bias broadcasts for fc1/fc2 — one (o_tile, 1, 1, BLOCK)
     * buffer per output stripe (m=1 N-chunk index). */
    int32_t **bias_fc1_per_call;   /* len = fc1.n_calls_per_gemm */
    int32_t **bias_fc2_per_call;   /* len = fc2.n_calls_per_gemm */

    /* CPU-side params */
    int8_t  *W_emb_flat;        /* (96, 32) int8 — patch-embed weight, im2col layout */
    int32_t  bias_emb_int32[D_MODEL];
    float   *bn_emb_mean,  *bn_emb_var;
    float   *pos2d;             /* (64, 96) — flattened pos_enc[0] */
    float   *bn_attn_mean, *bn_attn_var;
    float   *bn_mlp_mean,  *bn_mlp_var;
    int8_t  *W_cls;             /* (24, 96) int8 */
    float   *b_cls;             /* (24,) float */
    double   W_cls_float[N_CLASSES * D_MODEL];   /* W_cls * cls_w, dequantised — double for bit-exact match */

    /* Scales (from scales.json) */
    float emb_in, emb_w, emb_out;
    float pos_in, pos_out;
    float attn_pre_out;
    float q_w, k_w, v_w, q_out, k_out, v_out, o_in, o_w;
    float softmax_in, softmax_out;
    float attn_residual;
    float mlp_bn_out, fc1_w, fc1_out, fc2_w, mlp_residual;
    float cls_w, cls_out;

    /* Coarse shifts (tuned values from config.json["tuned_shifts"]) */
    int sh_q, sh_k, sh_v, sh_qk, sh_av, sh_o, sh_fc1, sh_fc2;
} TransformerCtx;


static void           *g_libvta = NULL;
static TransformerCtx  g_ctx;
static const DLDevice  g_vta_dev = {kDLExtDev, 0};
static const DLDevice  g_cpu_dev = {kDLCPU,    0};


/* ====================== libvta + module loading ======================== */
static int load_libvta(void) {
    g_libvta = dlopen("/home/xilinx/tvm-src/build/libvta.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_libvta) {
        fprintf(stderr, "dlopen libvta.so failed: %s\n", dlerror());
        return 1;
    }
    return 0;
}


static DLTensor *alloc_tensor(const int64_t *shape, int ndim,
                               DLDataTypeCode code, int bits, DLDevice dev) {
    DLTensor *t = NULL;
    TVM_CHECK(TVMArrayAlloc(shape, ndim, (int)code, bits, 1,
                            (int)dev.device_type, dev.device_id, &t));
    return t;
}


static void load_module(VtaModule *vm, const char *export_dir) {
    snprintf(vm->so_path, sizeof(vm->so_path),
             "%s/modules/%s.so", export_dir, vm->name);
    TVM_CHECK(TVMModLoadFromFile(vm->so_path, "so", &vm->mod));
    /* TVM cross-compiled GEMM modules export via "default_function" or
     * the user-set name. Try the user name first; fall back if missing. */
    int rc = TVMModGetFunction(vm->mod, vm->name, 0, &vm->func);
    if (rc != 0 || vm->func == NULL) {
        TVM_CHECK(TVMModGetFunction(vm->mod, "default_function", 0, &vm->func));
    }

    /* DMA + CPU staging buffers — one set per module, reused across all calls */
    int64_t a_shape[] = {vm->o_tile, vm->n_tiles, 1, BLOCK};
    int64_t w_shape[] = {1, vm->n_tiles, BLOCK, BLOCK};
    int64_t d_shape[] = {vm->o_tile, 1, 1, BLOCK};
    int64_t c_shape[] = {vm->o_tile, 1, 1, BLOCK};
    vm->A_dl  = alloc_tensor(a_shape, 4, kDLInt, 8,  g_vta_dev);
    vm->W_dl  = alloc_tensor(w_shape, 4, kDLInt, 8,  g_vta_dev);
    vm->D_dl  = alloc_tensor(d_shape, 4, kDLInt, 32, g_vta_dev);
    vm->C_dl  = alloc_tensor(c_shape, 4, kDLInt, 8,  g_vta_dev);
    vm->A_cpu = alloc_tensor(a_shape, 4, kDLInt, 8,  g_cpu_dev);
    vm->W_cpu = alloc_tensor(w_shape, 4, kDLInt, 8,  g_cpu_dev);
    vm->D_cpu = alloc_tensor(d_shape, 4, kDLInt, 32, g_cpu_dev);
    vm->C_cpu = alloc_tensor(c_shape, 4, kDLInt, 8,  g_cpu_dev);
}


/* ====================== int4 packing (matches Python) ================== */
/* Same convention as benchmark_vta_transformer.py:pack_int4_for_vta —
 * first half of flat byte buffer holds packed nibble pairs (lo=even, hi=odd),
 * second half is zero-padded. Caller passes n = total int4 elements. */
static void pack_int4(const int8_t *src, int8_t *dst, size_t n) {
    size_t half = n / 2;
    for (size_t i = 0; i < half; i++) {
        uint8_t lo = (uint8_t)src[2*i]     & 0xF;
        uint8_t hi = (uint8_t)src[2*i + 1] & 0xF;
        dst[i] = (int8_t)((hi << 4) | lo);
    }
    memset(dst + half, 0, n - half);
}


/* ====================== Timing instrumentation ========================
 * Per-stage accumulator. Updated by infer_one(); reported by main()
 * when --timing is given. CLOCK_MONOTONIC throughout. */
enum {
    T_PATCH_EMBED, T_POSITIONAL, T_BN_ATTN,
    T_VTA_QKV, T_CPU_REQUANT_QKV, T_CPU_HEAD_SPLIT,
    T_VTA_QKT, T_CPU_SOFTMAX, T_VTA_AV,
    T_CPU_HEAD_CONCAT, T_VTA_O_PROJ, T_CPU_RESID_ATTN,
    T_BN_MLP, T_VTA_FC1, T_CPU_RELU_QUANT,
    T_VTA_FC2, T_CPU_RESID_MLP, T_CLASSIFIER,
    T_NUM_STAGES
};
static double g_timings[T_NUM_STAGES];

static const char *g_stage_names[T_NUM_STAGES] = {
    "patch_embed",     "positional",      "bn_attn",
    "vta_qkv",         "cpu_requant_qkv", "cpu_head_split",
    "vta_qkt",         "cpu_softmax",     "vta_av",
    "cpu_head_concat", "vta_o_proj",      "cpu_resid_attn",
    "bn_mlp",          "vta_fc1",         "cpu_relu_quant",
    "vta_fc2",         "cpu_resid_mlp",   "classifier",
};
static const char *g_stage_types[T_NUM_STAGES] = {
    "CPU","CPU","CPU",
    "VTA","CPU","CPU",
    "VTA","CPU","VTA",
    "CPU","VTA","CPU",
    "CPU","VTA","CPU",
    "VTA","CPU","CPU",
};

static inline double diff_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) * 1e-6;
}


/* ====================== Numerics: round-to-even + clip ================== */
/* Python uses np.round (banker's: round half to even). C99 round() is
 * round-half-away-from-zero. rint() uses the current rounding mode, which
 * defaults to FE_TONEAREST (= round-half-to-even per IEEE-754). Use rint()
 * everywhere for bit-exact match with the numpy reference. */
static inline int8_t clip_round_i8(double x, int lo, int hi) {
    double r = rint(x);
    if (r > hi) r = hi;
    else if (r < lo) r = lo;
    return (int8_t)r;
}
static inline int32_t clip_round_i32(double x, int lo, int hi) {
    double r = rint(x);
    if (r > hi) r = hi;
    else if (r < lo) r = lo;
    return (int32_t)r;
}


/* ====================== Minimal .npy v1 reader =========================
 * Reads numpy v1 .npy files with little-endian dtypes <i1, <i4, <f4 and
 * up to 8-d shape. Allocates one buffer for the data; caller frees with
 * npy_free.  This is enough for transformer_export's weights / biases /
 * cpu_params. ================================================ */
typedef struct {
    int      ndim;
    int64_t  shape[8];
    char     dtype[8];      /* e.g. "<i1", "<i4", "<f4" */
    int      itemsize;      /* 1, 4 */
    size_t   n_elements;
    void    *data;          /* malloc'd; free via npy_free */
} NpyArray;

static void npy_free(NpyArray *a) {
    if (a && a->data) { free(a->data); a->data = NULL; }
}

/* Find the value associated with `key` in a Python-dict-style header
 * substring. Returns pointer just past the colon and whitespace, or NULL. */
static const char *npy_find_value(const char *header, const char *key) {
    const char *p = strstr(header, key);
    if (!p) return NULL;
    p = strchr(p, ':');
    if (!p) return NULL;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

static int npy_load(const char *path, NpyArray *out) {
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "npy_load: cannot open %s\n", path); return -1; }

    unsigned char magic[10];
    if (fread(magic, 1, 10, f) != 10) { fclose(f); return -1; }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "npy_load: bad magic in %s\n", path); fclose(f); return -1;
    }
    int major = magic[6];
    /* uint16 LE header length for v1.x; uint32 for v2+. We only handle v1. */
    if (major != 1) {
        fprintf(stderr, "npy_load: %s is .npy v%d (only v1 supported)\n", path, major);
        fclose(f); return -1;
    }
    int header_len = (int)magic[8] | ((int)magic[9] << 8);
    char header[1024];
    if (header_len >= (int)sizeof(header)) {
        fprintf(stderr, "npy_load: %s header too large\n", path); fclose(f); return -1;
    }
    if (fread(header, 1, (size_t)header_len, f) != (size_t)header_len) {
        fclose(f); return -1;
    }
    header[header_len] = '\0';

    /* Parse 'descr' */
    const char *p = npy_find_value(header, "'descr'");
    if (!p || *p != '\'') { fprintf(stderr, "npy_load: missing descr in %s\n", path); fclose(f); return -1; }
    p++;
    int dlen = 0;
    while (p[dlen] && p[dlen] != '\'' && dlen < (int)sizeof(out->dtype) - 1) {
        out->dtype[dlen] = p[dlen]; dlen++;
    }
    out->dtype[dlen] = '\0';
    if      (!strcmp(out->dtype, "<i1") || !strcmp(out->dtype, "|i1")) out->itemsize = 1;
    else if (!strcmp(out->dtype, "<i4")) out->itemsize = 4;
    else if (!strcmp(out->dtype, "<f4")) out->itemsize = 4;
    else { fprintf(stderr, "npy_load: unsupported dtype %s in %s\n", out->dtype, path); fclose(f); return -1; }

    /* Reject fortran_order=True */
    p = npy_find_value(header, "'fortran_order'");
    if (p && !strncmp(p, "True", 4)) {
        fprintf(stderr, "npy_load: %s is fortran-ordered (unsupported)\n", path);
        fclose(f); return -1;
    }

    /* Parse 'shape': (n0, n1, ...) */
    p = npy_find_value(header, "'shape'");
    if (!p || *p != '(') { fprintf(stderr, "npy_load: missing shape in %s\n", path); fclose(f); return -1; }
    p++;
    out->ndim = 0;
    out->n_elements = 1;
    while (*p && *p != ')' && out->ndim < 8) {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out->shape[out->ndim++] = (int64_t)v;
        out->n_elements *= (size_t)v;
        p = end;
    }
    if (out->ndim == 0) {
        fprintf(stderr, "npy_load: zero-d array in %s\n", path); fclose(f); return -1;
    }

    /* Read data */
    size_t bytes = out->n_elements * (size_t)out->itemsize;
    out->data = malloc(bytes);
    if (!out->data) { fclose(f); return -1; }
    if (fread(out->data, 1, bytes, f) != bytes) {
        fprintf(stderr, "npy_load: short read of %s (wanted %zu bytes)\n", path, bytes);
        free(out->data); out->data = NULL; fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

/* Convenience: load + assert dtype matches expected. */
static int npy_load_typed(const char *path, NpyArray *out, const char *expect_dtype) {
    if (npy_load(path, out) != 0) return -1;
    if (strcmp(out->dtype, expect_dtype) != 0 &&
        !(!strcmp(expect_dtype, "<i1") && !strcmp(out->dtype, "|i1"))) {
        fprintf(stderr, "npy_load_typed: %s expected %s got %s\n",
                path, expect_dtype, out->dtype);
        npy_free(out); return -1;
    }
    return 0;
}


/* ====================== One VTA call (4-arg) =========================== */
/* Mirrors vta_infer.c:841-868 (VTA_CALL_BIAS). Caller has already populated
 * vm->A_cpu / vm->W_cpu / vm->D_cpu with int4-packed / int32 bias bytes;
 * result lands in vm->C_cpu after this returns. The D2H copy is the
 * implicit sync barrier (waits for VTA DMA-out to complete).
 *
 * Includes the same retry-on-all-zero behaviour as the Python runner: an
 * intermittent VTA glitch on this PL produces all-zero C; up to 3 retries
 * with fresh H2D recover from it. Logs to stderr. */
static void vta_call(VtaModule *vm) {
    const size_t C_bytes = (size_t)vm->o_tile * BLOCK;
    for (int attempt = 0; attempt < 4; attempt++) {
        TVM_CHECK(TVMArrayCopyFromTo(vm->A_cpu, vm->A_dl, NULL));
        TVM_CHECK(TVMArrayCopyFromTo(vm->W_cpu, vm->W_dl, NULL));
        TVM_CHECK(TVMArrayCopyFromTo(vm->D_cpu, vm->D_dl, NULL));
        memset(vm->C_cpu->data, 0, C_bytes);
        TVM_CHECK(TVMArrayCopyFromTo(vm->C_cpu, vm->C_dl, NULL));

        TVMValue args[4];
        int      codes[4] = {kTVMDLTensorHandle, kTVMDLTensorHandle,
                             kTVMDLTensorHandle, kTVMDLTensorHandle};
        args[0].v_handle = vm->A_dl;
        args[1].v_handle = vm->W_dl;
        args[2].v_handle = vm->D_dl;
        args[3].v_handle = vm->C_dl;
        TVMValue rv;
        int      rt;
        TVM_CHECK(TVMFuncCall(vm->func, args, codes, 4, &rv, &rt));

        TVM_CHECK(TVMArrayCopyFromTo(vm->C_dl, vm->C_cpu, NULL));  /* sync */

        const int8_t *C = (const int8_t *)vm->C_cpu->data;
        int nonzero = 0;
        for (size_t i = 0; i < C_bytes; i++) {
            if (C[i] != 0) { nonzero = 1; break; }
        }
        if (nonzero) return;
        if (attempt < 3) {
            fprintf(stderr, "[retry] %s: zero output, retry %d/3\n",
                    vm->name, attempt + 1);
        }
    }
    /* All 4 attempts produced zero; downstream sees zeros (caller still
     * proceeds — accuracy will reflect any unrecoverable failures). */
}


/* ====================== m=1 N-chunked path (per M-chunk) ================ */
/* For one M-chunk's worth of A_packed (already nibble-packed), issue
 * n_calls_per_gemm VTA calls — one per output BLOCK_OUT stripe — and
 * concatenate (o_tile, BLOCK) outputs into out_int8 row-by-row.
 *
 *   A_packed     : (o_tile * n_tiles * BLOCK) bytes, contiguous int8 nibbles
 *   W_slices     : array[n_calls_per_gemm] of (n_tiles * BLOCK * BLOCK) bytes
 *   D_per_call   : array[n_calls_per_gemm] of (o_tile * BLOCK) int32, or
 *                  NULL for zero bias (we memset D_cpu to 0 ourselves)
 *   out_int8     : (o_tile * N_full) int8 — destination
 */
static void vta_gemm_m1(VtaModule *vm,
                         const int8_t  *A_packed,
                         int8_t * const *W_slices,
                         const int32_t * const *D_per_call,
                         int8_t        *out_int8) {
    const size_t A_bytes = (size_t)(vm->o_tile * vm->n_tiles * BLOCK);
    const size_t W_bytes = (size_t)(vm->n_tiles * BLOCK * BLOCK);
    const size_t D_bytes = (size_t)(vm->o_tile * BLOCK) * sizeof(int32_t);

    for (int mc = 0; mc < vm->n_calls_per_gemm; mc++) {
        memcpy(vm->A_cpu->data, A_packed, A_bytes);
        memcpy(vm->W_cpu->data, W_slices[mc], W_bytes);
        if (D_per_call != NULL) {
            memcpy(vm->D_cpu->data, D_per_call[mc], D_bytes);
        } else {
            memset(vm->D_cpu->data, 0, D_bytes);
        }
        vta_call(vm);
        /* C_cpu->data shape (o_tile, 1, 1, BLOCK) -> stripe column
         * [mc*BLOCK, (mc+1)*BLOCK) of the (o_tile, N_full) output. */
        const int8_t *C = (const int8_t *)vm->C_cpu->data;
        for (int row = 0; row < vm->o_tile; row++) {
            memcpy(out_int8 + row * vm->N_full + mc * BLOCK,
                   C + row * BLOCK, BLOCK);
        }
    }
}


/* ====================== GEMM helpers (M-chunk wrappers) ================ */
/* Three high-level helpers correspond to benchmark_vta_transformer.py's
 *   _proj           static W, no real bias  (q/k/v/o)
 *   _runtime_w_gemm runtime W from caller   (qkt, av)
 *   _mlp            static W, real bias      (fc1, fc2)
 * Each does outer M-chunking of (M_full, K) input into n_m_chunks pieces
 * of (o_tile, K) and calls vta_gemm_m1 per chunk. */
static void gemm_proj(VtaModule *vm,
                       const int8_t *in_unpacked,        /* (M_full, K) */
                       int8_t * const *W_slices,         /* [n_calls_per_gemm] */
                       int8_t       *out_int8 /* (M_full, N_full) */) {
    /* Scratch sized for the largest valid (o_tile * n_tiles * BLOCK) across
     * all 6 modules: fc2 with o=8, n=24 -> 8*24*16 = 3072. Round up. */
    static int8_t scratch[8 * 1024];
    const int chunk_K_bytes  = vm->o_tile * vm->K;
    const int packed_bytes   = vm->o_tile * vm->n_tiles * BLOCK;
    if ((size_t)packed_bytes > sizeof(scratch)) {
        fprintf(stderr, "scratch undersized for %s (%d > %zu)\n",
                vm->name, packed_bytes, sizeof(scratch));
        exit(1);
    }
    for (int mc = 0; mc < vm->n_m_chunks; mc++) {
        const int8_t *chunk_in = in_unpacked + mc * chunk_K_bytes;
        pack_int4(chunk_in, scratch, (size_t)packed_bytes);
        vta_gemm_m1(vm, scratch, W_slices, NULL,
                     out_int8 + mc * vm->o_tile * vm->N_full);
    }
}

static void gemm_runtime_w(VtaModule *vm,
                            const int8_t *A_unpacked,
                            int8_t * const *W_slices,
                            int8_t       *out_int8) {
    /* Identical chunk loop to gemm_proj — just a name for clarity. */
    gemm_proj(vm, A_unpacked, W_slices, out_int8);
}

static void gemm_mlp(VtaModule *vm,
                      const int8_t  *in_unpacked,
                      int8_t * const *W_slices,
                      const int32_t * const *D_per_call,
                      int8_t        *out_int8) {
    static int8_t scratch[8 * 1024];
    const int chunk_K_bytes = vm->o_tile * vm->K;
    const int packed_bytes  = vm->o_tile * vm->n_tiles * BLOCK;
    for (int mc = 0; mc < vm->n_m_chunks; mc++) {
        const int8_t *chunk_in = in_unpacked + mc * chunk_K_bytes;
        pack_int4(chunk_in, scratch, (size_t)packed_bytes);
        vta_gemm_m1(vm, scratch, W_slices, D_per_call,
                     out_int8 + mc * vm->o_tile * vm->N_full);
    }
}


/* ====================== CPU helper STUBS (Phase 1) =====================
 * Each function has a TODO comment block above with the EXACT computation
 * to implement in Phase 2 (line-for-line mirror of benchmark.infer()).
 * Phase 1 bodies are no-ops so the shared library compiles and links.
 * Calling infer_batch() before Phase 2 fills these in WILL produce
 * garbage predictions — that is expected. ======================= */

/* 1) Patch embed (CPU INT8) — matches benchmark.infer() patch_embedding
 *    block. signal layout in C: (1024, 2) float32, indexed signal[t*2+ch]. */
static void cpu_patch_embed(const float *signal, uint8_t *emb_q) {
    /* Quantize + im2col into patches (64, 32) int8.
     * patches[p, ch*16 + k] = clip(round(signal[p*16+k, ch] / emb_in), -128, 127) */
    int8_t patches[M_FULL * 32];
    for (int p = 0; p < M_FULL; p++) {
        for (int ch = 0; ch < 2; ch++) {
            for (int k = 0; k < 16; k++) {
                double s = (double)signal[(p * 16 + k) * 2 + ch];
                patches[p * 32 + ch * 16 + k] = clip_round_i8(s / SCALE_EMB_IN, -128, 127);
            }
        }
    }
    /* GEMM: acc[p, j] = sum_k patches[p, k] * W_emb_flat[j, k] + bias_emb_int32[j] */
    const double scale = SCALE_EMB_W * SCALE_EMB_IN;
    const double inv_emb_out = 1.0 / SCALE_EMB_OUT;
    for (int p = 0; p < M_FULL; p++) {
        for (int j = 0; j < D_MODEL; j++) {
            int32_t acc = g_ctx.bias_emb_int32[j];
            const int8_t *W_row = g_ctx.W_emb_flat + j * 32;
            for (int k = 0; k < 32; k++)
                acc += (int32_t)patches[p * 32 + k] * (int32_t)W_row[k];
            /* Dequant + BN + ReLU + uint8 quant */
            double f = (double)acc * scale;
            double bn = (f - (double)g_ctx.bn_emb_mean[j])
                        / sqrt((double)g_ctx.bn_emb_var[j] + 1e-5);
            if (bn < 0.0) bn = 0.0;
            double q = rint(bn * inv_emb_out);
            if (q > 255) q = 255;
            else if (q < 0) q = 0;
            emb_q[p * D_MODEL + j] = (uint8_t)q;
        }
    }
}

/* 2) Positional add: see deployment_config "positional_encoding" notes. */
static void cpu_positional(const uint8_t *emb_q, int8_t *after_pos) {
    const double inv_pos_in  = 1.0 / SCALE_POS_IN;
    const double inv_pos_out = 1.0 / SCALE_POS_OUT;
    for (int i = 0; i < M_FULL * D_MODEL; i++) {
        double emb_for_pos = (double)emb_q[i] * SCALE_EMB_OUT;
        int32_t emb_q2 = clip_round_i32(emb_for_pos * inv_pos_in, -128, 127);
        int32_t pos_q  = clip_round_i32((double)g_ctx.pos2d[i] * inv_pos_in, -128, 127);
        double sum_f = ((double)emb_q2 + (double)pos_q) * SCALE_POS_IN;
        after_pos[i] = clip_round_i8(sum_f * inv_pos_out, -128, 127);
    }
}

/* 3) BN_attn + INT4 quant. */
static void cpu_bn_attn(const int8_t *after_pos, int8_t *pre_int4) {
    const double inv_pre_out = 1.0 / SCALE_ATTN_PRE_OUT;
    for (int p = 0; p < M_FULL; p++) {
        for (int j = 0; j < D_MODEL; j++) {
            double x = (double)after_pos[p * D_MODEL + j] * SCALE_POS_OUT;
            double bn = (x - (double)g_ctx.bn_attn_mean[j])
                        / sqrt((double)g_ctx.bn_attn_var[j] + 1e-5);
            pre_int4[p * D_MODEL + j] = clip_round_i8(bn * inv_pre_out, -8, 7);
        }
    }
}

/* 4) head_split: Qh[h, m, d] = Q_int4[m, h*D_HEAD + d]. */
static void head_split(const int8_t *Q_int4, const int8_t *K_int4, const int8_t *V_int4,
                       int8_t *Qh, int8_t *Kh, int8_t *Vh) {
    for (int h = 0; h < N_HEADS; h++) {
        for (int m = 0; m < M_FULL; m++) {
            for (int d = 0; d < D_HEAD; d++) {
                int dst = h * M_FULL * D_HEAD + m * D_HEAD + d;
                int src = m * D_MODEL + h * D_HEAD + d;
                Qh[dst] = Q_int4[src];
                Kh[dst] = K_int4[src];
                Vh[dst] = V_int4[src];
            }
        }
    }
}

/* 5) cpu_requant_after_vta — EXACT mirror of benchmark.cpu_requant_after_vta.
 *      recovered = x_int8 * 2^coarse_shift
 *      real      = recovered * (w_scale * in_scale * extra_factor)
 *      out_int   = clip(round(real / out_scale), clip_lo, clip_hi)    */
static void cpu_requant(const int8_t *x_int8, int8_t *out_int4,
                         int n_elements, int coarse_shift,
                         double w_scale, double in_scale, double out_scale,
                         int clip_lo, int clip_hi, double extra_factor) {
    const double scale = (double)(1 << coarse_shift) * w_scale * in_scale * extra_factor;
    const double inv_out = 1.0 / out_scale;
    for (int i = 0; i < n_elements; i++) {
        double real = (double)x_int8[i] * scale;
        out_int4[i] = clip_round_i8(real * inv_out, clip_lo, clip_hi);
    }
}

/* 6) tile_and_pack_runtime_w: tile W (W_rows, W_cols) into m_tiles slices,
 *    pack each to int4 nibbles, and populate slice_ptrs[i] to point at the
 *    static internal storage. The static storage is overwritten on each
 *    call — caller MUST consume slices before the next invocation.
 *
 *    Tile layout: tiled[m, n, i, k] = W[m*BLOCK + i, n*BLOCK + k]; pack_int4
 *    operates on flat bytes in that order.
 *    Max W footprint here: V_T (32, 64) and K (64, 32) — both 2048 bytes
 *    of slice storage (m_tiles*n_tiles*BLOCK*BLOCK = 4*2*256 or 2*4*256). */
static void tile_and_pack_runtime_w(const int8_t *W, int W_rows, int W_cols,
                                     int8_t **slice_ptrs) {
    static int8_t s_storage[8 * 1024];
    static int8_t s_unpacked[2 * 1024];   /* one slice's worth, unpacked */
    int m_tiles = W_rows / BLOCK;
    int n_tiles = W_cols / BLOCK;
    size_t slice_n = (size_t)n_tiles * BLOCK * BLOCK;
    if ((size_t)m_tiles * slice_n > sizeof(s_storage) || slice_n > sizeof(s_unpacked)) {
        fprintf(stderr, "tile_and_pack_runtime_w: storage too small for (%d, %d)\n",
                W_rows, W_cols);
        exit(1);
    }
    for (int m = 0; m < m_tiles; m++) {
        /* Tile this m-slice (n, i, k) -> W[m*BLOCK + i, n*BLOCK + k]. */
        for (int n = 0; n < n_tiles; n++) {
            for (int i = 0; i < BLOCK; i++) {
                for (int k = 0; k < BLOCK; k++) {
                    s_unpacked[n * BLOCK * BLOCK + i * BLOCK + k] =
                        W[(m * BLOCK + i) * W_cols + n * BLOCK + k];
                }
            }
        }
        slice_ptrs[m] = s_storage + (size_t)m * slice_n;
        pack_int4(s_unpacked, slice_ptrs[m], slice_n);
    }
}

/* 7) softmax: per row of (rows, cols) int4 scores -> float softmax ->
 *    requant to int4 [-8, 7] via softmax_in / softmax_out. */
static void cpu_softmax_then_quant(const int8_t *scores_int4,
                                    int rows, int cols,
                                    int8_t *attn_int4) {
    static double row[64];   /* max cols = M_FULL = 64 */
    const double inv_softmax_out = 1.0 / SCALE_SOFTMAX_OUT;
    for (int r = 0; r < rows; r++) {
        double maxv = -INFINITY;
        for (int c = 0; c < cols; c++) {
            row[c] = (double)scores_int4[r * cols + c] * SCALE_SOFTMAX_IN;
            if (row[c] > maxv) maxv = row[c];
        }
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            row[c] = exp(row[c] - maxv);
            sum += row[c];
        }
        double inv_sum = 1.0 / sum;
        for (int c = 0; c < cols; c++) {
            attn_int4[r * cols + c] =
                clip_round_i8(row[c] * inv_sum * inv_softmax_out, -8, 7);
        }
    }
}

/* 8) head_concat: ctx_cat[m, h*D_HEAD + d] = av_int4[h, m, d]. */
static void head_concat(const int8_t *av_int4, int8_t *ctx_cat) {
    for (int m = 0; m < M_FULL; m++) {
        for (int h = 0; h < N_HEADS; h++) {
            for (int d = 0; d < D_HEAD; d++) {
                ctx_cat[m * D_MODEL + h * D_HEAD + d] =
                    av_int4[h * M_FULL * D_HEAD + m * D_HEAD + d];
            }
        }
    }
}

/* 9) Attention residual: int32 sum of o_int4 + skip (skip = INT4-quantized
 *    after_pos at attn_residual scale).  No further requant. */
static void cpu_attn_residual(const int8_t *o_int4, const int8_t *after_pos,
                               int32_t *attn_block_out) {
    const double inv_residual = 1.0 / SCALE_ATTN_RESIDUAL;
    for (int i = 0; i < M_FULL * D_MODEL; i++) {
        double s = (double)after_pos[i] * SCALE_POS_OUT;
        int32_t skip = clip_round_i32(s * inv_residual, -8, 7);
        attn_block_out[i] = (int32_t)o_int4[i] + skip;
    }
}

/* 10) BN_mlp + INT4 quant (same pattern as bn_attn). */
static void cpu_bn_mlp(const int32_t *attn_block_out, int8_t *mlp_pre) {
    const double inv_bn_out = 1.0 / SCALE_MLP_BN_OUT;
    for (int p = 0; p < M_FULL; p++) {
        for (int j = 0; j < D_MODEL; j++) {
            double x = (double)attn_block_out[p * D_MODEL + j] * SCALE_ATTN_RESIDUAL;
            double bn = (x - (double)g_ctx.bn_mlp_mean[j])
                        / sqrt((double)g_ctx.bn_mlp_var[j] + 1e-5);
            mlp_pre[p * D_MODEL + j] = clip_round_i8(bn * inv_bn_out, -8, 7);
        }
    }
}

/* 11) fc1 post-VTA: requant to unsigned [0, 15], shift to signed [-8, 7]. */
static void cpu_fc1_postvta(const int8_t *fc1_out8, int8_t *fc1_signed) {
    const double scale = (double)(1 << SH_FC1) * SCALE_MLP_BN_OUT * SCALE_FC1_W;
    const double inv_fc1_out = 1.0 / SCALE_FC1_OUT;
    for (int i = 0; i < M_FULL * D_FF; i++) {
        double real = (double)fc1_out8[i] * scale;
        int32_t unsigned_int4 = clip_round_i32(real * inv_fc1_out, 0, 15);
        fc1_signed[i] = (int8_t)(unsigned_int4 - 8);     /* -8..7 */
    }
}

/* MLP residual buffer — file-scope static so cpu_residual_mlp and
 * cpu_classifier can share it without changing inference math. Same .bss
 * footprint as the previous local-static inside cpu_classifier. */
static int32_t s_mlp_block[M_FULL * D_MODEL];

/* 12a) MLP residual: fc2_int4 + skip(attn_block_out at mlp_residual scale).
 *      Writes to s_mlp_block. */
static void cpu_residual_mlp(const int8_t *fc2_int4, const int32_t *attn_block_out) {
    const double inv_mlp_residual = 1.0 / SCALE_MLP_RESIDUAL;
    for (int i = 0; i < M_FULL * D_MODEL; i++) {
        double s = (double)attn_block_out[i] * SCALE_ATTN_RESIDUAL;
        int32_t skip = clip_round_i32(s * inv_mlp_residual, -8, 7);
        s_mlp_block[i] = (int32_t)fc2_int4[i] + skip;
    }
}

/* 12b) Classifier: GAP over rows -> dequant via mlp_residual ->
 *      logits = gap @ W_cls_float.T + b_cls -> argmax. Reads s_mlp_block. */
static int cpu_classifier(void) {
    double gap[D_MODEL];
    for (int j = 0; j < D_MODEL; j++) gap[j] = 0.0;
    for (int p = 0; p < M_FULL; p++) {
        for (int j = 0; j < D_MODEL; j++) {
            gap[j] += (double)s_mlp_block[p * D_MODEL + j] * SCALE_MLP_RESIDUAL;
        }
    }
    const double inv_M = 1.0 / (double)M_FULL;
    for (int j = 0; j < D_MODEL; j++) gap[j] *= inv_M;
    double logits[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) {
        double s = (double)g_ctx.b_cls[c];
        const double *W_row = g_ctx.W_cls_float + c * D_MODEL;
        for (int j = 0; j < D_MODEL; j++) s += gap[j] * W_row[j];
        logits[c] = s;
    }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (logits[c] > logits[best]) best = c;
    return best;
}


/* ====================== Per-sample inference =========================== */
static int infer_one(const float *signal /* (1024, 2) */) {
    /* Static intermediate buffers — same lifetimes as the Python locals.
     * Total live memory ~120 KB; fine for the runner's stack/.bss. */
    static uint8_t emb_q   [M_FULL * D_MODEL];
    static int8_t  after_pos[M_FULL * D_MODEL];
    static int8_t  pre_int4 [M_FULL * D_MODEL];
    static int8_t  Q_int8   [M_FULL * D_MODEL], K_int8[M_FULL * D_MODEL], V_int8[M_FULL * D_MODEL];
    static int8_t  Q_int4   [M_FULL * D_MODEL], K_int4[M_FULL * D_MODEL], V_int4[M_FULL * D_MODEL];
    static int8_t  Qh       [N_HEADS * M_FULL * D_HEAD];
    static int8_t  Kh       [N_HEADS * M_FULL * D_HEAD];
    static int8_t  Vh       [N_HEADS * M_FULL * D_HEAD];
    static int8_t  qkt_int8 [M_FULL * M_FULL];     /* per head (64,64) */
    static int8_t  presoftmax[M_FULL * M_FULL];
    static int8_t  attn_int4[M_FULL * M_FULL];
    static int8_t  av_int8_h[M_FULL * D_HEAD];
    static int8_t  av_int4_all[N_HEADS * M_FULL * D_HEAD];
    static int8_t  ctx_cat  [M_FULL * D_MODEL];
    static int8_t  o_int8   [M_FULL * D_MODEL];
    static int8_t  o_int4   [M_FULL * D_MODEL];
    static int32_t attn_block_out[M_FULL * D_MODEL];
    static int8_t  mlp_pre  [M_FULL * D_MODEL];
    static int8_t  fc1_out8 [M_FULL * D_FF];
    static int8_t  fc1_signed[M_FULL * D_FF];
    static int8_t  fc2_out8 [M_FULL * D_MODEL];
    static int8_t  fc2_int4 [M_FULL * D_MODEL];

    struct timespec t0, t1;

    /* 1. Patch embedding (CPU) — T_PATCH_EMBED */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_patch_embed(signal, emb_q);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_PATCH_EMBED] += diff_ms(t0, t1);

    /* 2. Positional encoding (CPU) — T_POSITIONAL */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_positional(emb_q, after_pos);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_POSITIONAL] += diff_ms(t0, t1);

    /* 3. BN_attn + INT4 quant (CPU) — T_BN_ATTN */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_bn_attn(after_pos, pre_int4);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_BN_ATTN] += diff_ms(t0, t1);

    /* 4. Q / K / V projections (VTA, M-chunked) — T_VTA_QKV */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gemm_proj(&g_ctx.proj_s3, pre_int4, g_ctx.W_q_slices, Q_int8);
    gemm_proj(&g_ctx.proj_s3, pre_int4, g_ctx.W_k_slices, K_int8);
    gemm_proj(&g_ctx.proj_s2, pre_int4, g_ctx.W_v_slices, V_int8);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_VTA_QKV] += diff_ms(t0, t1);

    /* 5a. CPU requant Q/K/V (CPU) — T_CPU_REQUANT_QKV */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_requant(Q_int8, Q_int4, M_FULL * D_MODEL, g_ctx.sh_q,
                g_ctx.q_w, g_ctx.attn_pre_out, g_ctx.q_out, -8, 7, 1.0f);
    cpu_requant(K_int8, K_int4, M_FULL * D_MODEL, g_ctx.sh_k,
                g_ctx.k_w, g_ctx.attn_pre_out, g_ctx.k_out, -8, 7, 1.0f);
    cpu_requant(V_int8, V_int4, M_FULL * D_MODEL, g_ctx.sh_v,
                g_ctx.v_w, g_ctx.attn_pre_out, g_ctx.v_out, -8, 7, 1.0f);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CPU_REQUANT_QKV] += diff_ms(t0, t1);

    /* 5b. Head split (CPU) — T_CPU_HEAD_SPLIT */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    head_split(Q_int4, K_int4, V_int4, Qh, Kh, Vh);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CPU_HEAD_SPLIT] += diff_ms(t0, t1);

    /* 6. Per-head attention. Per-head sub-stages are accumulated across
     * the 3 heads into T_VTA_QKT / T_CPU_SOFTMAX / T_VTA_AV /
     * T_CPU_HEAD_CONCAT (the av-requant portion of head_concat). */
    for (int h = 0; h < N_HEADS; h++) {
        const int8_t *Q_h = &Qh[h * M_FULL * D_HEAD];
        const int8_t *K_h = &Kh[h * M_FULL * D_HEAD];
        const int8_t *V_h = &Vh[h * M_FULL * D_HEAD];

        /* Q @ K^T (VTA): tile K_h + VTA call */
        int8_t *K_slices[4];
        clock_gettime(CLOCK_MONOTONIC, &t0);
        tile_and_pack_runtime_w(K_h, 64, 32, K_slices);
        gemm_runtime_w(&g_ctx.qkt, Q_h, K_slices, qkt_int8);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_timings[T_VTA_QKT] += diff_ms(t0, t1);

        /* CPU requant of qkt + softmax (CPU) — T_CPU_SOFTMAX */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        cpu_requant(qkt_int8, presoftmax, M_FULL * M_FULL, g_ctx.sh_qk,
                    g_ctx.q_out, g_ctx.k_out, g_ctx.softmax_in, -8, 7, SQRT_96);
        cpu_softmax_then_quant(presoftmax, M_FULL, M_FULL, attn_int4);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_timings[T_CPU_SOFTMAX] += diff_ms(t0, t1);

        /* attn @ V (VTA): V_T transpose + tile + VTA call */
        int8_t *V_slices[2];
        clock_gettime(CLOCK_MONOTONIC, &t0);
        static int8_t V_T[D_HEAD * M_FULL];
        for (int row = 0; row < D_HEAD; row++)
            for (int col = 0; col < M_FULL; col++)
                V_T[row * M_FULL + col] = V_h[col * D_HEAD + row];
        tile_and_pack_runtime_w(V_T, D_HEAD, M_FULL, V_slices);
        gemm_runtime_w(&g_ctx.av, attn_int4, V_slices, av_int8_h);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_timings[T_VTA_AV] += diff_ms(t0, t1);

        /* Per-head av requant (CPU) — accumulated into T_CPU_HEAD_CONCAT */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        cpu_requant(av_int8_h, &av_int4_all[h * M_FULL * D_HEAD],
                    M_FULL * D_HEAD, g_ctx.sh_av,
                    g_ctx.softmax_out, g_ctx.v_out, g_ctx.o_in, -8, 7, 1.0f);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_timings[T_CPU_HEAD_CONCAT] += diff_ms(t0, t1);
    }

    /* 7a. Concat heads (CPU) — also T_CPU_HEAD_CONCAT */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    head_concat(av_int4_all, ctx_cat);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CPU_HEAD_CONCAT] += diff_ms(t0, t1);

    /* 7b. O projection (VTA) — T_VTA_O_PROJ */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gemm_proj(&g_ctx.proj_s3, ctx_cat, g_ctx.W_o_slices, o_int8);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_VTA_O_PROJ] += diff_ms(t0, t1);

    /* 7c+8. CPU requant O + attention residual (CPU) — T_CPU_RESID_ATTN */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_requant(o_int8, o_int4, M_FULL * D_MODEL, g_ctx.sh_o,
                g_ctx.o_in, g_ctx.o_w, g_ctx.attn_residual, -8, 7, 1.0f);
    cpu_attn_residual(o_int4, after_pos, attn_block_out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CPU_RESID_ATTN] += diff_ms(t0, t1);

    /* 9. BN_mlp + INT4 quant (CPU) — T_BN_MLP */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_bn_mlp(attn_block_out, mlp_pre);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_BN_MLP] += diff_ms(t0, t1);

    /* 10a. fc1 VTA — T_VTA_FC1 */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gemm_mlp(&g_ctx.fc1, mlp_pre, g_ctx.W_fc1_slices,
             (const int32_t * const *)g_ctx.bias_fc1_per_call, fc1_out8);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_VTA_FC1] += diff_ms(t0, t1);

    /* 10b. ReLU + unsigned quant + ZP shift (CPU) — T_CPU_RELU_QUANT */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_fc1_postvta(fc1_out8, fc1_signed);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CPU_RELU_QUANT] += diff_ms(t0, t1);

    /* 11a. fc2 VTA — T_VTA_FC2 (CPU requant of fc2 lives in T_CPU_RESID_MLP) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gemm_mlp(&g_ctx.fc2, fc1_signed, g_ctx.W_fc2_slices,
             (const int32_t * const *)g_ctx.bias_fc2_per_call, fc2_out8);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_VTA_FC2] += diff_ms(t0, t1);

    /* 11b+12a. fc2 requant + MLP residual (CPU) — T_CPU_RESID_MLP */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_requant(fc2_out8, fc2_int4, M_FULL * D_MODEL, g_ctx.sh_fc2,
                g_ctx.fc1_out, g_ctx.fc2_w, g_ctx.mlp_residual, -8, 7, 1.0f);
    cpu_residual_mlp(fc2_int4, attn_block_out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CPU_RESID_MLP] += diff_ms(t0, t1);

    /* 12b. Classifier (GAP + matmul + argmax, CPU) — T_CLASSIFIER */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int pred = cpu_classifier();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_timings[T_CLASSIFIER] += diff_ms(t0, t1);

    return pred;
}


/* ====================== Loader helpers ================================= */
/* Static-weight loader: reads <weights>/<gemm>_W_tiled.npy (int8 shape
 * (m_full, n_tiles, BLOCK, BLOCK)), pre-packs each m-slice with pack_int4,
 * and stores pointers in *out_slices (one outer alloc + one backing alloc).
 * Convention for cleanup: *out_slices[0] is the start of the contiguous
 * backing buffer; cleanup frees (*out_slices)[0] then (*out_slices). */
static int load_static_w(const char *export_dir, const char *gemm,
                          int m_full, int n_tiles, int8_t ***out_slices) {
    char path[512];
    snprintf(path, sizeof(path), "%s/weights/%s_W_tiled.npy", export_dir, gemm);
    NpyArray w;
    if (npy_load_typed(path, &w, "<i1") != 0) return -1;
    if (w.ndim != 4 || w.shape[0] != m_full || w.shape[1] != n_tiles ||
        w.shape[2] != BLOCK || w.shape[3] != BLOCK) {
        fprintf(stderr, "load_static_w: %s shape %lld,%lld,%lld,%lld != (%d,%d,%d,%d)\n",
                path, (long long)w.shape[0], (long long)w.shape[1],
                (long long)w.shape[2], (long long)w.shape[3],
                m_full, n_tiles, BLOCK, BLOCK);
        npy_free(&w); return -1;
    }
    size_t slice_n = (size_t)n_tiles * BLOCK * BLOCK;
    int8_t  *backing = (int8_t *)malloc((size_t)m_full * slice_n);
    int8_t **slices  = (int8_t **)malloc((size_t)m_full * sizeof(int8_t *));
    if (!backing || !slices) {
        fprintf(stderr, "load_static_w: malloc failed for %s\n", path);
        free(backing); free(slices); npy_free(&w); return -1;
    }
    const int8_t *src = (const int8_t *)w.data;
    for (int m = 0; m < m_full; m++) {
        slices[m] = backing + (size_t)m * slice_n;
        pack_int4(src + (size_t)m * slice_n, slices[m], slice_n);
    }
    npy_free(&w);
    *out_slices = slices;
    return 0;
}

/* Real bias loader: reads <weights>/<file> int32 shape (n_calls, BLOCK),
 * broadcasts each row to (o_tile, BLOCK) and stores n_calls per-call
 * buffers. Cleanup convention: per_call[0] is backing buffer. */
static int load_real_bias(const char *export_dir, const char *file,
                           int n_calls, int o_tile, int32_t ***out_per_call) {
    char path[512];
    snprintf(path, sizeof(path), "%s/weights/%s", export_dir, file);
    NpyArray b;
    if (npy_load_typed(path, &b, "<i4") != 0) return -1;
    if (b.ndim != 2 || b.shape[0] != n_calls || b.shape[1] != BLOCK) {
        fprintf(stderr, "load_real_bias: %s shape %lld,%lld != (%d, %d)\n",
                path, (long long)b.shape[0], (long long)b.shape[1], n_calls, BLOCK);
        npy_free(&b); return -1;
    }
    size_t per_call_n = (size_t)o_tile * BLOCK;
    int32_t  *backing  = (int32_t *)malloc((size_t)n_calls * per_call_n * sizeof(int32_t));
    int32_t **per_call = (int32_t **)malloc((size_t)n_calls * sizeof(int32_t *));
    if (!backing || !per_call) {
        fprintf(stderr, "load_real_bias: malloc failed\n");
        free(backing); free(per_call); npy_free(&b); return -1;
    }
    const int32_t *src = (const int32_t *)b.data;
    for (int n = 0; n < n_calls; n++) {
        per_call[n] = backing + (size_t)n * per_call_n;
        for (int row = 0; row < o_tile; row++) {
            memcpy(per_call[n] + (size_t)row * BLOCK,
                   src + (size_t)n * BLOCK, BLOCK * sizeof(int32_t));
        }
    }
    npy_free(&b);
    *out_per_call = per_call;
    return 0;
}

static int load_cpu_f32(const char *export_dir, const char *name,
                         size_t expect_n, float **out) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpu_params/%s", export_dir, name);
    NpyArray a;
    if (npy_load_typed(path, &a, "<f4") != 0) return -1;
    if (a.n_elements != expect_n) {
        fprintf(stderr, "load_cpu_f32: %s expected %zu got %zu\n", path, expect_n, a.n_elements);
        npy_free(&a); return -1;
    }
    *out = (float *)a.data;     /* transfer ownership; freed in cleanup_runtime */
    return 0;
}

static int load_cpu_i8(const char *export_dir, const char *name,
                        size_t expect_n, int8_t **out) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpu_params/%s", export_dir, name);
    NpyArray a;
    if (npy_load_typed(path, &a, "<i1") != 0) return -1;
    if (a.n_elements != expect_n) {
        fprintf(stderr, "load_cpu_i8: %s expected %zu got %zu\n", path, expect_n, a.n_elements);
        npy_free(&a); return -1;
    }
    *out = (int8_t *)a.data;
    return 0;
}


/* ====================== Init / cleanup / main =========================== */
static int init_runtime(const char *export_dir) {
    if (load_libvta() != 0) return 1;

    /* Module geometry — o32 build (transformer_export_o32/config.json).
     * o_tile bumped where o×n stays <= 200 (PL safe zone), to halve the
     * M-chunk count for proj/qkt/av and fc1. fc2 unchanged (o×n=192 already
     * at the safe bound). Hardcoded since config doesn't vary per inference;
     * see transformer_export_o32/config.json["modules"] for the source. */
    g_ctx.proj_s3 = (VtaModule){.name = "proj_k96_s3_m1", .M_full = 64, .K =  96, .N_full = 96, .o_tile = 32, .n_tiles =  6, .n_m_chunks = 2, .n_calls_per_gemm =  6};
    g_ctx.proj_s2 = (VtaModule){.name = "proj_k96_s2_m1", .M_full = 64, .K =  96, .N_full = 96, .o_tile = 32, .n_tiles =  6, .n_m_chunks = 2, .n_calls_per_gemm =  6};
    g_ctx.qkt     = (VtaModule){.name = "qkt_s3_m1",      .M_full = 64, .K =  32, .N_full = 64, .o_tile = 32, .n_tiles =  2, .n_m_chunks = 2, .n_calls_per_gemm =  4};
    g_ctx.av      = (VtaModule){.name = "av_s3_m1",       .M_full = 64, .K =  64, .N_full = 32, .o_tile = 32, .n_tiles =  4, .n_m_chunks = 2, .n_calls_per_gemm =  2};
    g_ctx.fc1     = (VtaModule){.name = "fc1_s3_m1",      .M_full = 64, .K =  96, .N_full = 384,.o_tile = 32, .n_tiles =  6, .n_m_chunks = 2, .n_calls_per_gemm = 24, .has_real_bias = 1};
    g_ctx.fc2     = (VtaModule){.name = "fc2_s4_m1",      .M_full = 64, .K = 384, .N_full = 96, .o_tile =  8, .n_tiles = 24, .n_m_chunks = 8, .n_calls_per_gemm =  6, .has_real_bias = 1};

    /* Load + alloc all 6 modules */
    load_module(&g_ctx.proj_s3, export_dir);
    load_module(&g_ctx.proj_s2, export_dir);
    load_module(&g_ctx.qkt,     export_dir);
    load_module(&g_ctx.av,      export_dir);
    load_module(&g_ctx.fc1,     export_dir);
    load_module(&g_ctx.fc2,     export_dir);

    /* Static weights — m_full per W is N_full/BLOCK_OUT for the call site
     * that uses it: q/k/o use proj_s3 (m=6), v uses proj_s2 (m=6),
     * fc1 (m=24), fc2 (m=6). */
    if (load_static_w(export_dir, "q_proj", 6,  6,  &g_ctx.W_q_slices)   != 0) return -1;
    if (load_static_w(export_dir, "k_proj", 6,  6,  &g_ctx.W_k_slices)   != 0) return -1;
    if (load_static_w(export_dir, "v_proj", 6,  6,  &g_ctx.W_v_slices)   != 0) return -1;
    if (load_static_w(export_dir, "o_proj", 6,  6,  &g_ctx.W_o_slices)   != 0) return -1;
    if (load_static_w(export_dir, "fc1",    24, 6,  &g_ctx.W_fc1_slices) != 0) return -1;
    if (load_static_w(export_dir, "fc2",    6,  24, &g_ctx.W_fc2_slices) != 0) return -1;

    /* Real biases (fc1/fc2). Disk shapes (24, 16) and (6, 16); broadcast
     * per-call to (o_tile, BLOCK). o32 build: fc1.o_tile=16, fc2.o_tile=8. */
    if (load_real_bias(export_dir, "fc1_bias_int32.npy",            24, 32,
                        &g_ctx.bias_fc1_per_call) != 0) return -1;
    if (load_real_bias(export_dir, "fc2_bias_int32_corrected.npy",  6,  8,
                        &g_ctx.bias_fc2_per_call) != 0) return -1;

    /* CPU params */
    if (load_cpu_i8 (export_dir, "emb_conv_W_int.npy",  D_MODEL * 32, &g_ctx.W_emb_flat   ) != 0) return -1;
    float *b_emb = NULL;
    if (load_cpu_f32(export_dir, "emb_conv_bias.npy",   D_MODEL,      &b_emb              ) != 0) return -1;
    if (load_cpu_f32(export_dir, "bn_emb_mean.npy",     D_MODEL,      &g_ctx.bn_emb_mean  ) != 0) return -1;
    if (load_cpu_f32(export_dir, "bn_emb_var.npy",      D_MODEL,      &g_ctx.bn_emb_var   ) != 0) return -1;
    float *pos_full = NULL;
    if (load_cpu_f32(export_dir, "pos_enc.npy",         M_FULL * D_MODEL, &pos_full       ) != 0) return -1;
    g_ctx.pos2d = pos_full;     /* pos_enc.npy stored as (1, 64, 96) flat = (64, 96) */
    if (load_cpu_f32(export_dir, "bn_attn_mean.npy",    D_MODEL,      &g_ctx.bn_attn_mean ) != 0) return -1;
    if (load_cpu_f32(export_dir, "bn_attn_var.npy",     D_MODEL,      &g_ctx.bn_attn_var  ) != 0) return -1;
    if (load_cpu_f32(export_dir, "bn_mlp_mean.npy",     D_MODEL,      &g_ctx.bn_mlp_mean  ) != 0) return -1;
    if (load_cpu_f32(export_dir, "bn_mlp_var.npy",      D_MODEL,      &g_ctx.bn_mlp_var   ) != 0) return -1;
    if (load_cpu_i8 (export_dir, "cls_W_int.npy",       N_CLASSES * D_MODEL, &g_ctx.W_cls ) != 0) return -1;
    if (load_cpu_f32(export_dir, "cls_bias.npy",        N_CLASSES,    &g_ctx.b_cls        ) != 0) return -1;

    /* Populate g_ctx scales/shifts from compile-time constants (matches
     * scales.json + config.json["tuned_shifts"]). */
    g_ctx.emb_in        = SCALE_EMB_IN;        g_ctx.emb_w         = SCALE_EMB_W;
    g_ctx.emb_out       = SCALE_EMB_OUT;
    g_ctx.pos_in        = SCALE_POS_IN;        g_ctx.pos_out       = SCALE_POS_OUT;
    g_ctx.attn_pre_out  = SCALE_ATTN_PRE_OUT;
    g_ctx.q_w           = SCALE_Q_W;           g_ctx.q_out         = SCALE_Q_OUT;
    g_ctx.k_w           = SCALE_K_W;           g_ctx.k_out         = SCALE_K_OUT;
    g_ctx.v_w           = SCALE_V_W;           g_ctx.v_out         = SCALE_V_OUT;
    g_ctx.softmax_in    = SCALE_SOFTMAX_IN;    g_ctx.softmax_out   = SCALE_SOFTMAX_OUT;
    g_ctx.o_in          = SCALE_O_IN;          g_ctx.o_w           = SCALE_O_W;
    g_ctx.attn_residual = SCALE_ATTN_RESIDUAL; g_ctx.mlp_bn_out    = SCALE_MLP_BN_OUT;
    g_ctx.fc1_w         = SCALE_FC1_W;         g_ctx.fc1_out       = SCALE_FC1_OUT;
    g_ctx.fc2_w         = SCALE_FC2_W;         g_ctx.mlp_residual  = SCALE_MLP_RESIDUAL;
    g_ctx.cls_w         = SCALE_CLS_W;         g_ctx.cls_out       = SCALE_CLS_OUT;
    g_ctx.sh_q  = SH_Q;  g_ctx.sh_k  = SH_K;  g_ctx.sh_v  = SH_V;
    g_ctx.sh_qk = SH_QK; g_ctx.sh_av = SH_AV; g_ctx.sh_o  = SH_O;
    g_ctx.sh_fc1 = SH_FC1; g_ctx.sh_fc2 = SH_FC2;

    /* Precompute: bias_emb_int32 = round(b_emb / (emb_w * emb_in)), int32. */
    {
        const double s = SCALE_EMB_W * SCALE_EMB_IN;
        for (int j = 0; j < D_MODEL; j++) {
            g_ctx.bias_emb_int32[j] = clip_round_i32((double)b_emb[j] / s, INT32_MIN, INT32_MAX);
        }
        free(b_emb);
    }

    /* Precompute: W_cls_float = W_cls.astype(double) * cls_w. */
    for (int i = 0; i < N_CLASSES * D_MODEL; i++) {
        g_ctx.W_cls_float[i] = (double)g_ctx.W_cls[i] * SCALE_CLS_W;
    }

    return 0;
}


static void free_module(VtaModule *vm) {
    if (vm->A_dl)  TVMArrayFree(vm->A_dl);
    if (vm->W_dl)  TVMArrayFree(vm->W_dl);
    if (vm->D_dl)  TVMArrayFree(vm->D_dl);
    if (vm->C_dl)  TVMArrayFree(vm->C_dl);
    if (vm->A_cpu) TVMArrayFree(vm->A_cpu);
    if (vm->W_cpu) TVMArrayFree(vm->W_cpu);
    if (vm->D_cpu) TVMArrayFree(vm->D_cpu);
    if (vm->C_cpu) TVMArrayFree(vm->C_cpu);
    if (vm->mod)   TVMModFree(vm->mod);
}

static void free_w_slices(int8_t **slices) {
    if (slices) {
        if (slices[0]) free(slices[0]);   /* contiguous backing storage */
        free(slices);
    }
}
static void free_bias_per_call(int32_t **per_call) {
    if (per_call) {
        if (per_call[0]) free(per_call[0]);
        free(per_call);
    }
}

static void cleanup_runtime(void) {
    free_module(&g_ctx.proj_s3);
    free_module(&g_ctx.proj_s2);
    free_module(&g_ctx.qkt);
    free_module(&g_ctx.av);
    free_module(&g_ctx.fc1);
    free_module(&g_ctx.fc2);
    free_w_slices(g_ctx.W_q_slices);
    free_w_slices(g_ctx.W_k_slices);
    free_w_slices(g_ctx.W_v_slices);
    free_w_slices(g_ctx.W_o_slices);
    free_w_slices(g_ctx.W_fc1_slices);
    free_w_slices(g_ctx.W_fc2_slices);
    free_bias_per_call(g_ctx.bias_fc1_per_call);
    free_bias_per_call(g_ctx.bias_fc2_per_call);
    free(g_ctx.W_emb_flat);
    free(g_ctx.bn_emb_mean);  free(g_ctx.bn_emb_var);
    free(g_ctx.pos2d);
    free(g_ctx.bn_attn_mean); free(g_ctx.bn_attn_var);
    free(g_ctx.bn_mlp_mean);  free(g_ctx.bn_mlp_var);
    free(g_ctx.W_cls);        free(g_ctx.b_cls);
    if (g_libvta) dlclose(g_libvta);
}


/* ====================== Timing breakdown printer ====================== */
static void print_timings(int n_samples) {
    if (n_samples <= 0) return;
    const double inv_n = 1.0 / (double)n_samples;
    double total = 0.0, total_vta = 0.0, total_cpu = 0.0;
    for (int s = 0; s < T_NUM_STAGES; s++) {
        total += g_timings[s];
        if (g_stage_types[s][0] == 'V') total_vta += g_timings[s];
        else                            total_cpu += g_timings[s];
    }
    const double total_per = total * inv_n;

    printf("\n");
    printf("Stage                   Type    ms/inf      %%total\n");
    printf("---------------------   -----   ---------   -------\n");
    for (int s = 0; s < T_NUM_STAGES; s++) {
        double ms_per = g_timings[s] * inv_n;
        double pct = total > 0 ? 100.0 * g_timings[s] / total : 0.0;
        printf("%-22s  %-5s   %8.3f    %5.1f%%\n",
               g_stage_names[s], g_stage_types[s], ms_per, pct);
    }
    printf("---------------------   -----   ---------   -------\n");
    double pct_vta = total > 0 ? 100.0 * total_vta / total : 0.0;
    double pct_cpu = total > 0 ? 100.0 * total_cpu / total : 0.0;
    printf("Total VTA: %8.3f ms (%.1f%%)\n", total_vta * inv_n, pct_vta);
    printf("Total CPU: %8.3f ms (%.1f%%)\n", total_cpu * inv_n, pct_cpu);
    printf("Total:     %8.3f ms\n", total_per);
}


/* ====================== CLI argument parsing =========================== */
static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s --weights <export_dir> --signals <sigs.npy> --labels <labels.npy>\n"
        "       [--n <num_samples>] [--out <results.json>] [--timing]\n"
        "\n"
        "  --weights : transformer_export/ directory (with modules/, weights/,\n"
        "              cpu_params/, config.json, scales.json)\n"
        "  --signals : eval signals.npy  (float32 shape (N, 1024, 2))\n"
        "  --labels  : eval labels.npy   (int32 shape (N,))\n"
        "  --n       : number of samples to run (default 1000; -1 = all)\n"
        "  --out     : optional JSON results file\n"
        "  --timing  : print per-stage CPU/VTA breakdown after the run\n"
        "\n"
        "Convert the .npz once on the host:\n"
        "    python3 -c \"import numpy as np; d = np.load('radioml2018_eval_snr_filtered.npz'); \\\n"
        "                 np.save('signals.npy', d['signals'].astype(np.float32)); \\\n"
        "                 np.save('labels.npy',  d['labels'].astype(np.int32))\"\n"
        "\n"
        "Bitstream must be loaded before invocation, e.g.\n"
        "    python3 -c \"from pynq import Overlay; Overlay('/home/xilinx/vta.bit')\"\n",
        prog);
}

/* Load eval signals + labels from two separate .npy files.
 * Convert from the .npz on the host with:
 *   python3 -c "import numpy as np; d = np.load('radioml2018_eval_snr_filtered.npz'); \
 *               np.save('signals.npy', d['signals'].astype(np.float32)); \
 *               np.save('labels.npy',  d['labels'].astype(np.int32))"
 *
 * Expects signals .npy float32 shape (N, 1024, 2) and labels .npy int32 shape (N,).
 * Allocates *out_signals and *out_labels via malloc; caller frees. */
static int load_eval_data(const char *signals_path, const char *labels_path,
                          float **out_signals, int32_t **out_labels, int *n_total) {
    NpyArray sigs, labs;
    if (npy_load_typed(signals_path, &sigs, "<f4") != 0) return -1;
    if (sigs.ndim != 3 || sigs.shape[1] != 1024 || sigs.shape[2] != 2) {
        fprintf(stderr, "load_eval_data: %s shape %lld,%lld,%lld != (N, 1024, 2)\n",
                signals_path, (long long)sigs.shape[0], (long long)sigs.shape[1],
                (long long)sigs.shape[2]);
        npy_free(&sigs); return -1;
    }
    if (npy_load_typed(labels_path, &labs, "<i4") != 0) {
        npy_free(&sigs); return -1;
    }
    if (labs.ndim != 1 || labs.shape[0] != sigs.shape[0]) {
        fprintf(stderr, "load_eval_data: labels shape mismatch\n");
        npy_free(&sigs); npy_free(&labs); return -1;
    }
    *n_total     = (int)sigs.shape[0];
    *out_signals = (float *)sigs.data;     /* transfer ownership */
    *out_labels  = (int32_t *)labs.data;
    sigs.data = NULL; labs.data = NULL;
    return 0;
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

int main(int argc, char **argv) {
    const char *weights_dir = NULL;
    const char *signals_path = NULL;
    const char *labels_path  = NULL;
    int         num_samples = 1000;
    const char *out_json    = NULL;
    int         do_timing   = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--weights") && i + 1 < argc) weights_dir  = argv[++i];
        else if (!strcmp(argv[i], "--signals")&& i + 1 < argc) signals_path = argv[++i];
        else if (!strcmp(argv[i], "--labels") && i + 1 < argc) labels_path  = argv[++i];
        else if (!strcmp(argv[i], "--n")      && i + 1 < argc) num_samples  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i + 1 < argc) out_json     = argv[++i];
        else if (!strcmp(argv[i], "--timing"))                   do_timing  = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }
    if (!weights_dir || !signals_path || !labels_path) {
        usage(argv[0]); return 1;
    }

    /* 1) Load 6 modules + DMA buffers + weights/biases/CPU params. */
    fprintf(stderr, "[init] weights=%s\n", weights_dir);
    if (init_runtime(weights_dir) != 0) {
        fprintf(stderr, "init_runtime failed\n");
        return 1;
    }

    /* 2) Load eval data. */
    fprintf(stderr, "[data] signals=%s labels=%s\n", signals_path, labels_path);
    float   *signals = NULL;
    int32_t *labels  = NULL;
    int      n_total = 0;
    if (load_eval_data(signals_path, labels_path, &signals, &labels, &n_total) != 0) {
        fprintf(stderr, "load_eval_data failed\n");
        cleanup_runtime();
        return 1;
    }
    if (num_samples < 0 || num_samples > n_total) num_samples = n_total;
    fprintf(stderr, "[run]  %d / %d samples\n", num_samples, n_total);

    /* 3) Hot loop. */
    int64_t t0 = now_ns();
    int     correct = 0;
    for (int i = 0; i < num_samples; i++) {
        const float *sig = signals + (size_t)i * 1024 * 2;
        int pred = infer_one(sig);
        if (pred == labels[i]) correct++;
    }
    int64_t t1 = now_ns();

    double  elapsed_s   = (double)(t1 - t0) / 1e9;
    double  fps         = num_samples / (elapsed_s > 0 ? elapsed_s : 1e-9);
    double  latency_ms  = elapsed_s * 1e3 / num_samples;
    double  accuracy    = 100.0 * correct / num_samples;
    fprintf(stderr,
        "[result] samples=%d correct=%d acc=%.2f%% fps=%.1f latency=%.3f ms\n",
        num_samples, correct, accuracy, fps, latency_ms);
    /* Also to stdout for easy capture. */
    printf("samples=%d correct=%d accuracy=%.2f fps=%.3f latency_ms=%.4f\n",
           num_samples, correct, accuracy, fps, latency_ms);

    if (out_json) {
        FILE *f = fopen(out_json, "w");
        if (f) {
            fprintf(f, "{\"samples\": %d, \"correct\": %d, \"accuracy\": %.4f, "
                       "\"fps\": %.3f, \"latency_ms\": %.4f}\n",
                    num_samples, correct, accuracy, fps, latency_ms);
            fclose(f);
        }
    }

    if (do_timing) {
        print_timings(num_samples);
    }

    free(signals);
    free(labels);
    cleanup_runtime();
    return 0;
}
