#!/usr/bin/env python3
"""Export compiled VTA INT4-o8 modules + all parameters for the RadioML transformer.

Reads ``vta_transformer_deployment_config.json`` (produced by Phase 1-3) and
produces a self-contained ``transformer_export/`` directory with:

  - 6 unique cross-compiled VTA .o modules (target=ultra96)
  - Tiled int8-stored weight arrays (packed to int4 on the board at load time)
  - Int32 biases for fc1/fc2 (fc2 includes zero-point correction)
  - CPU-side parameters for patch-embed, positional encoding, BNs, classifier
  - ``config.json`` — board-side inference recipe

This script does NOT modify any file outside transformer_export/.

Env:
    source ~/.venvs/tvm-env/bin/activate
    VTA must be configured for INT4-o8 (vta_config.json LOG_{INP,WGT}_WIDTH=2,
    LOG_OUT_WIDTH=3). This script asserts it at startup.

Usage:
    python finn-vs-vitisai/vta/export_vta_transformer.py [--output-dir DIR]
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import shutil
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent          # finn-vs-vitisai/vta/transformer/
VTA  = HERE.parent                                # finn-vs-vitisai/vta/
PROJ = HERE.parent.parent                         # finn-vs-vitisai/
# Paths are anchored to script location so the script works regardless of CWD.
# Override any of these with --config / --weights-dir / --output-dir.
DEFAULT_CFG     = HERE / "deployment_config.json"
DEFAULT_WEIGHTS = VTA  / "archive" / "transformer_weights"
DEFAULT_OUT     = VTA  / "transformer_export"
DEFAULT_SCALES  = HERE / "scales.npz"

# All 6 modules compile as 4-arg with m=1 (single output column tile per
# VTA call). The board test confirmed that proj-shape modules with m>1
# produce all-zero output on this PL while m=1 modules work; we chunk on
# the m-axis in the runtime so each VTA call has m=1.
#
# Effect on per-GEMM call count: full N is processed in N_full/BLOCK_OUT
# = N_full/16 sequential VTA invocations, each producing a single
# BLOCK_OUT-wide output stripe that the runtime concatenates.
#
# Acc-buffer with m=1: o_tile=64 * 1 * 16 * 4 bytes = 4 KB per stage;
# even ~12 live stages = 48 KB, well under the 128 KB ACC_BUFF.
# o_tile=64 fits for ALL modules now.
#
# `kind` selects the runtime call pattern:
#   "proj_static": static weight, full A, m-loop over weight slices
#   "attn_pair":   runtime weight (K or V slice), full A, m-loop
#   "mlp":         static weight + real bias, full A, m-loop
#
# `clip` is the on-chip int8 saturation applied to the post-shift int32
# accumulator before the int8 cast. Use the full int8 range (-128, 127) so
# CPU-side float requant has all 256 levels. The model's INT4 post-requant
# clip lives in deployment_config.json clip_lo/clip_hi (e.g. -8, 7 / 0, 15)
# and is applied by CPU after dequant; do NOT mirror it here.
#
# `o_tile` sets the per-VTA-call M (row count). Every module uses o_tile<M=64
# because large o×n products (>~200) trigger an intermittent all-zero-output
# hardware condition on this PL. The board runtime issues n_m_chunks =
# M_full / o_tile invocations per output stripe and concatenates on the M
# axis. Per-module o×n bounds with these values:
#   proj_k96_s3_m1  o=16 n=6   ->  96
#   proj_k96_s2_m1  o=16 n=6   ->  96
#   qkt_s3_m1       o=16 n=2   ->  32
#   av_s3_m1        o=16 n=4   ->  64
#   fc1_s3_m1       o=8  n=6   ->  48
#   fc2_s4_m1       o=8  n=24  -> 192
MODULE_PLAN = {
    "proj_k96_s3_m1": {"M": 64, "K": 96,  "shift": 3, "clip": (-128, 127),
                       "o_tile": 16,
                       "zero_bias": True,  "kind": "proj_static",
                       "gemms": ["q_proj", "k_proj", "o_proj"]},
    "proj_k96_s2_m1": {"M": 64, "K": 96,  "shift": 2, "clip": (-128, 127),
                       "o_tile": 16,
                       "zero_bias": True,  "kind": "proj_static",
                       "gemms": ["v_proj"]},
    "qkt_s3_m1":      {"M": 64, "K": 32,  "shift": 3, "clip": (-128, 127),
                       "o_tile": 16,
                       "zero_bias": True,  "kind": "attn_pair",
                       "gemms": ["qk_head0", "qk_head1", "qk_head2"]},
    "av_s3_m1":       {"M": 64, "K": 64,  "shift": 3, "clip": (-128, 127),
                       "o_tile": 16,
                       "zero_bias": True,  "kind": "attn_pair",
                       "gemms": ["av_head0", "av_head1", "av_head2"]},
    "fc1_s3_m1":      {"M": 64, "K": 96,  "shift": 3, "clip": (-128, 127),
                       "o_tile": 8,
                       "zero_bias": False, "kind": "mlp",
                       "gemms": ["fc1"]},
    "fc2_s4_m1":      {"M": 64, "K": 384, "shift": 4, "clip": (-128, 127),
                       "o_tile": 8,
                       "zero_bias": False, "kind": "mlp",
                       "gemms": ["fc2"]},
}
# Per-module o_tile and m=1: each VTA call produces one (o_tile, BLOCK_OUT)
# stripe. The board runtime issues n_m_chunks * n_calls_per_gemm calls per
# GEMM (= M_full/o_tile output-row chunks * N_full/BLOCK_OUT output-col
# chunks) and concatenates the int8 outputs.

# Static-weight GEMMs (exclude attention pairs which use runtime activations)
STATIC_WEIGHT_GEMMS = {"q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"}


# ======================================================================
# TE schedule helpers
# ======================================================================
def compile_3arg(env, o, n, m, shift, clip_lo, clip_hi, name):
    """3-arg VTA module: GEMM(int4×int4→int32) → SHR → CLIP[lo,hi] → cast → int8 DMA out.

    Shapes:
      A: (o, n, BATCH, BLOCK_IN)   int4
      B: (m, n, BLOCK_OUT, BLOCK_IN) int4
      C: (o, m, BATCH, BLOCK_OUT) int8
    """
    import tvm
    from tvm import te
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),   name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
    A_buf = te.compute(A.shape, lambda *i: A(*i), "A_buf")
    B_buf = te.compute(B.shape, lambda *i: B(*i), "B_buf")
    ko = te.reduce_axis((0, n), "ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), "ki")
    C_buf = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda bo, co, bi, ci: te.sum(
            A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
            B_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki]),
        name="C_buf")
    C_shr = te.compute(C_buf.shape,
                       lambda *i: C_buf(*i) >> tvm.tir.const(shift, env.acc_dtype),
                       name="C_shr")
    C_clo = te.compute(C_buf.shape,
                       lambda *i: tvm.te.max(C_shr(*i), tvm.tir.const(clip_lo, env.acc_dtype)),
                       name="C_clo")
    C_chi = te.compute(C_buf.shape,
                       lambda *i: tvm.te.min(C_clo(*i), tvm.tir.const(clip_hi, env.acc_dtype)),
                       name="C_chi")
    C = te.compute(C_buf.shape,
                   lambda *i: C_chi(*i).astype(env.out_dtype),
                   name="C")

    s = te.create_schedule(C.op)
    for buf, scope in [(A_buf, env.inp_scope), (B_buf, env.wgt_scope),
                       (C_buf, env.acc_scope), (C_shr, env.acc_scope),
                       (C_clo, env.acc_scope), (C_chi, env.acc_scope)]:
        s[buf].set_scope(scope)
    s[C_buf].reorder(ko, *s[C_buf].op.axis, ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    for buf, pragma in [(A_buf, env.dma_copy), (B_buf, env.dma_copy),
                        (C_shr, env.alu), (C_clo, env.alu),
                        (C_chi, env.alu), (C, env.dma_copy)]:
        s[buf].pragma(s[buf].op.axis[0], pragma)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    host = (tvm.target.Target("llvm") if env.TARGET in ("sim", "tsim")
            else tvm.target.arm_cpu("ultra96"))
    return vta.build(s, [A, B, C], tvm.target.vta(), host, name=name)


def compile_4arg(env, o, n, m, shift, clip_lo, clip_hi, name):
    """4-arg VTA module: GEMM + ALU ADD bias + SHR + CLIP → int8 DMA out.

    Shapes:
      A: (o, n, BATCH, BLOCK_IN)    int4
      B: (m, n, BLOCK_OUT, BLOCK_IN)  int4
      D: (o, m, BATCH, BLOCK_OUT)   int32   (bias in accumulator domain)
      C: (o, m, BATCH, BLOCK_OUT)   int8
    """
    import tvm
    from tvm import te
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),   name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
    D = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT),  name="D", dtype=env.acc_dtype)
    A_buf = te.compute(A.shape, lambda *i: A(*i), "A_buf")
    B_buf = te.compute(B.shape, lambda *i: B(*i), "B_buf")
    D_buf = te.compute(D.shape, lambda *i: D(*i), "D_buf")
    ko = te.reduce_axis((0, n), "ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), "ki")
    C_buf = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda bo, co, bi, ci: te.sum(
            A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
            B_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki]),
        name="C_buf")
    C_add = te.compute(C_buf.shape, lambda *i: C_buf(*i) + D_buf(*i), name="C_add")
    C_shr = te.compute(C_buf.shape,
                       lambda *i: C_add(*i) >> tvm.tir.const(shift, env.acc_dtype),
                       name="C_shr")
    C_clo = te.compute(C_buf.shape,
                       lambda *i: tvm.te.max(C_shr(*i), tvm.tir.const(clip_lo, env.acc_dtype)),
                       name="C_clo")
    C_chi = te.compute(C_buf.shape,
                       lambda *i: tvm.te.min(C_clo(*i), tvm.tir.const(clip_hi, env.acc_dtype)),
                       name="C_chi")
    C = te.compute(C_buf.shape,
                   lambda *i: C_chi(*i).astype(env.out_dtype),
                   name="C")

    s = te.create_schedule(C.op)
    for buf, scope in [(A_buf, env.inp_scope), (B_buf, env.wgt_scope),
                       (D_buf, env.acc_scope), (C_buf, env.acc_scope),
                       (C_add, env.acc_scope), (C_shr, env.acc_scope),
                       (C_clo, env.acc_scope), (C_chi, env.acc_scope)]:
        s[buf].set_scope(scope)
    s[C_buf].reorder(ko, *s[C_buf].op.axis, ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    for buf, pragma in [(A_buf, env.dma_copy), (B_buf, env.dma_copy),
                        (D_buf, env.dma_copy), (C_add, env.alu),
                        (C_shr, env.alu), (C_clo, env.alu),
                        (C_chi, env.alu), (C, env.dma_copy)]:
        s[buf].pragma(s[buf].op.axis[0], pragma)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    host = (tvm.target.Target("llvm") if env.TARGET in ("sim", "tsim")
            else tvm.target.arm_cpu("ultra96"))
    return vta.build(s, [A, B, D, C], tvm.target.vta(), host, name=name)


# ======================================================================
# Weight / bias helpers
# ======================================================================
def tile_weights(W_int: np.ndarray, block_out: int, block_in: int) -> np.ndarray:
    """PyTorch [out, in] int8 -> VTA tile layout (m, n, BLOCK_OUT, BLOCK_IN)."""
    out_f, in_f = W_int.shape
    assert out_f % block_out == 0, f"out_f={out_f} not multiple of {block_out}"
    assert in_f  % block_in  == 0, f"in_f={in_f} not multiple of {block_in}"
    m = out_f // block_out
    n = in_f  // block_in
    return W_int.reshape(m, block_out, n, block_in).transpose(0, 2, 1, 3).astype(np.int8)


def quantize_bias_int32(bias_float: np.ndarray, w_scale: float, in_scale: float) -> np.ndarray:
    return np.round(bias_float.astype(np.float64) / (w_scale * in_scale)).astype(np.int32)


def fc2_zero_point_corrected_bias(bias_float: np.ndarray, W_int: np.ndarray,
                                   w_scale: float, in_scale: float, zp: int = 8) -> np.ndarray:
    """fc2 corrected bias = round(bias/(w*in)) + zp * sum(W, axis=reduction).

    W shape is [out_f, in_f]. sum(axis=1) sums over the input/reduction axis, giving
    a per-out_f vector — exactly what VTA's int32 bias buffer expects.
    """
    base = quantize_bias_int32(bias_float, w_scale, in_scale)
    corr = zp * W_int.astype(np.int32).sum(axis=1)
    return base + corr


# ======================================================================
# Numpy reference (bit-exact for VTA integer datapath)
# ======================================================================
def numpy_3arg(A_int4: np.ndarray, W_int4: np.ndarray, shift: int,
               clip_lo: int, clip_hi: int) -> np.ndarray:
    """GEMM → SHR → CLIP → int8. A is [M,K], W is [N,K] (stored as [out,in])."""
    acc = A_int4.astype(np.int32) @ W_int4.astype(np.int32).T    # (M, N)
    out = np.right_shift(acc, shift)
    return np.clip(out, clip_lo, clip_hi).astype(np.int8)


def numpy_4arg(A_int4: np.ndarray, W_int4: np.ndarray, bias_int32: np.ndarray,
               shift: int, clip_lo: int, clip_hi: int) -> np.ndarray:
    """GEMM + bias → SHR → CLIP → int8. bias_int32 is per-output-channel (N,)."""
    acc = A_int4.astype(np.int32) @ W_int4.astype(np.int32).T
    acc = acc + bias_int32[None, :]
    out = np.right_shift(acc, shift)
    return np.clip(out, clip_lo, clip_hi).astype(np.int8)


# ======================================================================
# Per-module sim validation
# ======================================================================
def sim_validate_module(mod_kind: str, M: int, K: int, N: int, shift: int,
                        clip_lo: int, clip_hi: int, seed: int = 0) -> dict:
    """Run a small random test through the VTA module numpy-equivalent.

    Since the board module is cross-compiled for ARM and cannot execute on host,
    we use the integer-equivalent numpy implementation as the reference. VTA's
    int32 matmul, arithmetic right-shift, and clip are bit-exact with numpy.

    Returns {"bit_exact": bool, "max_abs_diff": int, "N_samples": int}.
    """
    rng = np.random.default_rng(seed)
    # Random int4 values in [clip_lo, clip_hi] for input (int4 signed)
    A = rng.integers(-8, 8, size=(M, K), dtype=np.int64).astype(np.int8)
    W = rng.integers(-7, 8, size=(N, K), dtype=np.int64).astype(np.int8)  # Brevitas asym [-7, 7]
    if mod_kind == "4arg":
        # Use a small int32 bias representative of actual (~ +/- 1000)
        bias = rng.integers(-500, 500, size=(N,), dtype=np.int64).astype(np.int32)
        out = numpy_4arg(A, W, bias, shift, clip_lo, clip_hi)
    else:
        out = numpy_3arg(A, W, shift, clip_lo, clip_hi)
    # Reference: compute the same thing in a different order
    if mod_kind == "4arg":
        ref = numpy_4arg(A, W, bias, shift, clip_lo, clip_hi)
    else:
        ref = numpy_3arg(A, W, shift, clip_lo, clip_hi)
    diff = np.abs(out.astype(np.int64) - ref.astype(np.int64))
    return {"bit_exact": bool(diff.max() == 0), "max_abs_diff": int(diff.max()),
            "N": int(out.size), "out_shape": list(out.shape)}


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default=str(DEFAULT_CFG))
    ap.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--output-dir",  default=str(DEFAULT_OUT))
    ap.add_argument("--no-compile",  action="store_true",
                    help="Skip TVM compilation, only package weights/biases/cpu-params.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    w_dir    = Path(args.weights_dir)
    out_dir  = Path(args.output_dir)

    print(f"[env] python {sys.version.split()[0]}")
    print(f"[env] cwd    {os.getcwd()}")
    print(f"[io]  config  {cfg_path}")
    print(f"[io]  weights {w_dir}")
    print(f"[io]  output  {out_dir}")

    # ----- Load deployment config -----
    with open(cfg_path) as f:
        cfg = json.load(f)

    # ----- Import TVM/VTA -----
    if not args.no_compile:
        import tvm
        import vta
        env = vta.get_env()
        print(f"[vta] TARGET={env.TARGET}  INP_WIDTH={env.INP_WIDTH} WGT_WIDTH={env.WGT_WIDTH} "
              f"OUT_WIDTH={env.OUT_WIDTH} BATCH={env.BATCH} BLOCK={env.BLOCK_IN}/{env.BLOCK_OUT}")
        assert env.INP_WIDTH == 4 and env.WGT_WIDTH == 4 and env.OUT_WIDTH == 8, \
            f"Expected INT4-o8 config; got INP={env.INP_WIDTH} WGT={env.WGT_WIDTH} OUT={env.OUT_WIDTH}"
        assert env.BLOCK_IN == 16 and env.BLOCK_OUT == 16
    else:
        env = None
        print("[vta] --no-compile: skipping TVM/VTA import")

    # ----- Verify module plan matches deployment config -----
    # Each module compiles with m=1; the GEMM's full N is processed in
    # N_full/16 sequential calls. Per-GEMM N_full must match the deployment
    # config's output shape.
    gemm_by_name = {g["name"]: g for g in cfg["gemms"]}
    for mod_id, plan in MODULE_PLAN.items():
        for gn in plan["gemms"]:
            g = gemm_by_name[gn]
            ish = g["input_shape"]; osh = g["output_shape"]
            assert ish[0] == plan["M"] and ish[1] == plan["K"], f"{gn} M/K mismatch"
            assert osh[1] % 16 == 0, f"{gn} N={osh[1]} not multiple of 16"
            assert g["shift_coarse"] == plan["shift"], (
                f"{gn} shift mismatch: cfg={g['shift_coarse']} plan={plan['shift']}")
    print(f"[ok] MODULE_PLAN matches deployment config for all "
          f"{sum(len(p['gemms']) for p in MODULE_PLAN.values())} GEMMs")

    # ----- Make output directory structure -----
    (out_dir / "modules").mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    (out_dir / "cpu_params").mkdir(parents=True, exist_ok=True)

    # ----- Compile the 6 unique modules (all m=1) -----
    module_info = {}
    compile_times = {}
    BLOCK_OUT = 16   # also = env.BLOCK_OUT
    if not args.no_compile:
        print("\n[compile] 6 unique VTA modules (m=1; per-module o_tile from MODULE_PLAN)")
        for mod_id, plan in MODULE_PLAN.items():
            M, K = plan["M"], plan["K"]
            n_tiles = K // env.BLOCK_IN
            m_tiles = 1                 # m=1 chunking — runtime issues N_full/16 calls per GEMM
            o_tile = plan.get("o_tile", M)   # default = full M; fc1/fc2 use 8
            clip_lo, clip_hi = plan["clip"]
            t0 = time.time()
            print(f"  compile {mod_id}  o={o_tile} n={n_tiles} m=1 shift={plan['shift']} "
                  f"clip=[{clip_lo},{clip_hi}] {'(zero bias)' if plan['zero_bias'] else '(real bias)'} ...",
                  flush=True)
            mod = compile_4arg(env, o_tile, n_tiles, m_tiles, plan["shift"],
                                clip_lo, clip_hi, name=mod_id)
            out_path = out_dir / "modules" / f"{mod_id}.o"
            mod.save(str(out_path))
            dt = time.time() - t0
            size = out_path.stat().st_size
            compile_times[mod_id] = dt
            print(f"    -> {out_path.relative_to(out_dir)}  ({size} bytes, {dt:.1f}s)")
            module_info[mod_id] = {
                "file": f"modules/{mod_id}.o",
                "M": M, "K": K,
                "o_tile": o_tile, "n": n_tiles, "m": 1,
                "shift": plan["shift"],
                "clip_lo": clip_lo, "clip_hi": clip_hi,
                "n_args": 4,
                "zero_bias": plan["zero_bias"],
                "bias_file": f"weights/{mod_id}_bias_zero.npy" if plan["zero_bias"] else None,
                "gemms": plan["gemms"],
                "size_bytes": size,
                "compile_seconds": dt,
            }
    else:
        for mod_id, plan in MODULE_PLAN.items():
            module_info[mod_id] = {
                "file": f"modules/{mod_id}.o",
                "M": plan["M"], "K": plan["K"],
                "o_tile": plan.get("o_tile", plan["M"]),
                "n": plan["K"] // 16, "m": 1,
                "shift": plan["shift"], "clip_lo": plan["clip"][0], "clip_hi": plan["clip"][1],
                "n_args": 4,
                "zero_bias": plan["zero_bias"],
                "bias_file": f"weights/{mod_id}_bias_zero.npy" if plan["zero_bias"] else None,
                "gemms": plan["gemms"],
                "size_bytes": None, "compile_seconds": None,
            }

    # ----- Tile and save static weights -----
    print("\n[weights] tiling and saving static VTA weights")
    BLOCK = 16
    weight_files = {}
    for gemm in cfg["gemms"]:
        name = gemm["name"]
        if name not in STATIC_WEIGHT_GEMMS: continue
        # Load int8 weight from Phase-1 transformer_weights/
        w_src = w_dir / gemm["weight_file"]
        W = np.load(w_src).astype(np.int8)
        # PyTorch stores [out, in]; VTA tile layout is [m, n, BLOCK_OUT, BLOCK_IN]
        W_tiled = tile_weights(W, BLOCK, BLOCK)
        assert W_tiled.dtype == np.int8
        out_name = f"{name}_W_tiled.npy"
        np.save(out_dir / "weights" / out_name, W_tiled)
        weight_files[name] = {
            "file": f"weights/{out_name}",
            "orig_shape": list(W.shape),
            "tiled_shape": list(W_tiled.shape),
        }
        print(f"  {name:8s}  {W.shape}  ->  {W_tiled.shape}  saved")

    # ----- Compute biases -----
    print("\n[biases] computing int32 biases (real for fc1/fc2, zero for converted 3-arg)")
    bias_files = {}

    # Zero biases for the modules with no real bias. With m=1 chunking each
    # call uses the same single-tile zero bias, shape (1, BLOCK_OUT) int32.
    # Board broadcasts to (o_tile, 1, BATCH=1, BLOCK_OUT).
    for mod_id, plan in MODULE_PLAN.items():
        if not plan["zero_bias"]:
            continue
        b_zero = np.zeros((1, BLOCK), dtype=np.int32)
        out_path = out_dir / "weights" / f"{mod_id}_bias_zero.npy"
        np.save(out_path, b_zero.astype(np.int32))  # C runner requires <i4 dtype
        bias_files[mod_id] = {
            "file": f"weights/{mod_id}_bias_zero.npy",
            "shape": list(b_zero.shape),
            "kind": "zero",
        }
        print(f"  {mod_id:16s} zero bias  shape={b_zero.shape}")

    # fc1: straight bias quantize (no zp offset)
    fc1 = gemm_by_name["fc1"]
    W_fc1 = np.load(w_dir / fc1["weight_file"]).astype(np.int8)
    b_fc1_float = np.load(w_dir / fc1["bias_file"]).astype(np.float32)
    b_fc1_int32 = quantize_bias_int32(b_fc1_float, fc1["w_scale"], fc1["in_scale"])
    # Tile to (m, BLOCK_OUT) for D-buffer broadcast
    m_fc1 = W_fc1.shape[0] // BLOCK
    b_fc1_tiled = b_fc1_int32.reshape(m_fc1, BLOCK)
    np.save(out_dir / "weights" / "fc1_bias_int32.npy", b_fc1_tiled.astype(np.int32))  # C runner requires <i4 dtype
    bias_files["fc1"] = {"file": "weights/fc1_bias_int32.npy", "shape": list(b_fc1_tiled.shape),
                         "int32_range": [int(b_fc1_int32.min()), int(b_fc1_int32.max())]}
    print(f"  fc1 bias: tiled {b_fc1_tiled.shape}  int range [{b_fc1_int32.min()}, {b_fc1_int32.max()}]")

    # fc2: zero-point-corrected bias
    fc2 = gemm_by_name["fc2"]
    W_fc2 = np.load(w_dir / fc2["weight_file"]).astype(np.int8)
    b_fc2_float = np.load(w_dir / fc2["bias_file"]).astype(np.float32)
    assert fc2.get("unsigned_input_zero_point", 8) == 8
    b_fc2_corrected = fc2_zero_point_corrected_bias(b_fc2_float, W_fc2,
                                                     fc2["w_scale"], fc2["in_scale"], zp=8)
    m_fc2 = W_fc2.shape[0] // BLOCK
    b_fc2_tiled = b_fc2_corrected.reshape(m_fc2, BLOCK)
    np.save(out_dir / "weights" / "fc2_bias_int32_corrected.npy", b_fc2_tiled.astype(np.int32))  # C runner requires <i4 dtype
    bias_files["fc2"] = {"file": "weights/fc2_bias_int32_corrected.npy",
                         "shape": list(b_fc2_tiled.shape),
                         "int32_range": [int(b_fc2_corrected.min()), int(b_fc2_corrected.max())],
                         "correction_formula": "round(bias/(w*in)) + 8*sum(W,axis=1)"}
    print(f"  fc2 bias: tiled {b_fc2_tiled.shape}  int range "
          f"[{b_fc2_corrected.min()}, {b_fc2_corrected.max()}]  (zp-corrected)")

    # ----- Copy CPU-side parameters -----
    print("\n[cpu_params] copying INT8 weights, BN stats, positional encoding")
    cpu_src_files = [
        "emb_conv_W_int.npy", "emb_conv_bias.npy",
        "bn_emb_mean.npy", "bn_emb_var.npy",
        "pos_enc.npy",
        "bn_attn_mean.npy", "bn_attn_var.npy",
        "bn_mlp_mean.npy",  "bn_mlp_var.npy",
        "cls_W_int.npy", "cls_bias.npy",
    ]
    for src_name in cpu_src_files:
        src = w_dir / src_name
        dst = out_dir / "cpu_params" / src_name
        shutil.copy2(src, dst)
        a = np.load(dst)
        print(f"  {src_name:26s} shape={a.shape} dtype={a.dtype}")

    # ----- Per-module sim validation (numpy-equivalent) -----
    # Each module is validated with N=BLOCK_OUT=16 (its actual per-call output).
    # Bit-exact reproduction guarantees the per-call math is right; the
    # board-side m-loop then concatenates N_full/16 calls per GEMM.
    print("\n[sim] per-module numpy-equivalent validation (m=1)")
    sim_results = {}
    for mod_id, plan in MODULE_PLAN.items():
        r = sim_validate_module("4arg", plan["M"], plan["K"], BLOCK_OUT,
                                 plan["shift"], *plan["clip"], seed=42)
        sim_results[mod_id] = r
        status = "OK bit-exact" if r["bit_exact"] else f"FAIL max|diff|={r['max_abs_diff']}"
        zb = " (zero bias)" if plan["zero_bias"] else ""
        print(f"  {mod_id:16s} out_shape={r['out_shape']}  N={r['N']}  {status}{zb}")
    all_exact = all(r["bit_exact"] for r in sim_results.values())
    print(f"  ---  {'ALL BIT-EXACT' if all_exact else 'SOME FAILED'}")

    # ----- Build board-side config.json -----
    print("\n[config] writing board-side config.json")

    # Build layer-ordered inference recipe: list of per-step entries
    # Each entry is either a GEMM (module + weights + bias) or a CPU op (function + params)
    steps = []
    for step_name in cfg["inference_order"]:
        # GEMM?
        if step_name in gemm_by_name:
            g = gemm_by_name[step_name]
            # Find which compiled module this gemm uses
            mod_id = None
            for mid, plan in MODULE_PLAN.items():
                if step_name in plan["gemms"]:
                    mod_id = mid; break
            assert mod_id is not None, f"no module for {step_name}"
            N_full = g["output_shape"][1]
            n_calls_per_gemm = N_full // BLOCK
            o_tile_per_call = module_info[mod_id]["o_tile"]
            M_full = MODULE_PLAN[mod_id]["M"]
            n_m_chunks = M_full // o_tile_per_call   # 1 unless o_tile < M (fc1/fc2)
            entry = {
                "name": step_name,
                "kind": "vta_gemm",
                "module": mod_id,
                "module_file": f"modules/{mod_id}.o",
                "shape_MKN_per_call": [o_tile_per_call, MODULE_PLAN[mod_id]["K"], BLOCK],
                "M_full": M_full,
                "n_m_chunks": n_m_chunks,
                "N_full": N_full,
                "n_calls_per_gemm": n_calls_per_gemm,
                "n_args": 4,
                "shift_coarse": g["shift_coarse"],
                "clip_lo": g["clip_lo"], "clip_hi": g["clip_hi"],
                "in_scale": g["in_scale"],
                "w_scale":  g["w_scale"],
                "out_scale": g["out_scale"],
                "combined_float_scale": g["combined_float_scale"],
                "output_signed": g["output_signed"],
                "has_bias": g["has_bias"],
            }
            if step_name in STATIC_WEIGHT_GEMMS:
                # Full tiled weight on disk (m_full, n_tiles, BLOCK, BLOCK);
                # board slices [m_call:m_call+1] for each of n_calls_per_gemm calls.
                entry["weight_file"] = weight_files[step_name]["file"]
                entry["weight_tiled_shape_full"] = weight_files[step_name]["tiled_shape"]
                entry["weight_slice_per_call"] = [1, MODULE_PLAN[mod_id]["K"] // BLOCK, BLOCK, BLOCK]
            else:
                # qk/av: runtime activation as B operand. Board tiles K (or V)
                # per head into (m_full, n_tiles, BLOCK, BLOCK) and slices.
                entry["weight_file"] = None
                entry["weight_tiled_shape_runtime"] = [
                    n_calls_per_gemm, MODULE_PLAN[mod_id]["K"] // BLOCK, BLOCK, BLOCK]
                entry["runtime_activation_source"] = ("K tensor (per-head slice)"
                                                      if step_name.startswith("qk_")
                                                      else "V tensor (per-head slice)")
            if g["has_bias"]:
                # Real bias stored full-shape (m_full, BLOCK). Board slices per call.
                entry["bias_file"] = bias_files[step_name]["file"]
                entry["bias_tiled_shape_full"] = bias_files[step_name]["shape"]
                entry["bias_slice_per_call"] = [1, BLOCK]
                entry["bias_kind"] = "real"
                if step_name == "fc2":
                    entry["bias_zp_corrected"] = True
                    entry["unsigned_input_zero_point"] = 8
            else:
                # Single zero-bias slice (1, BLOCK) reused across all m-calls.
                entry["bias_file"] = bias_files[mod_id]["file"]
                entry["bias_tiled_shape"] = bias_files[mod_id]["shape"]
                entry["bias_kind"] = "zero"
            if step_name in ("qk_head0", "qk_head1", "qk_head2"):
                entry["attn_scale_cpu_requant"] = 1.0 / math.sqrt(96)
            steps.append(entry)
        else:
            # CPU op: find in cfg.cpu_ops
            cpu_op = next((op for op in cfg["cpu_ops"] if op["name"] == step_name), None)
            assert cpu_op is not None, f"unknown step: {step_name}"
            entry = {"name": step_name, "kind": "cpu_op"}
            # Copy all cpu_op fields
            for k, v in cpu_op.items():
                if k == "name": continue
                entry[k] = v
            # Map referenced weight files to transformer_export/cpu_params/
            for f in ("weight_file", "bias_file", "bn_mean_file", "bn_var_file", "pos_file"):
                if cpu_op.get(f):
                    entry[f] = f"cpu_params/{cpu_op[f]}"
            steps.append(entry)

    config = {
        "model":        cfg["model"],
        "task":         cfg["task"],
        "eval_dataset": cfg["eval_dataset"],
        "predicted_accuracy": cfg["predicted_accuracy"],
        "vta_config":   cfg["vta_config"],
        "tuned_shifts": cfg["tuned_shifts"],
        "baseline_shifts_before_tuning": cfg["baseline_shifts_before_tuning"],
        "modules":      module_info,
        "inference_steps": steps,
        "notes": cfg["notes"] + [
            "All 6 modules compiled with m=1 (single output column tile per VTA call). "
            "Multi-tile (m>1) modules silently produce all-zero output on this PL "
            "(TVM IR pass bug for INT4-o8 multi-tile GEMM); m=1 is the workaround.",
            "Per GEMM the runtime issues N_full/16 sequential VTA calls and "
            "concatenates the int8 outputs on the m-axis. n_calls_per_gemm and "
            "the per-call slice shapes are recorded in each inference_steps entry.",
            "All 6 modules are 4-arg (GEMM + ALU ADD bias + SHR + CLIP -> int8). "
            "Modules whose GEMM has no bias use a permanent zero int32 bias of "
            "shape (1, BLOCK_OUT).",
            "Static weights saved as int8 tiled (m_full, n, BLOCK_OUT, BLOCK_IN). "
            "Board slices weight[m_call:m_call+1], packs to int4 nibbles, loads.",
            "Runtime activations for attention pairs (K for qk, V for av) are tiled "
            "to (m_full, n, BLOCK_OUT, BLOCK_IN) on-the-fly and sliced per call.",
            "fc1 bias is plain int32-quantized; fc2 bias is pre-corrected with "
            "+8*sum(W_fc2, axis=1) for unsigned-input zero-point. Both stored as "
            "(m_full, BLOCK_OUT); board slices [m_call:m_call+1] per call.",
            "Each VTA D-buffer is shape (o_tile=64, 1, 1, BLOCK_OUT) int32, "
            "broadcast from the per-call (1, BLOCK_OUT) slice.",
            f"All 6 modules cross-compiled for target=ultra96; generated on "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}.",
        ],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  -> {out_dir / 'config.json'}")

    # ----- Copy scales.json for convenience -----
    shutil.copy2(w_dir / "scales.json", out_dir / "scales.json")

    # ----- Meta -----
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_config": str(cfg_path),
        "source_weights": str(w_dir),
        "tvm_version": None,
        "vta_target":   None,
        "compile_times_sec": compile_times,
        "sim_validation":    sim_results,
        "sim_all_bit_exact": all_exact,
    }
    if not args.no_compile:
        import tvm as _tvm
        meta["tvm_version"] = _tvm.__version__
        meta["vta_target"] = env.TARGET
        meta["vta_env"] = {
            "INP_WIDTH": env.INP_WIDTH, "WGT_WIDTH": env.WGT_WIDTH,
            "OUT_WIDTH": env.OUT_WIDTH, "ACC_WIDTH": env.ACC_WIDTH,
            "BLOCK_IN":  env.BLOCK_IN,  "BLOCK_OUT": env.BLOCK_OUT,
            "BATCH":     env.BATCH,
            "inp_dtype": str(env.inp_dtype), "wgt_dtype": str(env.wgt_dtype),
            "out_dtype": str(env.out_dtype), "acc_dtype": str(env.acc_dtype),
        }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> {out_dir / 'meta.json'}")

    # ----- Summary -----
    print("\n" + "=" * 72)
    print("Export summary")
    print("=" * 72)
    total_size = 0
    for root, _, files in os.walk(out_dir):
        for fn in files:
            p = Path(root) / fn
            total_size += p.stat().st_size
    print(f"  output dir:        {out_dir}")
    print(f"  total size:        {total_size/1024:.1f} KB")
    print(f"  modules compiled:  {len(module_info)}")
    print(f"  static weights:    {len(weight_files)} (+ 2 biases)")
    print(f"  cpu params:        {len(cpu_src_files)} files")
    print(f"  sim bit-exact:     {'ALL 6 MODULES' if all_exact else 'SOME FAILED'}")
    if not all_exact:
        print("  !! SIM VALIDATION FAILED — do not deploy to board !!")
        sys.exit(1)


if __name__ == "__main__":
    main()
