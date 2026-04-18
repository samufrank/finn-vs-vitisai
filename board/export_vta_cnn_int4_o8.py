#!/usr/bin/env python3
"""Export VTA CNN INT4-input/INT8-output modules for board-side inference.

Deploys the per-channel BN-folded [8,16] model via Mode G pipeline:
  - INT4 input with zero-point offset (Brevitas [0,15] → VTA [-8,7] via zp=8)
  - INT8 DMA output (clip [-128, 127], 256 levels)
  - Per-channel weight scales (from BN-fold + per-channel quant)
  - Corrected int32 bias = fold_bias_int32 + zp * sum(W_int, input_axis)

Architecture: Conv1(1→8) → ReLU → MaxPool → Conv2(8→16) → ReLU → MaxPool →
              AvgPool → Dense(16→10). BN folded into conv weights.

Per hidden conv layer: 4-arg module (A, B, D_bias, C_out).
  VTA: GEMM → ALU ADD corrected_bias → SHR → CLIP[-128,127] → int8 out.
Dense (last): 3-arg module (A, B, C_out).
  VTA: GEMM → SHR → CLIP[-128,127] → int8 out.
  CPU: dequant + add corrected_float_bias (includes zp correction) → argmax.

Usage:
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \\
        python3 ../finn-vs-vitisai/board/export_vta_cnn_int4_o8.py
"""
import argparse
import json
import math
import os
import sys

import numpy as np

WEIGHTS_DIR = os.path.expanduser(
    '~/dev/CEN571-final/tvm-v0.12.0/vta_mnist_weights_int4_cnn_perchan')
OUTPUT_DIR = os.path.expanduser(
    '~/dev/CEN571-final/tvm-v0.12.0/vta_export/cnn_mnist_int4_o8_perchan')

ZERO_POINT = 8
INT8_CLIP_LO = -128
INT8_CLIP_HI = 127

# Tiling: Conv1 o=196×4, Conv2 o=49×4.
CONV_TILING = [
    {"o_tile": 196, "n_chunks": 4},
    {"o_tile": 49,  "n_chunks": 4},
]


def pad_to_block(arr, block, axis):
    """Pad arr along axis to a multiple of block."""
    s = arr.shape[axis]
    if s % block == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, block - s % block)
    return np.pad(arr, pad_width, mode='constant')


def compile_gemm_bias_shr_clip_int8(env, o, n, m, shift, clip_lo, clip_hi):
    """4-arg module: GEMM + ALU ADD bias + SHR + CLIP → int8 out."""
    import tvm
    from tvm import te
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
    D = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="D", dtype=env.acc_dtype)
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
            axis=[ko, ki]), name="C_buf")
    C_add = te.compute(C_buf.shape, lambda *i: C_buf(*i) + D_buf(*i), name="C_add")
    C_shr = te.compute(C_buf.shape,
        lambda *i: C_add(*i) >> tvm.tir.const(shift, env.acc_dtype), name="C_shr")
    C_clo = te.compute(C_buf.shape,
        lambda *i: tvm.te.max(C_shr(*i), tvm.tir.const(clip_lo, env.acc_dtype)), name="C_clo")
    C_chi = te.compute(C_buf.shape,
        lambda *i: tvm.te.min(C_clo(*i), tvm.tir.const(clip_hi, env.acc_dtype)), name="C_chi")
    C = te.compute(C_buf.shape,
        lambda *i: C_chi(*i).astype(env.out_dtype), name="C")

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
    return vta.build(s, [A, B, D, C], tvm.target.vta(), host, name="my_gemm")


def compile_gemm_shr_clip_int8(env, o, n, m, shift, clip_lo, clip_hi):
    """3-arg module: GEMM + SHR + CLIP → int8 out (no bias in VTA)."""
    import tvm
    from tvm import te
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
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
            axis=[ko, ki]), name="C_buf")
    C_shr = te.compute(C_buf.shape,
        lambda *i: C_buf(*i) >> tvm.tir.const(shift, env.acc_dtype), name="C_shr")
    C_clo = te.compute(C_buf.shape,
        lambda *i: tvm.te.max(C_shr(*i), tvm.tir.const(clip_lo, env.acc_dtype)), name="C_clo")
    C_chi = te.compute(C_buf.shape,
        lambda *i: tvm.te.min(C_clo(*i), tvm.tir.const(clip_hi, env.acc_dtype)), name="C_chi")
    C = te.compute(C_buf.shape,
        lambda *i: C_chi(*i).astype(env.out_dtype), name="C")

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
    return vta.build(s, [A, B, C], tvm.target.vta(), host, name="my_gemm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=WEIGHTS_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    import tvm
    import vta
    env = vta.get_env()
    print(f"[env] TARGET={env.TARGET} INP={env.INP_WIDTH} WGT={env.WGT_WIDTH} "
          f"OUT={env.OUT_WIDTH} ACC={env.ACC_WIDTH} BLOCK={env.BLOCK_IN}/{env.BLOCK_OUT}")
    assert env.OUT_WIDTH == 8, f"Expected OUT_WIDTH=8, got {env.OUT_WIDTH}"
    assert env.INP_WIDTH == 4

    os.makedirs(args.output_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.weights_dir, "meta.json")))
    act_scale = [float(np.load(os.path.join(args.weights_dir, f"act_scale_{e['index']}.npy")))
                 for e in meta["act_scales"]]

    layers = []

    # ---- Conv layers ----
    for ci, cl in enumerate(meta["conv_layers"]):
        W_int = np.load(os.path.join(args.weights_dir, f"W{cl['index']}.npy"))
        w_scale = np.load(os.path.join(args.weights_dir, f"w_scale_{cl['index']}.npy"))
        bias_float = np.load(os.path.join(args.weights_dir, f"b{cl['index']}.npy"))
        C_out, C_in, kH, kW = W_int.shape
        tiling = CONV_TILING[ci]

        # HWC transpose + flatten for im2col GEMM
        W_flat = W_int.transpose(0, 2, 3, 1).reshape(C_out, -1)

        # Pad to BLOCK multiples
        W_flat = pad_to_block(W_flat, env.BLOCK_OUT, axis=0)
        W_flat = pad_to_block(W_flat, env.BLOCK_IN, axis=1)
        out_f_padded, in_f_padded = W_flat.shape
        m = out_f_padded // env.BLOCK_OUT
        n = in_f_padded // env.BLOCK_IN

        # Tile to VTA layout: (m, n, BLOCK_OUT, BLOCK_IN)
        W_tiled = W_flat.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)

        # Per-channel combined_scale: w_scale[c] * act_scale[ci]
        w_scale_vec = w_scale if w_scale.ndim >= 1 else np.full(C_out, float(w_scale))
        combined_scale = w_scale_vec.astype(np.float64) * act_scale[ci]
        # Pad combined_scale for padded output channels
        cs_padded = np.zeros(out_f_padded, dtype=np.float64)
        cs_padded[:C_out] = combined_scale

        # Zero-point correction: zp * sum(W_flat_padded[c, :])
        zp_corr = ZERO_POINT * W_flat.astype(np.int32).sum(axis=1)  # (out_f_padded,)

        # Fold bias to int32: round(bias_float / combined_scale)
        bias_int = np.zeros(out_f_padded, dtype=np.int32)
        for c in range(C_out):
            bias_int[c] = round(float(bias_float[c]) / combined_scale[c])

        # Corrected bias: fold_bias_int32 + zp_correction
        corrected_bias = bias_int + zp_corr

        # Calibrated shift (from sim: [2, 2] for convs)
        shift = [2, 2][ci]

        # Tile bias to VTA layout: (o_tile, m, BATCH, BLOCK_OUT) for broadcast
        # Bias is per-output-channel, broadcast over spatial. For the VTA ALU ADD
        # the bias tensor must match output shape (o, m, BATCH, BLOCK_OUT).
        # Tile corrected_bias to (m, BLOCK_OUT): (1, 16) for [8,16] model
        corrected_bias_tiled = corrected_bias.reshape(m, env.BLOCK_OUT)

        # Compile the 4-arg module for this layer's tile shape
        print(f"\nConv{ci+1}: W{W_int.shape} → padded ({out_f_padded},{in_f_padded}) "
              f"→ tiled ({m},{n},{env.BLOCK_OUT},{env.BLOCK_IN}) "
              f"o_tile={tiling['o_tile']} shift={shift}")
        print(f"  zp_corr range [{zp_corr[:C_out].min()}, {zp_corr[:C_out].max()}]")
        print(f"  corrected_bias range [{corrected_bias[:C_out].min()}, "
              f"{corrected_bias[:C_out].max()}]")

        mod = compile_gemm_bias_shr_clip_int8(
            env, tiling["o_tile"], n, m, shift, INT8_CLIP_LO, INT8_CLIP_HI)
        mod_path = os.path.join(args.output_dir, f"layer{ci}.o")
        mod.save(mod_path)
        print(f"  → {mod_path} ({os.path.getsize(mod_path)} bytes)")

        # Save tiled weights + corrected bias
        np.save(os.path.join(args.output_dir, f"W{ci}_tiled.npy"),
                W_tiled.astype(np.int8))
        np.save(os.path.join(args.output_dir, f"b{ci}_corrected.npy"),
                corrected_bias_tiled.astype(np.int32))

        layers.append({
            "type": "conv", "index": ci,
            "W_shape_orig": list(W_int.shape),
            "W_shape_padded": [out_f_padded, in_f_padded],
            "m": m, "n": n,
            "o_tile": tiling["o_tile"], "n_chunks": tiling["n_chunks"],
            "o_total": tiling["o_tile"] * tiling["n_chunks"],
            "shift": shift, "clip_lo": INT8_CLIP_LO, "clip_hi": INT8_CLIP_HI,
            "n_args": 4,
            "module_file": f"layer{ci}.o",
            "W_file": f"W{ci}_tiled.npy",
            "bias_file": f"b{ci}_corrected.npy",
            "C_out_valid": C_out,
            "w_scale": w_scale_vec.tolist(),
            "act_scale_in": act_scale[ci],
            "act_scale_out": act_scale[ci + 1],
            "combined_scale": combined_scale.tolist(),
            # benchmark.py compatibility fields:
            "weight_file": f"W{ci}_tiled.npy",
            "in_f": in_f_padded, "out_f": out_f_padded,
            "real_out": C_out,
            "n_tiles": n, "m_tiles": m,
            "has_vta_bias": True,
            "in_scale": act_scale[ci],
            "kernel_size": 3, "padding": 1,
            "in_channels": C_in, "out_channels": C_out,
            "pool": 2,
        })

    # ---- Dense layer ----
    dl = meta["dense_layers"][0]
    W_dense = np.load(os.path.join(args.weights_dir, f"W{dl['index']}.npy"))
    w_scale_d = np.load(os.path.join(args.weights_dir, f"w_scale_{dl['index']}.npy"))
    bias_d_float = np.load(os.path.join(args.weights_dir, f"b{dl['index']}.npy"))
    out_d, in_d = W_dense.shape

    W_d_padded = pad_to_block(pad_to_block(W_dense, env.BLOCK_OUT, 0), env.BLOCK_IN, 1)
    out_d_p, in_d_p = W_d_padded.shape
    m_d = out_d_p // env.BLOCK_OUT
    n_d = in_d_p // env.BLOCK_IN
    W_d_tiled = W_d_padded.reshape(m_d, env.BLOCK_OUT, n_d, env.BLOCK_IN).transpose(0, 2, 1, 3)

    # Dense shift (from sim: 0 for dense)
    shift_d = 0

    # Dense: 4-arg module (bias in VTA). Required for Mode G — the 3-arg
    # pattern clips W @ x_vta to [-128,127] BEFORE the zero-point correction
    # is absorbed. With zpc up to ±136, this silently saturates output channels
    # whose correction magnitude exceeds 127. Using 4-arg: VTA does
    # GEMM + corrected_bias + SHR + CLIP, so the zp correction is applied
    # before narrowing. CPU just dequants and argmaxes (no bias addition).
    w_scale_d_vec = w_scale_d if w_scale_d.ndim >= 1 else np.full(out_d, float(w_scale_d))
    last_act_scale = act_scale[-1]
    combined_d = w_scale_d_vec.astype(np.float64) * last_act_scale

    # Corrected int32 bias for VTA ALU ADD (same formula as conv layers)
    cs_d_padded = np.zeros(out_d_p, dtype=np.float64)
    cs_d_padded[:out_d] = combined_d
    zp_corr_d = ZERO_POINT * W_d_padded.astype(np.int32).sum(axis=1)  # (out_d_p,)
    bias_d_int = np.zeros(out_d_p, dtype=np.int32)
    for c in range(out_d):
        bias_d_int[c] = round(float(bias_d_float[c]) / combined_d[c])
    corrected_bias_d = bias_d_int + zp_corr_d  # (out_d_p,)
    corrected_bias_d_tiled = corrected_bias_d.reshape(m_d, env.BLOCK_OUT)

    print(f"\nDense: W{W_dense.shape} → padded ({out_d_p},{in_d_p}) "
          f"→ tiled ({m_d},{n_d},{env.BLOCK_OUT},{env.BLOCK_IN}) shift={shift_d}")
    print(f"  zp_corr_d range [{zp_corr_d[:out_d].min()}, {zp_corr_d[:out_d].max()}]")
    print(f"  corrected_bias_d range [{corrected_bias_d[:out_d].min()}, "
          f"{corrected_bias_d[:out_d].max()}]")

    mod_d = compile_gemm_bias_shr_clip_int8(
        env, 1, n_d, m_d, shift_d, INT8_CLIP_LO, INT8_CLIP_HI)
    mod_d_path = os.path.join(args.output_dir, f"layer{len(meta['conv_layers'])}.o")
    mod_d.save(mod_d_path)
    print(f"  → {mod_d_path} ({os.path.getsize(mod_d_path)} bytes)")

    np.save(os.path.join(args.output_dir, f"W{len(meta['conv_layers'])}_tiled.npy"),
            W_d_tiled.astype(np.int8))
    np.save(os.path.join(args.output_dir, f"b{len(meta['conv_layers'])}_corrected.npy"),
            corrected_bias_d_tiled.astype(np.int32))

    layers.append({
        "type": "dense", "index": len(meta["conv_layers"]),
        "W_shape_orig": list(W_dense.shape),
        "m": m_d, "n": n_d,
        "o_tile": 1, "n_chunks": 1, "o_total": 1,
        "shift": shift_d, "clip_lo": INT8_CLIP_LO, "clip_hi": INT8_CLIP_HI,
        "n_args": 4,
        "module_file": f"layer{len(meta['conv_layers'])}.o",
        "W_file": f"W{len(meta['conv_layers'])}_tiled.npy",
        "bias_file": f"b{len(meta['conv_layers'])}_corrected.npy",
        "C_out_valid": out_d,
        "w_scale": w_scale_d_vec.tolist(),
        "act_scale_in": last_act_scale,
        "combined_scale": combined_d[:out_d].tolist(),
        # benchmark.py compatibility fields:
        "weight_file": f"W{len(meta['conv_layers'])}_tiled.npy",
        "in_f": in_d_p, "out_f": out_d_p,
        "real_out": out_d,
        "n_tiles": n_d, "m_tiles": m_d,
        "has_vta_bias": True,
        "in_scale": last_act_scale,
        "pool": 0,
    })

    # ---- Config JSON ----
    config = {
        "model_type": "cnn_perchan_o8",
        "architecture": meta["architecture"],
        "num_layers": len(layers),
        "zero_point": ZERO_POINT,
        "out_dtype": "int8",
        "requant_mode": "vta_native_o8",
        "clock_mhz": 166,
        "act_scales_brevitas": [float(s) for s in act_scale],
        "BLOCK_IN": env.BLOCK_IN, "BLOCK_OUT": env.BLOCK_OUT, "BATCH": env.BATCH,
        "layers": layers,
        "pipeline": (
            "Mode G: input offset-encoded [0,15]→[-8,7] via zp=8. "
            "Hidden conv: 4-arg (GEMM+corrected_bias+SHR+CLIP[-128,127]→int8). "
            "Dense: 3-arg (GEMM+SHR+CLIP→int8) + CPU float corrected_bias + argmax. "
            "CPU between layers: per-channel dequant, ReLU, MaxPool, requant to [0,15] then offset."
        ),
    }
    cfg_path = os.path.join(args.output_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[done] Exported to {args.output_dir}")
    for fn in sorted(os.listdir(args.output_dir)):
        print(f"  {fn} ({os.path.getsize(os.path.join(args.output_dir, fn))} bytes)")


if __name__ == "__main__":
    main()
