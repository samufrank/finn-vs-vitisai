#!/usr/bin/env python3
"""Export VTA CNN INT4-input/INT8-output modules for board-side inference.

Mode G pipeline: INT4 input via zero-point offset (Brevitas [0,15] →
VTA [-8,7] via zp=8), INT8 DMA output, per-channel weight scales,
int32 corrected bias = fold_bias_int32 + zp * sum(W_int, input_axis).

Per hidden conv layer: 4-arg module (A, B, D_bias, C_out).
  VTA: GEMM → ALU ADD corrected_bias → SHR → CLIP[-128,127] → int8 out.
Dense (last): 4-arg module (A, B, D_bias, C_out) too — required for Mode G
  to apply zp correction before INT8 narrowing (the legacy 3-arg form
  silently saturates output channels whose zp correction exceeds ±127).
  CPU: dequant + argmax (no bias addition; bias absorbed into int32).

Generalized in this revision (was hardcoded to tiny [8,16]):
  --size selects channels via models.cnn.get_cnn_config (tiny → small →
    medium → deep_3 → large)
  Inline weight extraction walks Brevitas Sequential dynamically. Two
    checkpoint shapes accepted:
      A. <prefix>_perchan_bnfold.pth: CNN_Brevitas_INT4_PerChan with BN
         already merged (BN modules become Identity, biases on convs).
         Direct load; quant_weight() returns per-channel scales.
      B. <prefix>_int4.pth (default Brevitas INT4): CNN_Brevitas_INT4
         with per-tensor weight quant + BN. Loaded directly, merge_bn
         applied here, then weights re-quantized as per-channel max-abs
         (since per-tensor scale loses precision after BN fold).
  Spatial dims tracked dynamically (28 → /pool → /pool → ...) so
    o_total per conv = current spatial (matches export_vta_cnn.py:WI3).
  CONV_TILING: o_tile = post-pool spatial when pool divides cleanly,
    n_chunks = pool²; otherwise o_tile = o_total, n_chunks = 1.
    For tiny [8,16] this reproduces the original [196/4, 49/4] tiling.
  Shifts: tiny uses calibrated [2, 2, 0] (regression-preserving). Other
    sizes use a worst-case heuristic ceil(log2(n_in * 15 * 7 / 127));
    deploy accuracy at non-tiny sizes will need real shift calibration
    via the archive Mode G sim before any board run.

Usage:
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \\
        python3 ../finn-vs-vitisai/board/export_vta_cnn_int4_o8.py \\
            --size tiny \\
            --checkpoint ../finn-vs-vitisai/finn/cnn_mnist_tiny_int4_perchan_bnfold.pth \\
            --output-dir ../finn-vs-vitisai/vta_exports/cnn_tiny_int4_perchan/
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ZERO_POINT = 8
INT8_CLIP_LO = -128
INT8_CLIP_HI = 127

# Per-size calibrated shift lists from board/calibrate_int4_shifts.py.
# Each entry is the full shift list (one per layer in topology order) that
# board calibration found to maximize end-to-end argmax accuracy on a 100-
# image MNIST cal set, validated against the full 10K test set:
#   tiny    [2, 2, 0]: 81.57% (board-validated deploy)
#   small   [2, 3, 0]: calibrated; board run pending
#   medium  [1, 3, 0]: 89.85% (numpy-sim; board run pending)
# Sizes not in this table fall back to the worst-case heuristic in
# derive_shifts() and PRINT A WARNING — those produce compilable artifacts
# but should be calibrated before deploy.
CALIBRATED_SHIFTS = {
    'tiny':   [2, 2, 0],
    'small':  [2, 3, 0],
    'medium': [1, 3, 0],
}


def pad_to_block(arr, block, axis):
    s = arr.shape[axis]
    if s % block == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, block - s % block)
    return np.pad(arr, pad_width, mode='constant')


# ---- VTA module compilation (unchanged from original) ---------------

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


# ---- Inline weight extraction (replaces meta.json + W{i}.npy upstream) ---

def load_brevitas_int4(checkpoint_path, size):
    """Load Brevitas INT4 CNN, BN-fold if needed, return per-layer dicts.

    Returns: (layers, act_scales)
      layers: list of dicts:
        conv: {type, W_int (C_out,C_in,kH,kW) int8, w_scale (C_out,) f64,
               b (C_out,) f32, kernel_size, padding, in_channels,
               out_channels, pool}
        dense: {type, W_int (out_f, in_f) int8, w_scale (out_f,) f64,
                b (out_f,) f32}
      act_scales: [synthetic_input, post_relu_1, post_relu_2, ...]
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, 'models'))
    import brevitas.nn as qnn
    from brevitas.nn.utils import merge_bn
    from cnn import (CNN_Brevitas_INT4, CNN_Brevitas_INT4_PerChan,
                     get_cnn_config)

    channels = get_cnn_config(size)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    # Detect checkpoint shape: post-fold PerChan has Identity for BN slots
    # (no running_mean keys), with bias keys on the convs.
    has_bn = any('running_mean' in k for k in sd.keys())

    if not has_bn:
        # Path A: post-fold PerChan checkpoint. Reproduce the exact structure
        # used by extract_int4_brevitas_cnn_perchan.py: Identity for BN
        # modules, biases on QuantConv2d. Brevitas's quant_weight() returns
        # the per-channel scales saved in the .pth.
        model = CNN_Brevitas_INT4_PerChan(in_channels=1, num_classes=10,
                                           channels=channels)
        feats = list(model.features)
        for i, m in enumerate(feats):
            if isinstance(m, nn.BatchNorm2d):
                model.features[i] = nn.Identity()
        for m in model.features:
            if isinstance(m, qnn.QuantConv2d) and m.bias is None:
                c_out = m.weight.shape[0]
                m.bias = nn.Parameter(torch.zeros(c_out))
        model.load_state_dict(sd, strict=True)
        used_brevitas_quant = True
    else:
        # Path B: per-tensor INT4 with BN. Load CNN_Brevitas_INT4 directly,
        # apply merge_bn, then re-quantize weights as per-channel max-abs
        # (per-tensor scale loses precision after BN fold's per-channel
        # gamma/sqrt(var+eps) spread).
        model = CNN_Brevitas_INT4(in_channels=1, num_classes=10,
                                   channels=channels)
        model.load_state_dict(sd, strict=True)
        feats = list(model.features)
        i = 0
        while i < len(feats):
            if isinstance(feats[i], qnn.QuantConv2d):
                conv = feats[i]
                j = i + 1
                while j < len(feats) and not isinstance(feats[j], qnn.QuantConv2d):
                    if isinstance(feats[j], nn.BatchNorm2d):
                        merge_bn(conv, feats[j])
                        model.features[j] = nn.Identity()
                        break
                    j += 1
                i = j
            else:
                i += 1
        used_brevitas_quant = False

    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 28, 28))

    layers = []
    relu_modules = []
    feats = list(model.features)
    i = 0
    while i < len(feats):
        m = feats[i]
        if isinstance(m, qnn.QuantConv2d):
            conv = m
            pool = 0
            j = i + 1
            while j < len(feats) and not isinstance(feats[j], qnn.QuantConv2d):
                if isinstance(feats[j], qnn.QuantReLU):
                    relu_modules.append(feats[j])
                elif isinstance(feats[j], nn.MaxPool2d):
                    p = feats[j].kernel_size
                    pool = p[0] if isinstance(p, tuple) else int(p)
                j += 1

            if used_brevitas_quant:
                qw = conv.quant_weight()
                W_int = qw.int().detach().numpy().astype(np.int8)
                w_scale = qw.scale.detach().numpy().astype(np.float64).reshape(-1)
                if w_scale.shape[0] == 1:
                    w_scale = np.full(W_int.shape[0], float(w_scale[0]))
            else:
                # Manual per-channel max-abs INT4 quantization on folded weights.
                W_float = conv.weight.detach().numpy().astype(np.float64)
                C_out = W_float.shape[0]
                abs_max = np.abs(W_float).reshape(C_out, -1).max(axis=1)
                # Avoid division by zero on dead channels.
                abs_max = np.maximum(abs_max, 1e-12)
                w_scale = abs_max / 7.0
                W_int = np.round(W_float /
                                  w_scale.reshape(C_out, 1, 1, 1)).clip(-7, 7).astype(np.int8)

            assert w_scale.shape == (W_int.shape[0],), \
                f"per-channel scale shape {w_scale.shape} vs C_out={W_int.shape[0]}"
            assert W_int.min() >= -7 and W_int.max() <= 7, \
                f"int range [{W_int.min()}, {W_int.max()}] outside [-7, 7]"

            if conv.bias is not None:
                b = conv.bias.detach().numpy().astype(np.float32)
            else:
                b = np.zeros(W_int.shape[0], dtype=np.float32)

            C_out, C_in, kH, _ = W_int.shape
            pad = conv.padding
            if isinstance(pad, tuple):
                pad = pad[0]

            layers.append({
                'type':         'conv',
                'W_int':        W_int,
                'w_scale':      w_scale,
                'b':            b,
                'kernel_size':  int(kH),
                'padding':      int(pad),
                'in_channels':  int(C_in),
                'out_channels': int(C_out),
                'pool':         int(pool),
            })
            i = j
        else:
            i += 1

    cls_lins = [m for m in model.classifier if isinstance(m, qnn.QuantLinear)]
    assert len(cls_lins) == 1, f'expected 1 QuantLinear; got {len(cls_lins)}'
    cls = cls_lins[0]

    if used_brevitas_quant:
        qw = cls.quant_weight()
        W_int_d = qw.int().detach().numpy().astype(np.int8)
        w_scale_d = qw.scale.detach().numpy().astype(np.float64).reshape(-1)
        if w_scale_d.shape[0] == 1:
            w_scale_d = np.full(W_int_d.shape[0], float(w_scale_d[0]))
    else:
        W_float = cls.weight.detach().numpy().astype(np.float64)
        out_f = W_float.shape[0]
        abs_max = np.abs(W_float).reshape(out_f, -1).max(axis=1)
        abs_max = np.maximum(abs_max, 1e-12)
        w_scale_d = abs_max / 7.0
        W_int_d = np.round(W_float / w_scale_d.reshape(out_f, 1)).clip(-7, 7).astype(np.int8)

    layers.append({
        'type':    'dense',
        'W_int':   W_int_d,
        'w_scale': w_scale_d,
        'b':       cls.bias.detach().numpy().astype(np.float32),
    })

    # Activation scales: synthetic input scale + per-ReLU learned scales.
    # Synthetic input scale 1/15 maps MNIST [0,1] → unsigned int4 [0,15].
    act_scales = [1.0 / 15.0]
    for relu in relu_modules:
        scaling_impl = (relu.act_quant.fused_activation_quant_proxy
                        .tensor_quant.scaling_impl)
        raw_value = float(scaling_impl.value.detach().item())
        act_scales.append(raw_value / 15.0)

    return layers, act_scales


def derive_tiling(o_total, pool):
    """Compute (o_tile, n_chunks) for a conv layer.

    Reproduces the tiny [196/4, 49/4] split: when pool² divides o_total,
    use that. Otherwise emit a single-chunk module covering o_total.
    """
    if pool > 0 and o_total % (pool * pool) == 0:
        o_tile = o_total // (pool * pool)
        n_chunks = pool * pool
    else:
        o_tile = o_total
        n_chunks = 1
    return o_tile, n_chunks


def derive_shifts(layers, size):
    """Per-layer right-shift amounts.

    Looks up CALIBRATED_SHIFTS first; falls back to a worst-case heuristic
    (ceil(log2(n_in_padded * 15 * 7 / 127))) for unknown sizes. The
    heuristic produces artifacts that compile and run but saturate a
    non-trivial fraction of activations — recalibrate via
    board/calibrate_int4_shifts.py before any deploy where accuracy matters.
    """
    if size in CALIBRATED_SHIFTS:
        shifts = list(CALIBRATED_SHIFTS[size])
        if len(shifts) != len(layers):
            raise ValueError(
                f"CALIBRATED_SHIFTS['{size}'] has {len(shifts)} entries but "
                f"the topology has {len(layers)} layers")
        return shifts

    shifts = []
    for L in layers:
        if L['type'] == 'conv':
            C_in, kH = L['in_channels'], L['kernel_size']
            n_in_padded = max(16, ((kH * kH * C_in + 15) // 16) * 16)
        else:
            n_in_padded = max(16, ((L['W_int'].shape[1] + 15) // 16) * 16)
        max_acc = n_in_padded * 15 * 7
        s = max(0, int(math.ceil(math.log2(max_acc / 127.0))))
        shifts.append(s)
    return shifts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='tiny',
                    help='CNN size config name (tiny, small, medium, deep_3, large)')
    ap.add_argument('--checkpoint', required=True,
                    help='Brevitas INT4 .pth (either *_perchan_bnfold.pth or '
                         'cnn_mnist_<size>_int4.pth — both auto-detected).')
    ap.add_argument('--output-dir', required=True,
                    help='Output directory for compiled artifacts.')
    args = ap.parse_args()

    import tvm
    import vta
    env = vta.get_env()
    print(f"[env] TARGET={env.TARGET} INP={env.INP_WIDTH} WGT={env.WGT_WIDTH} "
          f"OUT={env.OUT_WIDTH} ACC={env.ACC_WIDTH} BLOCK={env.BLOCK_IN}/{env.BLOCK_OUT}")
    assert env.OUT_WIDTH == 8, f"Expected OUT_WIDTH=8, got {env.OUT_WIDTH}"
    assert env.INP_WIDTH == 4

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading Brevitas INT4 CNN: {args.checkpoint} (size={args.size})")
    layers_raw, act_scales = load_brevitas_int4(args.checkpoint, args.size)
    for i, L in enumerate(layers_raw):
        if L['type'] == 'conv':
            print(f"  Layer {i} (conv): W{tuple(L['W_int'].shape)} "
                  f"int range [{L['W_int'].min()}, {L['W_int'].max()}], "
                  f"per-channel w_scale [{L['w_scale'].min():.6f}, {L['w_scale'].max():.6f}], "
                  f"bias range [{L['b'].min():+.4f}, {L['b'].max():+.4f}], "
                  f"pool={L['pool']}")
        else:
            print(f"  Layer {i} (dense): W{tuple(L['W_int'].shape)} "
                  f"int range [{L['W_int'].min()}, {L['W_int'].max()}], "
                  f"per-channel w_scale [{L['w_scale'].min():.6f}, {L['w_scale'].max():.6f}]")
    print(f"  act_scales (input + per-ReLU): {[f'{s:.6f}' for s in act_scales]}")

    shifts = derive_shifts(layers_raw, args.size)
    if args.size in CALIBRATED_SHIFTS:
        print(f"  shifts: {shifts}  (calibrated via board/calibrate_int4_shifts.py)")
    else:
        print(f"  shifts: {shifts}  (heuristic — recalibrate before deploy)")

    # Spatial walk: 28×28 input, each pool divides spatial.
    cur_h, cur_w = 28, 28

    config_layers = []
    layer_index = 0
    for li, L in enumerate(layers_raw):
        shift = shifts[li]
        if L['type'] == 'conv':
            o_total = cur_h * cur_w
            o_tile, n_chunks = derive_tiling(o_total, L['pool'])

            W_int = L['W_int']                                  # (C_out, C_in, kH, kW)
            C_out, C_in, kH, kW = W_int.shape
            # HWC transpose + flatten for im2col GEMM input order.
            W_flat = W_int.transpose(0, 2, 3, 1).reshape(C_out, -1)
            W_flat = pad_to_block(W_flat, env.BLOCK_OUT, axis=0)
            W_flat = pad_to_block(W_flat, env.BLOCK_IN, axis=1)
            out_f_padded, in_f_padded = W_flat.shape
            m = out_f_padded // env.BLOCK_OUT
            n = in_f_padded // env.BLOCK_IN
            W_tiled = W_flat.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)

            # Per-channel combined scale, padded out to BLOCK_OUT multiple.
            in_scale = act_scales[li]
            combined = L['w_scale'].astype(np.float64) * in_scale  # (C_out,)
            cs_padded = np.zeros(out_f_padded, dtype=np.float64)
            cs_padded[:C_out] = combined

            # Zero-point correction: zp * sum(W_int_padded[c, :]).
            zp_corr = ZERO_POINT * W_flat.astype(np.int32).sum(axis=1)  # (out_f_padded,)

            # Folded bias as int32: round(b_float / combined_scale).
            bias_int = np.zeros(out_f_padded, dtype=np.int32)
            for c in range(C_out):
                bias_int[c] = round(float(L['b'][c]) / combined[c])
            corrected_bias = bias_int + zp_corr
            corrected_bias_tiled = corrected_bias.reshape(m, env.BLOCK_OUT)

            print(f"\nConv{li}: W{tuple(W_int.shape)} → padded ({out_f_padded},{in_f_padded}) "
                  f"→ tiled ({m},{n},{env.BLOCK_OUT},{env.BLOCK_IN}) "
                  f"o_total={o_total} o_tile={o_tile} n_chunks={n_chunks} shift={shift}")
            print(f"  zp_corr range [{zp_corr[:C_out].min()}, {zp_corr[:C_out].max()}]")
            print(f"  corrected_bias range [{corrected_bias[:C_out].min()}, "
                  f"{corrected_bias[:C_out].max()}]")

            mod = compile_gemm_bias_shr_clip_int8(
                env, o_tile, n, m, shift, INT8_CLIP_LO, INT8_CLIP_HI)
            mod_path = os.path.join(args.output_dir, f"layer{layer_index}.o")
            mod.save(mod_path)
            print(f"  → {os.path.basename(mod_path)} ({os.path.getsize(mod_path)} bytes)")

            np.save(os.path.join(args.output_dir, f"W{layer_index}_tiled.npy"),
                    W_tiled.astype(np.int8))
            np.save(os.path.join(args.output_dir, f"b{layer_index}_corrected.npy"),
                    corrected_bias_tiled.astype(np.int32))

            config_layers.append({
                "type": "conv", "index": layer_index,
                "W_shape_orig": list(W_int.shape),
                "W_shape_padded": [out_f_padded, in_f_padded],
                "m": m, "n": n,
                "o_tile": o_tile, "n_chunks": n_chunks, "o_total": o_total,
                "shift": shift, "clip_lo": INT8_CLIP_LO, "clip_hi": INT8_CLIP_HI,
                "n_args": 4,
                "module_file": f"layer{layer_index}.o",
                "W_file": f"W{layer_index}_tiled.npy",
                "bias_file": f"b{layer_index}_corrected.npy",
                "C_out_valid": C_out,
                "w_scale": L['w_scale'].tolist(),
                "act_scale_in": in_scale,
                "act_scale_out": act_scales[li + 1] if li + 1 < len(act_scales) else None,
                "combined_scale": combined.tolist(),
                "weight_file": f"W{layer_index}_tiled.npy",
                "in_f": in_f_padded, "out_f": out_f_padded,
                "real_out": C_out,
                "n_tiles": n, "m_tiles": m,
                "has_vta_bias": True,
                "in_scale": in_scale,
                "kernel_size": kH, "padding": L['padding'],
                "in_channels": C_in, "out_channels": C_out,
                "pool": L['pool'],
            })

            if L['pool'] > 0:
                cur_h //= L['pool']
                cur_w //= L['pool']

        else:  # dense
            W_int = L['W_int']
            out_d, in_d = W_int.shape
            W_d_padded = pad_to_block(pad_to_block(W_int, env.BLOCK_OUT, 0),
                                       env.BLOCK_IN, 1)
            out_d_p, in_d_p = W_d_padded.shape
            m_d = out_d_p // env.BLOCK_OUT
            n_d = in_d_p // env.BLOCK_IN
            W_d_tiled = W_d_padded.reshape(m_d, env.BLOCK_OUT, n_d, env.BLOCK_IN).transpose(0, 2, 1, 3)

            in_scale = act_scales[-1]
            combined = L['w_scale'].astype(np.float64) * in_scale
            zp_corr_d = ZERO_POINT * W_d_padded.astype(np.int32).sum(axis=1)
            bias_d_int = np.zeros(out_d_p, dtype=np.int32)
            for c in range(out_d):
                bias_d_int[c] = round(float(L['b'][c]) / combined[c])
            corrected_bias_d = bias_d_int + zp_corr_d
            corrected_bias_d_tiled = corrected_bias_d.reshape(m_d, env.BLOCK_OUT)

            print(f"\nDense (layer{layer_index}): W{tuple(W_int.shape)} → padded "
                  f"({out_d_p},{in_d_p}) → tiled ({m_d},{n_d},{env.BLOCK_OUT},{env.BLOCK_IN}) "
                  f"shift={shift}")
            print(f"  zp_corr_d range [{zp_corr_d[:out_d].min()}, {zp_corr_d[:out_d].max()}]")
            print(f"  corrected_bias_d range [{corrected_bias_d[:out_d].min()}, "
                  f"{corrected_bias_d[:out_d].max()}]")

            mod = compile_gemm_bias_shr_clip_int8(
                env, 1, n_d, m_d, shift, INT8_CLIP_LO, INT8_CLIP_HI)
            mod_path = os.path.join(args.output_dir, f"layer{layer_index}.o")
            mod.save(mod_path)
            print(f"  → {os.path.basename(mod_path)} ({os.path.getsize(mod_path)} bytes)")

            np.save(os.path.join(args.output_dir, f"W{layer_index}_tiled.npy"),
                    W_d_tiled.astype(np.int8))
            np.save(os.path.join(args.output_dir, f"b{layer_index}_corrected.npy"),
                    corrected_bias_d_tiled.astype(np.int32))

            config_layers.append({
                "type": "dense", "index": layer_index,
                "W_shape_orig": list(W_int.shape),
                "m": m_d, "n": n_d,
                "o_tile": 1, "n_chunks": 1, "o_total": 1,
                "shift": shift, "clip_lo": INT8_CLIP_LO, "clip_hi": INT8_CLIP_HI,
                "n_args": 4,
                "module_file": f"layer{layer_index}.o",
                "W_file": f"W{layer_index}_tiled.npy",
                "bias_file": f"b{layer_index}_corrected.npy",
                "C_out_valid": out_d,
                "w_scale": L['w_scale'].tolist(),
                "act_scale_in": in_scale,
                "combined_scale": combined[:out_d].tolist(),
                "weight_file": f"W{layer_index}_tiled.npy",
                "in_f": in_d_p, "out_f": out_d_p,
                "real_out": out_d,
                "n_tiles": n_d, "m_tiles": m_d,
                "has_vta_bias": True,
                "in_scale": in_scale,
                "pool": 0,
            })

        layer_index += 1

    config = {
        "model_type": "cnn_perchan_o8",
        "size": args.size,
        "architecture": f"cnn_{args.size}_int4_perchan_mnist",
        "num_layers": len(config_layers),
        "zero_point": ZERO_POINT,
        "out_dtype": "int8",
        "requant_mode": "vta_native_o8",
        "clock_mhz": 166,
        "act_scales_brevitas": [float(s) for s in act_scales],
        "BLOCK_IN": env.BLOCK_IN, "BLOCK_OUT": env.BLOCK_OUT, "BATCH": env.BATCH,
        "layers": config_layers,
        "pipeline": (
            "Mode G: input offset-encoded [0,15]→[-8,7] via zp=8. "
            "Hidden conv: 4-arg (GEMM+corrected_bias+SHR+CLIP[-128,127]→int8). "
            "Dense: 4-arg (GEMM+corrected_bias+SHR+CLIP[-128,127]→int8) + CPU dequant + argmax. "
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
