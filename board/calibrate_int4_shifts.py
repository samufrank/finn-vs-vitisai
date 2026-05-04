#!/usr/bin/env python3
"""calibrate_int4_shifts.py — analytical shift calibration for VTA CNN INT4
deploys (Mode G: zero-point-offset INT4 input, INT8 output).

For each layer:
  1. Numpy-simulate the Brevitas-equivalent forward path over N calibration
     MNIST images (default 100). Use float passthrough between layers (no
     SHR yet) so each layer's max-abs corrected-accumulator value is the
     unconstrained upper bound — the true ceiling any SHR has to fit under.
  2. Initial shift = max(0, ceil(log2(max_abs / 127))). At shift = this
     value, no calibration sample saturates the int8 [-128, 127] window.
  3. Greedy per-layer ±1 search: with all other layers fixed at their best,
     try {init-1, init, init+1} for the current layer, run end-to-end
     argmax accuracy on the cal set, keep the shift with highest accuracy
     (tie → prefer the analytical init). Move to the next layer.
  4. Final shifts + accuracy printed; the shift list is what
     export_vta_cnn_int4_o8.py should be hand-edited to use (replacing
     the worst-case heuristic in derive_shifts() for non-tiny sizes).

Reuses load_brevitas_int4 from export_vta_cnn_int4_o8 — same checkpoint
detection (PerChan-bnfold direct vs per-tensor + merge_bn fallback), same
weight extraction. Topology-agnostic: 2-conv tiny, 3-conv deep_3 / large,
and any future N-conv CNN_Brevitas_INT4 variant work without code changes.

Usage:
  python3 board/calibrate_int4_shifts.py --size tiny
  python3 board/calibrate_int4_shifts.py --size medium
  python3 board/calibrate_int4_shifts.py --size deep_3 --num-samples 200
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, THIS_DIR)
from export_vta_cnn_int4_o8 import load_brevitas_int4

ZERO_POINT   = 8
INT8_CLIP_LO = -128
INT8_CLIP_HI = 127


# ---- MNIST (raw idx files) -----------------------------------------------

def load_mnist_test(mnist_raw_dir, n=None):
    """Returns (images uint8 [0..255] shape (N,28,28), labels uint8 (N,))."""
    with open(os.path.join(mnist_raw_dir, 't10k-images-idx3-ubyte'), 'rb') as f:
        magic, count, H, W = struct.unpack('>IIII', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, H, W).copy()
    with open(os.path.join(mnist_raw_dir, 't10k-labels-idx1-ubyte'), 'rb') as f:
        magic, count = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
    if n is not None:
        imgs = imgs[:n]
        labels = labels[:n]
    return imgs, labels


# ---- numpy simulator helpers ---------------------------------------------

def im2col_with_pad_value(x_chw, kH, kW, pad, pad_value):
    """im2col over a CHW int array with configurable pad value (needed for
    offset-encoded activations where 'zero activation' maps to -zero_point,
    not 0)."""
    C, H, W = x_chw.shape
    if pad > 0:
        x_pad = np.full((C, H + 2 * pad, W + 2 * pad), pad_value,
                        dtype=x_chw.dtype)
        x_pad[:, pad:pad + H, pad:pad + W] = x_chw
    else:
        x_pad = x_chw
    out_H = H + 2 * pad - kH + 1
    out_W = W + 2 * pad - kW + 1
    cols = np.empty((out_H * out_W, kH * kW * C), dtype=x_chw.dtype)
    idx = 0
    for i in range(out_H):
        for j in range(out_W):
            cols[idx] = x_pad[:, i:i + kH, j:j + kW].transpose(1, 2, 0).reshape(-1)
            idx += 1
    return cols


def conv_gemm_offset(x_int_chw, W_int, pad, pad_value):
    """int32 GEMM via im2col with offset-aware padding. Returns (C_out, H, W)."""
    C_out, C_in, kH, kW = W_int.shape
    _, H, W = x_int_chw.shape
    out_H = H + 2 * pad - kH + 1
    out_W = W + 2 * pad - kW + 1
    patches = im2col_with_pad_value(x_int_chw.astype(np.int32),
                                     kH, kW, pad, pad_value)
    W_flat = W_int.transpose(0, 2, 3, 1).reshape(C_out, -1).astype(np.int32)
    acc = patches @ W_flat.T                              # (out_H * out_W, C_out)
    return acc.reshape(out_H, out_W, C_out).transpose(2, 0, 1)


def maxpool2d(x_chw, k, s):
    """Max pool over CHW. k = kernel, s = stride."""
    C, H, W = x_chw.shape
    out_h = (H - k) // s + 1
    out_w = (W - k) // s + 1
    out = np.zeros((C, out_h, out_w), dtype=x_chw.dtype)
    for i in range(out_h):
        for j in range(out_w):
            out[:, i, j] = x_chw[:, i * s:i * s + k, j * s:j * s + k].max(axis=(1, 2))
    return out


def adaptive_avg_pool1(x_chw):
    """AdaptiveAvgPool2d(1) → mean over spatial axes, returns (C,)."""
    return x_chw.mean(axis=(1, 2))


def compute_bias_int32(b_float, w_scale_per_ch, in_scale):
    """bias_int32[c] = round(b_float[c] / (w_scale[c] * in_scale))."""
    combined = w_scale_per_ch.astype(np.float64) * in_scale
    return np.round(b_float.astype(np.float64) / combined).astype(np.int32)


def compute_zp_correction_conv(W_int):
    """zp * sum(W_int_padded[c, :]) — but on the original (C_out,C_in,kH,kW)
    int4 tensor. Result is (C_out,). The exported deploy version sums over
    the BLOCK-padded W_flat (which adds zero-padded columns); since added
    zeros contribute 0 to the sum, results agree."""
    return ZERO_POINT * W_int.astype(np.int32).sum(axis=(1, 2, 3))


def compute_zp_correction_dense(W_int):
    """Dense W shape (out_f, in_f). Returns (out_f,)."""
    return ZERO_POINT * W_int.astype(np.int32).sum(axis=1)


# ---- forward passes -------------------------------------------------------

def _layer_setup(L, in_scale):
    """Precompute per-layer correction terms used by both forward variants."""
    combined  = L['w_scale'].astype(np.float64) * in_scale
    bias_int  = compute_bias_int32(L['b'], L['w_scale'], in_scale)
    if L['type'] == 'conv':
        zp_corr = compute_zp_correction_conv(L['W_int'])
    else:
        zp_corr = compute_zp_correction_dense(L['W_int'])
    return combined, bias_int, zp_corr


def forward_for_calibration(img_28x28_float01, layers, act_scales):
    """Float-passthrough Mode G forward (no SHR). Returns max_abs of corrected
    accumulator per layer — independent of any shift choice. The float
    passthrough skips the SHR + clip narrowing so layers downstream see the
    unconstrained ceiling, which is exactly what the analytical shift formula
    needs."""
    x_bre = np.clip(np.round(img_28x28_float01[None] / act_scales[0]),
                    0, 15).astype(np.int32)
    x_vta = x_bre - ZERO_POINT                                 # (1, 28, 28)
    max_abs = []

    for li, L in enumerate(layers):
        in_scale = act_scales[li] if L['type'] == 'conv' else act_scales[-1]
        combined, bias_int, zp_corr = _layer_setup(L, in_scale)

        if L['type'] == 'conv':
            acc = conv_gemm_offset(x_vta, L['W_int'],
                                    pad=L['padding'], pad_value=-ZERO_POINT)
            corrected = acc + (bias_int + zp_corr)[:, None, None]
            max_abs.append(float(np.abs(corrected).max()))

            # Float passthrough: convert int32 corrected → float activation,
            # ReLU, MaxPool, requant for next layer. No SHR applied.
            float_acc = corrected.astype(np.float64) * combined[:, None, None]
            relued = np.maximum(float_acc, 0.0)
            pool = L['pool']
            pooled = maxpool2d(relued, pool, pool) if pool > 0 else relued
            next_scale = act_scales[li + 1] if li + 1 < len(act_scales) else act_scales[-1]
            x_bre = np.clip(np.round(pooled / next_scale), 0, 15).astype(np.int32)
            x_vta = x_bre - ZERO_POINT

        else:  # dense
            x_avg_float = adaptive_avg_pool1(
                x_vta.astype(np.float64) * in_scale + ZERO_POINT * in_scale)
            x_bre_d = np.clip(np.round(x_avg_float / in_scale), 0, 15).astype(np.int32)
            x_vta_d = x_bre_d - ZERO_POINT                     # (C_in,)

            acc = L['W_int'].astype(np.int32) @ x_vta_d.astype(np.int32)
            corrected = acc + bias_int + zp_corr
            max_abs.append(float(np.abs(corrected).max()))

    return max_abs


def forward_with_shifts(img_28x28_float01, layers, act_scales, shifts):
    """Full Mode G forward with given shifts. Returns predicted class index.

    SHR scaling: after `corrected >> shift`, the int8 value y represents
    (W·x + b) / (w_scale · x_scale · 2^shift) — i.e., the original Brevitas
    pre-quant value divided by 2^shift. Dequant therefore multiplies by
    `combined · 2^shift` to recover the float that Brevitas's
    Conv → BN-fold → bias would have produced. Skipping the 2^shift factor
    propagates a 4× / 16× / etc. scale error through every subsequent
    re-quant, collapsing predictions."""
    x_bre = np.clip(np.round(img_28x28_float01[None] / act_scales[0]),
                    0, 15).astype(np.int32)
    x_vta = x_bre - ZERO_POINT

    for li, L in enumerate(layers):
        in_scale = act_scales[li] if L['type'] == 'conv' else act_scales[-1]
        combined, bias_int, zp_corr = _layer_setup(L, in_scale)
        shift = shifts[li]
        post_shr_scale = combined * (1 << shift)               # absorbs the >> shift

        if L['type'] == 'conv':
            acc = conv_gemm_offset(x_vta, L['W_int'],
                                    pad=L['padding'], pad_value=-ZERO_POINT)
            corrected = acc + (bias_int + zp_corr)[:, None, None]
            shifted = corrected >> shift                       # arithmetic shift
            clipped = np.clip(shifted, INT8_CLIP_LO, INT8_CLIP_HI).astype(np.int32)

            float_acc = clipped.astype(np.float64) * post_shr_scale[:, None, None]
            relued = np.maximum(float_acc, 0.0)
            pool = L['pool']
            pooled = maxpool2d(relued, pool, pool) if pool > 0 else relued
            next_scale = act_scales[li + 1] if li + 1 < len(act_scales) else act_scales[-1]
            x_bre = np.clip(np.round(pooled / next_scale), 0, 15).astype(np.int32)
            x_vta = x_bre - ZERO_POINT

        else:  # dense
            x_avg_float = adaptive_avg_pool1(
                x_vta.astype(np.float64) * in_scale + ZERO_POINT * in_scale)
            x_bre_d = np.clip(np.round(x_avg_float / in_scale), 0, 15).astype(np.int32)
            x_vta_d = x_bre_d - ZERO_POINT

            acc = L['W_int'].astype(np.int32) @ x_vta_d.astype(np.int32)
            corrected = acc + bias_int + zp_corr
            shifted = corrected >> shift
            clipped = np.clip(shifted, INT8_CLIP_LO, INT8_CLIP_HI).astype(np.int32)
            float_logits = clipped.astype(np.float64) * post_shr_scale
            return int(np.argmax(float_logits))

    raise RuntimeError('no dense layer found in `layers`')


# ---- calibration --------------------------------------------------------

def initial_shifts_from_max_abs(max_abs_per_layer):
    """Per-layer init: ceil(log2(max_abs / 127)), floored at 0."""
    return [max(0, int(math.ceil(math.log2(v / 127.0)))) if v > 127 else 0
            for v in max_abs_per_layer]


def evaluate_accuracy(layers, act_scales, shifts, images, labels):
    """images: uint8 [0..255]; converts to float [0..1] inside the loop to
    match Brevitas's CNN_Brevitas_INT4 input convention (torchvision ToTensor)."""
    correct = 0
    for img, lbl in zip(images, labels):
        pred = forward_with_shifts(img.astype(np.float32) / 255.0,
                                    layers, act_scales, shifts)
        if pred == int(lbl):
            correct += 1
    return correct / len(images)


def calibrate(layers, act_scales, images, labels, verbose=True):
    """Two-phase calibration: max_abs → analytical init → greedy ±1 per layer.

    Returns (final_shifts, final_accuracy, max_abs_per_layer, init_shifts).
    """
    n = len(layers)

    # Phase 1: per-layer max_abs over the cal set.
    if verbose:
        print(f"\n[1] Measuring max-abs corrected accumulator per layer "
              f"({len(images)} cal images, float-passthrough)...")
    max_abs = [0.0] * n
    for img in images:
        per = forward_for_calibration(img.astype(np.float32) / 255.0,
                                       layers, act_scales)
        for i, v in enumerate(per):
            if v > max_abs[i]:
                max_abs[i] = v

    init_shifts = initial_shifts_from_max_abs(max_abs)
    if verbose:
        for i, (m, s) in enumerate(zip(max_abs, init_shifts)):
            print(f"    layer {i} ({layers[i]['type']:5s}): "
                  f"max_abs={m:9.1f}  →  init shift = ceil(log2({m:.0f}/127)) = {s}")

    # Phase 2: greedy per-layer ±1 search.
    if verbose:
        print(f"\n[2] Greedy per-layer ±1 search "
              f"(3 candidates × {n} layers, end-to-end argmax accuracy on cal set)...")
    current = list(init_shifts)
    for li in range(n):
        candidates = sorted({max(0, init_shifts[li] - 1),
                              init_shifts[li],
                              init_shifts[li] + 1})
        results = []
        for s in candidates:
            trial = list(current)
            trial[li] = s
            acc = evaluate_accuracy(layers, act_scales, trial, images, labels)
            results.append((s, acc))
            if verbose:
                print(f"    layer {li} ({layers[li]['type']:5s}) "
                      f"shift={s}: acc = {acc * 100:6.2f}%  "
                      f"({int(acc * len(images))}/{len(images)})")
        # Tie-break: prefer the analytical init shift.
        results.sort(key=lambda sa: (sa[1], -abs(sa[0] - init_shifts[li])),
                     reverse=True)
        best_s, best_a = results[0]
        current[li] = best_s
        if verbose:
            print(f"    → layer {li} best shift = {best_s} (acc = {best_a*100:.2f}%)")

    final_acc = evaluate_accuracy(layers, act_scales, current, images, labels)
    return current, final_acc, max_abs, init_shifts


# ---- main ----------------------------------------------------------------

def default_checkpoint(size):
    """Default INT4 checkpoint per size. Tiny uses the per-channel BN-folded
    variant (matches the deployed tiny INT4-o8 export). Other sizes use the
    plain INT4 .pth (load_brevitas_int4 path B: warm-load + merge_bn)."""
    if size == 'tiny':
        return os.path.join(REPO_ROOT, 'finn',
                             'cnn_mnist_tiny_int4_perchan_bnfold.pth')
    return os.path.join(REPO_ROOT, 'finn', f'cnn_mnist_{size}_int4.pth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', required=True,
                    help='CNN size config (tiny, small, medium, deep_3, large)')
    ap.add_argument('--checkpoint', default=None,
                    help='Brevitas INT4 .pth (default: per-size from finn/)')
    ap.add_argument('--mnist-dir',
                    default=os.path.join(REPO_ROOT, 'data', 'MNIST', 'raw'),
                    help='MNIST raw idx directory')
    ap.add_argument('--num-samples', type=int, default=100,
                    help='Calibration sample count (default 100)')
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.size)
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    print(f"=== VTA CNN INT4 shift calibration ===")
    print(f"Size:       {args.size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MNIST dir:  {args.mnist_dir}")
    print(f"Cal samples:{args.num_samples}")
    print()

    print(f"Loading Brevitas INT4 checkpoint...")
    layers, act_scales = load_brevitas_int4(args.checkpoint, args.size)
    n_conv  = sum(1 for L in layers if L['type'] == 'conv')
    n_dense = sum(1 for L in layers if L['type'] == 'dense')
    print(f"  → {len(layers)} layers ({n_conv} conv + {n_dense} dense)")
    for li, L in enumerate(layers):
        if L['type'] == 'conv':
            print(f"    layer {li} (conv): W{tuple(L['W_int'].shape)} "
                  f"pool={L['pool']} pad={L['padding']}")
        else:
            print(f"    layer {li} (dense): W{tuple(L['W_int'].shape)}")
    print(f"  act_scales: {[f'{s:.6f}' for s in act_scales]}")

    print(f"\nLoading MNIST test set...")
    images, labels = load_mnist_test(args.mnist_dir, n=args.num_samples)
    print(f"  → {len(images)} images shape {images.shape}, "
          f"labels balanced check: {np.bincount(labels, minlength=10).tolist()}")

    final_shifts, final_acc, max_abs, init_shifts = calibrate(
        layers, act_scales, images, labels, verbose=True)

    print()
    print('=' * 64)
    print(f"RESULT — size={args.size}")
    print('=' * 64)
    print(f"  max_abs per layer:  {['%.1f' % v for v in max_abs]}")
    print(f"  init shifts (analytical): {init_shifts}")
    print(f"  final shifts (after greedy ±1): {final_shifts}")
    print(f"  Mode G simulated accuracy on {len(images)} cal images: "
          f"{final_acc*100:.2f}% ({int(final_acc * len(images))}/{len(images)})")
    print()
    print(f"To use these shifts in export_vta_cnn_int4_o8.py, hand-edit")
    print(f"derive_shifts() to return {final_shifts} for size='{args.size}'.")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
