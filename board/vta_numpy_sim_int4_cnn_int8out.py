#!/usr/bin/env python3
"""Host-side numpy simulator for VTA INT4-input/INT8-output CNN MNIST.

Models the mixed-precision bitstream: INT4 input/weights, INT8 DMA output.
SHR from int32 to int8 (clip [-128, 127], 256 levels) instead of int4
(clip [-8, 7], 16 levels). Shift values are much smaller.

Supports both:
  - Wide [16,32] no-BN model (per-tensor weight scales, no BN step)
  - Per-channel BN-folded [8,16] model (per-channel weight scales, no BN step
    post-fold, conv has bias from fold)

Usage:
    python vta_numpy_sim_int4_cnn_int8out.py --model wide
    python vta_numpy_sim_int4_cnn_int8out.py --model perchan

Modes A/B/C/D same semantics, but clip bounds change:
  Mode B/D SHR clip: [-128, 127] (int8) instead of [-8, 7] (int4)
  Mode A/C: unchanged (float requant, no SHR)
  Activation clip_max: 15 (A/B) or 7 (C/D) — same as before (input is int4)
"""
import argparse
import gzip
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vta_numpy_sim_int4_cnn import (
    im2col, maxpool2d, adaptive_avg_pool1, conv_gemm,
    load_mnist_images, load_mnist_labels,
)
from vta_numpy_sim_int4_cnn_perchan_mode_G import (
    im2col_with_pad_value, conv_gemm_offset,
)

WIDE_DIR = os.path.expanduser(
    '~/dev/CEN571-final/tvm-v0.12.0/vta_mnist_weights_int4_cnn_nobn_wide')
PERCHAN_DIR = os.path.expanduser(
    '~/dev/CEN571-final/tvm-v0.12.0/vta_mnist_weights_int4_cnn_perchan')

INT8_CLIP_LO = -128
INT8_CLIP_HI = 127


def load_weights_generic(weights_dir):
    meta = json.load(open(os.path.join(weights_dir, 'meta.json')))
    W_conv, w_scale_conv, b_conv = [], [], []
    for layer in meta['conv_layers']:
        i = layer['index']
        W_conv.append(np.load(os.path.join(weights_dir, f"W{i}.npy")))
        ws = np.load(os.path.join(weights_dir, f"w_scale_{i}.npy"))
        w_scale_conv.append(ws if ws.ndim > 0 else float(ws))
        b_conv.append(np.load(os.path.join(weights_dir, f"b{i}.npy")))
    W_dense, w_scale_dense, b_dense = [], [], []
    for layer in meta['dense_layers']:
        i = layer['index']
        W_dense.append(np.load(os.path.join(weights_dir, f"W{i}.npy")))
        ws = np.load(os.path.join(weights_dir, f"w_scale_{i}.npy"))
        w_scale_dense.append(ws if ws.ndim > 0 else float(ws))
        b_dense.append(np.load(os.path.join(weights_dir, f"b{i}.npy")))
    act_scale = []
    for entry in meta['act_scales']:
        act_scale.append(float(np.load(
            os.path.join(weights_dir, f"act_scale_{entry['index']}.npy"))))
    per_channel = any(isinstance(s, np.ndarray) for s in w_scale_conv)
    return (W_conv, w_scale_conv, b_conv, W_dense, w_scale_dense, b_dense,
            act_scale, meta, per_channel)


def derive_board_scales(meta):
    board = [1.0 / 7.0]
    for entry in meta['act_scales'][1:]:
        board.append(float(entry['raw_value']) / 7.0)
    return board


def compute_shift_int8(max_abs):
    """Shift to bring int32 acc into int8 [-128, 127]."""
    if max_abs <= 127:
        return 0
    return int(math.ceil(math.log2(max_abs / 127.0)))


def _to_per_ch(scale, C_out):
    """Ensure scale is a (C_out,) array for broadcasting."""
    if isinstance(scale, np.ndarray) and scale.ndim >= 1:
        return scale.astype(np.float64)
    return np.full(C_out, float(scale), dtype=np.float64)


def calibrate_shifts_int8(cal_images, W_conv, w_scale_conv, b_conv,
                           W_dense, w_scale_dense, b_dense, act_scale,
                           clip_max, per_channel):
    """Calibrate per-tensor shifts targeting int8 [-128, 127]."""
    n_layers = len(W_conv) + len(W_dense)
    max_records = [0.0] * n_layers
    for img in cal_images:
        x_int = np.clip(np.round(img[None, :, :] / act_scale[0]),
                       0, clip_max).astype(np.int32)
        for i, W in enumerate(W_conv):
            acc_chw = conv_gemm(x_int, W, pad=1)
            C_out = W.shape[0]
            cs = _to_per_ch(w_scale_conv[i], C_out) * act_scale[i]
            bias_int = np.round(b_conv[i].astype(np.float64) / cs).astype(np.int32)
            acc_with_bias = acc_chw + bias_int[:, None, None]
            max_records[i] = max(max_records[i], float(np.abs(acc_with_bias).max()))
            # Continue Mode A float forward
            float_acc = acc_chw.astype(np.float64) * cs[:, None, None] + b_conv[i][:, None, None]
            pooled = maxpool2d(np.maximum(float_acc, 0.0), 2, 2)
            x_int = np.clip(np.round(pooled / act_scale[i+1]),
                           0, clip_max).astype(np.int32)
        last_s = act_scale[len(W_conv)]
        x_avg = adaptive_avg_pool1(x_int.astype(np.float64) * last_s)
        x_d = np.clip(np.round(x_avg / last_s), 0, clip_max).astype(np.int32)
        dense_acc = W_dense[0].astype(np.int32) @ x_d
        C_d = W_dense[0].shape[0]
        cs_d = _to_per_ch(w_scale_dense[0], C_d) * last_s
        bias_d_int = np.round(b_dense[0].astype(np.float64) / cs_d).astype(np.int32)
        max_records[-1] = max(max_records[-1], float(np.abs(dense_acc + bias_d_int).max()))
    shifts = [compute_shift_int8(m) for m in max_records]
    return shifts, max_records


def simulate_int8out(images, labels, W_conv, w_scale_conv, b_conv,
                     W_dense, w_scale_dense, b_dense, act_scale,
                     clip_max, use_shift, shifts, per_channel, n_limit=None):
    n = len(images) if n_limit is None else min(n_limit, len(images))
    correct = 0
    for img_idx in range(n):
        x_int = np.clip(np.round(images[img_idx][None, :, :] / act_scale[0]),
                       0, clip_max).astype(np.int32)
        for i, W in enumerate(W_conv):
            acc_chw = conv_gemm(x_int, W, pad=1)
            C_out = W.shape[0]
            cs = _to_per_ch(w_scale_conv[i], C_out) * act_scale[i]

            if use_shift:
                bias_int = np.round(b_conv[i].astype(np.float64) / cs).astype(np.int32)
                acc_chw = acc_chw + bias_int[:, None, None]
                s = shifts[i]
                shifted = acc_chw >> s if s >= 0 else acc_chw << (-s)
                clipped = np.clip(shifted, INT8_CLIP_LO, INT8_CLIP_HI).astype(np.int32)
                dequant = (2.0 ** s) * cs
                float_acc = clipped.astype(np.float64) * dequant[:, None, None]
            else:
                float_acc = (acc_chw.astype(np.float64) * cs[:, None, None]
                            + b_conv[i][:, None, None])

            pooled = maxpool2d(np.maximum(float_acc, 0.0), 2, 2)
            x_int = np.clip(np.round(pooled / act_scale[i+1]),
                           0, clip_max).astype(np.int32)

        last_s = act_scale[len(W_conv)]
        x_avg = adaptive_avg_pool1(x_int.astype(np.float64) * last_s)
        x_d = np.clip(np.round(x_avg / last_s), 0, clip_max).astype(np.int32)
        dense_acc = W_dense[0].astype(np.int32) @ x_d
        C_d = W_dense[0].shape[0]
        cs_d = _to_per_ch(w_scale_dense[0], C_d) * last_s

        if use_shift:
            bias_d_int = np.round(b_dense[0].astype(np.float64) / cs_d).astype(np.int32)
            dense_final = dense_acc + bias_d_int
            pred = int(np.argmax(dense_final))
        else:
            dense_float = (dense_acc.astype(np.float64) * cs_d
                          + b_dense[0].astype(np.float64))
            pred = int(np.argmax(dense_float))

        if pred == labels[img_idx]:
            correct += 1
    return correct, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['wide', 'perchan'])
    args = ap.parse_args()

    weights_dir = WIDE_DIR if args.model == 'wide' else PERCHAN_DIR
    (W_conv, w_scale_conv, b_conv, W_dense, w_scale_dense, b_dense,
     act_scale, meta, per_channel) = load_weights_generic(weights_dir)
    board_act_scale = derive_board_scales(meta)

    label = "Wide [16,32] no-BN" if args.model == 'wide' else "Per-channel BN-fold [8,16]"
    print("=" * 72)
    print(f"VTA INT4-in / INT8-out CNN sim — {label}")
    print("=" * 72)
    print(f"Loaded from: {weights_dir}")
    print(f"Per-channel weights: {per_channel}")
    for i, W in enumerate(W_conv):
        s = w_scale_conv[i]
        sr = f"[{s.min():.4f}, {s.max():.4f}]" if isinstance(s, np.ndarray) else f"{s:.4f}"
        print(f"  Conv{i+1}: W{W.shape}  w_scale {sr}  bias [{b_conv[i].min():+.3f}, {b_conv[i].max():+.3f}]")
    print(f"  Brevitas-native act_scales: {[f'{s:.4f}' for s in act_scale]}")
    print(f"  Board-realistic act_scales: {[f'{s:.4f}' for s in board_act_scale]}")

    mnist_dir = os.path.expanduser(
        '~/dev/CEN571-final/finn-vs-vitisai/finn/data/MNIST/raw')
    test_imgs = load_mnist_images(f'{mnist_dir}/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels(f'{mnist_dir}/t10k-labels-idx1-ubyte.gz')
    train_imgs = load_mnist_images(f'{mnist_dir}/train-images-idx3-ubyte.gz')

    cal = train_imgs[:500]

    print("\n" + "=" * 72)
    print("Shift calibration (500 train imgs, int8 target [-128, 127])")
    print("=" * 72)
    shifts_bn, max_bn = calibrate_shifts_int8(
        cal, W_conv, w_scale_conv, b_conv, W_dense, w_scale_dense, b_dense,
        act_scale, clip_max=15, per_channel=per_channel)
    shifts_board, max_board = calibrate_shifts_int8(
        cal, W_conv, w_scale_conv, b_conv, W_dense, w_scale_dense, b_dense,
        board_act_scale, clip_max=7, per_channel=per_channel)
    print(f"  Brevitas [0,15]: max_abs={[f'{m:.0f}' for m in max_bn]}  shifts={shifts_bn}")
    print(f"  Board [0,7]:     max_abs={[f'{m:.0f}' for m in max_board]}  shifts={shifts_board}")

    print("\n" + "=" * 72)
    print("Evaluation on full 10K MNIST")
    print("=" * 72)
    results = {}
    for tag, scales, clip_max, use_shift, shifts in [
        ('A', act_scale, 15, False, None),
        ('B', act_scale, 15, True, shifts_bn),
        ('C', board_act_scale, 7, False, None),
        ('D', board_act_scale, 7, True, shifts_board),
    ]:
        t0 = time.time()
        c, t = simulate_int8out(test_imgs, test_labels, W_conv, w_scale_conv,
                                b_conv, W_dense, w_scale_dense, b_dense,
                                scales, clip_max=clip_max,
                                use_shift=use_shift, shifts=shifts,
                                per_channel=per_channel)
        results[tag] = 100 * c / t
        print(f"Mode {tag}:  {results[tag]:.2f}% ({c}/{t})  [{time.time()-t0:.1f}s]")

    A, B, C, D = results['A'], results['B'], results['C'], results['D']
    brevitas_acc = meta.get('brevitas_full10K_accuracy',
                            meta.get('brevitas_10img_label_agreement', '?'))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Brevitas:   {brevitas_acc}")
    print(f"  Mode A:     {A:.2f}%")
    print(f"  Mode B:     {B:.2f}%  (SHR to int8, [0,15])")
    print(f"  Mode C:     {C:.2f}%  (float, [0,7])")
    print(f"  Mode D:     {D:.2f}%  (SHR to int8, [0,7])")
    print(f"  A-B gap:    {A-B:+.2f}  (int8 SHR cost at [0,15])")
    print(f"  A-C gap:    {A-C:+.2f}  ([0,15]→[0,7] range cost)")
    print(f"  C-D gap:    {C-D:+.2f}  (int8 SHR cost at [0,7])")
    print(f"  A-D gap:    {A-D:+.2f}  (total)")
    print(f"  Shifts (Brevitas): {shifts_bn}")
    print(f"  Shifts (board):    {shifts_board}")
    print()
    gate = abs(D - A) <= 3.0
    print(f"  GATE (|D-A| ≤ 3 pts): {'PASS' if gate else 'FAIL'}  |D-A|={abs(D-A):.2f}")
    need_g = abs(A - C) >= 3.0
    print(f"  Need Mode G? |A-C|={abs(A-C):.2f}  → "
          f"{'YES (>3 pts)' if need_g else 'NO (<3 pts, range waste negligible)'}")

    # ---- Mode G: zero-point offset + int8 output ----
    if need_g:
        print("\n" + "=" * 72)
        print("Mode G / G_float: zero-point offset (ZP=8) + int8 output")
        print("=" * 72)
        ZP = 8

        zp_corr = []
        for W in W_conv:
            zp_corr.append(ZP * W.astype(np.int32).sum(axis=(1, 2, 3)))
        zp_corr.append(ZP * W_dense[0].astype(np.int32).sum(axis=1))

        # Calibrate Mode G shifts at int8 output
        max_g = [0.0] * (len(W_conv) + 1)
        for img in train_imgs[:500]:
            x_bre = np.clip(np.round(img[None, :, :] / act_scale[0]),
                           0, 15).astype(np.int32)
            x_vta = x_bre - ZP
            for i, W in enumerate(W_conv):
                acc = conv_gemm_offset(x_vta, W, pad=1, pad_value=-ZP)
                C_out = W.shape[0]
                cs = _to_per_ch(w_scale_conv[i], C_out) * act_scale[i]
                bias_int = np.round(b_conv[i].astype(np.float64) / cs).astype(np.int32)
                corrected = acc + (bias_int + zp_corr[i])[:, None, None]
                max_g[i] = max(max_g[i], float(np.abs(corrected).max()))
                float_acc = corrected.astype(np.float64) * cs[:, None, None]
                pooled = maxpool2d(np.maximum(float_acc, 0.0), 2, 2)
                x_bre = np.clip(np.round(pooled / act_scale[i+1]),
                               0, 15).astype(np.int32)
                x_vta = x_bre - ZP
            last_s = act_scale[len(W_conv)]
            x_avg = adaptive_avg_pool1(
                (x_vta + ZP).astype(np.float64) * last_s)
            x_bre_d = np.clip(np.round(x_avg / last_s), 0, 15).astype(np.int32)
            x_vta_d = x_bre_d - ZP
            C_d = W_dense[0].shape[0]
            cs_d = _to_per_ch(w_scale_dense[0], C_d) * last_s
            bias_d_int = np.round(b_dense[0].astype(np.float64) / cs_d).astype(np.int32)
            dense_acc = W_dense[0].astype(np.int32) @ x_vta_d.astype(np.int32)
            max_g[-1] = max(max_g[-1], float(np.abs(dense_acc + bias_d_int + zp_corr[-1]).max()))

        shifts_g = [compute_shift_int8(m) for m in max_g]
        print(f"  G max_abs: {[f'{m:.0f}' for m in max_g]}  shifts: {shifts_g}")

        def run_mode_g(images, labels, use_shift, shifts_g_in, n_limit=None):
            n = len(images) if n_limit is None else min(n_limit, len(images))
            correct = 0
            for idx in range(n):
                x_bre = np.clip(np.round(images[idx][None, :, :] / act_scale[0]),
                               0, 15).astype(np.int32)
                x_vta = x_bre - ZP
                for i, W in enumerate(W_conv):
                    acc = conv_gemm_offset(x_vta, W, pad=1, pad_value=-ZP)
                    C_out = W.shape[0]
                    cs = _to_per_ch(w_scale_conv[i], C_out) * act_scale[i]
                    bias_int = np.round(b_conv[i].astype(np.float64) / cs).astype(np.int32)
                    corrected = acc + (bias_int + zp_corr[i])[:, None, None]
                    if use_shift:
                        s = shifts_g_in[i]
                        shifted = corrected >> s if s >= 0 else corrected << (-s)
                        clipped = np.clip(shifted, INT8_CLIP_LO, INT8_CLIP_HI).astype(np.int32)
                        dequant = (2.0 ** s) * cs
                        float_acc = clipped.astype(np.float64) * dequant[:, None, None]
                    else:
                        float_acc = corrected.astype(np.float64) * cs[:, None, None]
                    pooled = maxpool2d(np.maximum(float_acc, 0.0), 2, 2)
                    x_bre = np.clip(np.round(pooled / act_scale[i+1]),
                                   0, 15).astype(np.int32)
                    x_vta = x_bre - ZP
                last_s = act_scale[len(W_conv)]
                x_avg = adaptive_avg_pool1(
                    (x_vta + ZP).astype(np.float64) * last_s)
                x_bre_d = np.clip(np.round(x_avg / last_s), 0, 15).astype(np.int32)
                x_vta_d = x_bre_d - ZP
                C_d = W_dense[0].shape[0]
                cs_d = _to_per_ch(w_scale_dense[0], C_d) * last_s
                dense_acc = W_dense[0].astype(np.int32) @ x_vta_d.astype(np.int32)
                if use_shift:
                    bias_d_int = np.round(b_dense[0].astype(np.float64) / cs_d).astype(np.int32)
                    pred = int(np.argmax(dense_acc + bias_d_int + zp_corr[-1]))
                else:
                    float_d = (dense_acc + zp_corr[-1]).astype(np.float64) * cs_d + b_dense[0].astype(np.float64)
                    pred = int(np.argmax(float_d))
                if pred == labels[idx]:
                    correct += 1
            return correct, n

        t0 = time.time()
        cg, tg = run_mode_g(test_imgs, test_labels, False, None)
        Gf = 100 * cg / tg
        print(f"  Mode G_float: {Gf:.2f}%  [{time.time()-t0:.1f}s]")

        t0 = time.time()
        cg, tg = run_mode_g(test_imgs, test_labels, True, shifts_g)
        G = 100 * cg / tg
        print(f"  Mode G:       {G:.2f}%  [{time.time()-t0:.1f}s]")

        print(f"\n  G_float vs A: {Gf-A:+.2f}  (offset range recovery)")
        print(f"  G vs G_float: {G-Gf:+.2f}  (int8 SHR cost with offset)")
        print(f"  G vs A:       {G-A:+.2f}  (total with offset + int8 SHR)")
        gate_g = abs(G - A) <= 3.0
        print(f"\n  GATE (|G-A| ≤ 3 pts): {'PASS' if gate_g else 'FAIL'}  |G-A|={abs(G-A):.2f}")


if __name__ == "__main__":
    main()
