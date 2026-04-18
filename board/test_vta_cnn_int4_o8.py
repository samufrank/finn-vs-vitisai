#!/usr/bin/env python3
"""Board-side CNN INT4-input/INT8-output inference via Mode G pipeline.

Standalone test — does NOT modify benchmark.py. Loads the per-channel BN-fold
[8,16] model from the export directory, runs inference on MNIST test images,
reports accuracy and timing.

Pipeline per conv layer:
  CPU: im2col with pad_value=-8 (offset-encoded activations)
  CPU: pack int4 nibbles for VTA DMA input
  VTA: GEMM + corrected_bias_int32 + SHR + CLIP[-128,127] → int8 output
  CPU: read int8 output (one byte per value, no nibble unpacking needed)
  CPU: per-channel dequant to float
  CPU: ReLU, MaxPool
  CPU: requant to [0,15], subtract zero_point=8 for next layer

Dense (last):
  VTA: GEMM + SHR + CLIP[-128,127] → int8 output
  CPU: per-channel dequant + corrected float bias + argmax

Usage (on board):
    cd /home/xilinx
    sudo LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH \\
         PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH \\
         python3 test_vta_cnn_int4_o8.py \\
             --model-dir /home/xilinx/models/vta/cnn_mnist_int4_o8_perchan/ \\
             --n-images 100
"""
import argparse
import ctypes
import gc
import gzip
import json
import math
import os
import struct
import sys
import time

import numpy as np


BITSTREAM = '1x16_i4w4o8a32_15_14_17_17.bit'
ZERO_POINT = 8


# ---- INT4 nibble packing (input only — output is int8, no packing) ----

def pack_int4_for_vta(vals_int8):
    vals = np.asarray(vals_int8, dtype=np.int8)
    flat = vals.flatten()
    n = len(flat)
    lo = flat[0::2].view(np.uint8) & 0xF
    hi = flat[1::2].view(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[:n // 2] = packed
    return out.reshape(vals.shape)


# ---- Im2col with offset-aware padding ----

def im2col(x_chw, kH, kW, pad, pad_value=0):
    C, H, W = x_chw.shape
    if pad > 0:
        x_pad = np.full((C, H + 2*pad, W + 2*pad), pad_value, dtype=x_chw.dtype)
        x_pad[:, pad:pad+H, pad:pad+W] = x_chw
    else:
        x_pad = x_chw
    out_H = H + 2*pad - kH + 1
    out_W = W + 2*pad - kW + 1
    cols = np.empty((out_H * out_W, kH * kW * C), dtype=x_chw.dtype)
    idx = 0
    for i in range(out_H):
        for j in range(out_W):
            cols[idx] = x_pad[:, i:i+kH, j:j+kW].transpose(1, 2, 0).reshape(-1)
            idx += 1
    return cols


def maxpool2d(x_chw, kernel, stride):
    C, H, W = x_chw.shape
    oH = (H - kernel) // stride + 1
    oW = (W - kernel) // stride + 1
    shape = (C, oH, oW, kernel, kernel)
    strides = (x_chw.strides[0], x_chw.strides[1]*stride, x_chw.strides[2]*stride,
               x_chw.strides[1], x_chw.strides[2])
    return np.lib.stride_tricks.as_strided(x_chw, shape=shape,
                                            strides=strides, writeable=False).max(axis=(3,4))


# ---- MNIST loader ----

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- Board setup ----

def setup_board():
    stale = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale): os.remove(stale)
    except: pass

    for p in [f'/root/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
              f'/home/xilinx/.vta_cache/ultra96/0_0_2/{BITSTREAM}']:
        if os.path.exists(p):
            print(f"[bitstream] {p}")
            try:
                from pynq import Overlay
                Overlay(p)
            except Exception as e:
                print(f"  Overlay: {e} (continuing)")
            break
    else:
        print(f"WARNING: bitstream {BITSTREAM} not found")

    for p in ['/home/xilinx/tvm-src/build/libvta.so']:
        if os.path.exists(p):
            print(f"[libvta] {p}")
            ctypes.CDLL(p, ctypes.RTLD_GLOBAL)
            break


# ---- Inference ----

def infer_one(img_hw, config, mods, W_nds, D_nds, C_nds, A_nds, act_scale):
    """Single-image CNN inference via Mode G + int8 output.

    img_hw: (28, 28) float32 in [0, 1].
    Returns: predicted class (int).
    All layers (conv + dense) use 4-arg VTA modules with corrected int32 bias
    inside VTA. No CPU-side bias addition.
    """
    BLK = config['BLOCK_IN']
    layers = config['layers']

    # Input: quantize to Brevitas [0,15], offset to VTA [-8,7]
    x_bre = np.clip(np.round(img_hw[None, :, :] / act_scale[0]),
                   0, 15).astype(np.int32)
    x_vta = (x_bre - ZERO_POINT).astype(np.int8)

    for ci in range(len(layers)):
        L = layers[ci]
        if L['type'] != 'conv':
            break  # reached dense

        o_tile = L['o_tile']
        n_chunks = L['n_chunks']
        n_tiles = L['n']
        m_tiles = L['m']
        C_out_valid = L['C_out_valid']

        # im2col on offset-encoded input, pad with -ZP
        patches = im2col(x_vta, 3, 3, pad=1, pad_value=-ZERO_POINT)
        total_in = n_tiles * BLK
        if patches.shape[1] < total_in:
            patches = np.pad(patches, ((0, 0), (0, total_in - patches.shape[1])))
        spatial = patches.shape[0]

        # Process in o-tile chunks
        full_out_int8 = np.zeros((spatial, BLK), dtype=np.int8)
        for ch in range(n_chunks):
            st = ch * o_tile
            en = min(st + o_tile, spatial)
            ao = en - st
            if ao <= 0:
                break

            # Shape A: (o_tile, n, 1, BLK) int4 packed
            a_chunk = patches[st:en].reshape(ao, n_tiles, 1, BLK).astype(np.int8)
            if ao < o_tile:
                a_chunk = np.pad(a_chunk, ((0, o_tile - ao), (0, 0), (0, 0), (0, 0)))
            a_packed = pack_int4_for_vta(a_chunk)

            # Copy to pre-allocated VTA buffers
            A_nds[ci].copyfrom(a_packed)
            C_nds[ci].copyfrom(np.zeros((o_tile, m_tiles, 1, BLK), dtype=np.int8))

            # 4-arg VTA call: GEMM + corrected_bias + SHR + CLIP[-128,127] → int8
            mods[ci](A_nds[ci], W_nds[ci], D_nds[ci], C_nds[ci])

            # Read int8 output directly (no nibble unpacking!)
            chunk_out = C_nds[ci].numpy()[:ao, 0, 0, :]
            full_out_int8[st:en] = chunk_out

        # CPU: per-channel dequant → ReLU → MaxPool → requant → offset
        H = int(math.sqrt(spatial))
        out_chw = full_out_int8[:, :C_out_valid].reshape(H, H, C_out_valid).transpose(2, 0, 1)
        del full_out_int8, patches  # free large intermediates early
        shift = L['shift']
        w_scale = np.array(L['w_scale'], dtype=np.float64)
        cs = w_scale * act_scale[ci] * (2.0 ** shift)
        float_out = out_chw.astype(np.float64) * cs[:, None, None]
        del out_chw
        post_relu = np.maximum(float_out, 0.0)
        del float_out
        pooled = maxpool2d(post_relu, 2, 2)
        del post_relu

        # Requant to Brevitas [0,15], then offset to VTA [-8,7]
        x_bre = np.clip(np.round(pooled / act_scale[ci + 1]),
                       0, 15).astype(np.int32)
        del pooled
        x_vta = (x_bre - ZERO_POINT).astype(np.int8)
        del x_bre

    # AdaptiveAvgPool: dequant to float, average, requant
    last_scale = act_scale[len([l for l in layers if l['type'] == 'conv'])]
    x_float = (x_vta.astype(np.int32) + ZERO_POINT).astype(np.float64) * last_scale
    x_avg = x_float.mean(axis=(1, 2))

    # Dense input: requant + offset
    x_d_bre = np.clip(np.round(x_avg / last_scale), 0, 15).astype(np.int32)
    x_d_vta = (x_d_bre - ZERO_POINT).astype(np.int8)

    # Pad to BLK, pack int4
    x_d_padded = np.zeros(BLK, dtype=np.int8)
    x_d_padded[:len(x_d_vta)] = x_d_vta
    a_d = pack_int4_for_vta(x_d_padded.reshape(1, 1, 1, BLK))

    di = len([l for l in layers if l['type'] == 'conv'])  # dense layer index
    dl = layers[di]
    A_nds[di].copyfrom(a_d)
    C_nds[di].copyfrom(np.zeros((1, dl['m'], 1, BLK), dtype=np.int8))

    # 4-arg VTA call: GEMM + corrected_bias + SHR + CLIP → int8
    # (corrected_bias includes zp correction inside VTA, so int8 output
    # represents the real logit without zp offset — no CPU bias needed)
    mods[di](A_nds[di], W_nds[di], D_nds[di], C_nds[di])
    dense_int8 = C_nds[di].numpy()[0, 0, 0, :]

    # CPU: per-channel dequant → argmax (no CPU bias — it's in VTA)
    C_d_valid = dl['C_out_valid']
    w_scale_d = np.array(dl['w_scale'], dtype=np.float64)
    cs_d = w_scale_d * last_scale * (2.0 ** dl['shift'])
    dense_float = dense_int8[:C_d_valid].astype(np.float64) * cs_d
    return int(np.argmax(dense_float))


model_dir_global = None  # set in main


def main():
    global model_dir_global
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--mnist-dir', default='/home/xilinx/mnist_data')
    ap.add_argument('--n-images', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=3)
    args = ap.parse_args()
    model_dir_global = args.model_dir

    setup_board()

    import tvm
    import tvm.runtime
    ctx = tvm.device("ext_dev", 0)

    # Load config
    config = json.load(open(os.path.join(args.model_dir, 'config.json')))
    act_scale = config['act_scales_brevitas']
    BLK = config['BLOCK_IN']

    # Load modules and weights
    mods, W_nds, D_nds, C_nds, A_nds = [], [], [], [], []
    for L in config['layers']:
        # Link .o → .so if needed
        o_path = os.path.join(args.model_dir, L['module_file'])
        so_path = o_path.replace('.o', '.so')
        if not os.path.exists(so_path):
            import subprocess
            subprocess.check_call([
                'gcc', '-shared', '-o', so_path, o_path,
                '-L/home/xilinx/tvm-src/build', '-ltvm_runtime'])
        mod = tvm.runtime.load_module(so_path)
        mods.append(mod)

        # Pre-allocate VTA buffers
        W = np.load(os.path.join(args.model_dir, L['W_file'])).astype(np.int8)
        W_packed = pack_int4_for_vta(W)
        W_nds.append(tvm.nd.array(W_packed, ctx))

        o = L['o_tile']
        A_nds.append(tvm.nd.array(
            np.zeros((o, L['n'], 1, BLK), dtype=np.int8), ctx))
        C_nds.append(tvm.nd.array(
            np.zeros((o, L['m'], 1, BLK), dtype=np.int8), ctx))

        if L['n_args'] == 4:
            bias = np.load(os.path.join(args.model_dir, L['bias_file']))
            bias_bc = np.ascontiguousarray(
                np.broadcast_to(
                    bias.reshape(1, L['m'], 1, BLK),
                    (o, L['m'], 1, BLK)),
                dtype=np.int32)
            D_nds.append(tvm.nd.array(bias_bc, ctx))
        else:
            D_nds.append(None)

    # MNIST
    if not os.path.exists(args.mnist_dir):
        args.mnist_dir = '/home/xilinx/mnist_data'
    imgs = load_mnist_images(os.path.join(args.mnist_dir, 't10k-images-idx3-ubyte.gz'))
    labels = load_mnist_labels(os.path.join(args.mnist_dir, 't10k-labels-idx1-ubyte.gz'))

    n = min(args.n_images, len(imgs))
    print(f"\nRunning {n} images (warmup={args.warmup})")

    # Warmup
    for _ in range(args.warmup):
        infer_one(imgs[0], config, mods, W_nds, D_nds, C_nds, A_nds, act_scale)

    # Inference
    correct = 0
    preds = []
    t0 = time.time()
    for i in range(n):
        pred = infer_one(imgs[i], config, mods, W_nds, D_nds, C_nds, A_nds,
                         act_scale)
        preds.append(pred)
        if pred == labels[i]:
            correct += 1
        if n <= 20 or (i + 1) % 500 == 0:
            if n <= 20:
                print(f"  [{i}] pred={pred} label={labels[i]} "
                      f"{'✓' if pred == labels[i] else '✗'}")
            else:
                gc.collect()
                print(f"  [{i+1}/{n}] acc so far: {100*correct/(i+1):.1f}%")
    elapsed = time.time() - t0

    acc = 100 * correct / n
    fps = n / elapsed
    lat = elapsed / n * 1000

    print(f"\nResults ({n} images):")
    print(f"  Accuracy:   {acc:.2f}% ({correct}/{n})")
    print(f"  Throughput: {fps:.1f} FPS")
    print(f"  Latency:    {lat:.1f} ms/image")
    print(f"  Elapsed:    {elapsed:.2f}s")

    results = {
        'accuracy': acc, 'correct': correct, 'total': n,
        'throughput_fps': fps, 'latency_ms': lat,
        'model_dir': args.model_dir,
        'bitstream': BITSTREAM,
        'pipeline': 'Mode G (zp=8) + int8 output',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
    }
    out_path = os.path.join(args.model_dir, 'board_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
