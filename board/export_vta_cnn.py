#!/usr/bin/env python3
"""Export pre-compiled VTA CNN modules for board-side inference.

Loads a Brevitas CNN checkpoint, folds BatchNorm into conv weights,
compiles VTA GEMM+shift+clip modules (cross-compiled for aarch64),
and saves everything as a self-contained model directory.

CNN inference on VTA uses im2col: each conv layer is converted to a
matrix multiply, then executed via the same GEMM+shift+clip schedule
as the MLP. MaxPool, GlobalAvgPool, and ReLU run on CPU between
VTA GEMM calls.

Architecture: CNN tiny [8, 16] on MNIST (28x28, 1 channel)
  Conv1(1->8, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
  Conv2(8->16, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
  GlobalAvgPool(7x7) -> Dense(16->10)

Usage (from Ubuntu host):
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 export_vta_cnn.py \
            --checkpoint ../finn-vs-vitisai/finn/cnn_mnist_tiny.pth \
            --output-dir ./vta_export/cnn_mnist_tiny/

Then copy to board:
    scp -r ./vta_export/cnn_mnist_tiny/ xilinx@192.168.3.1:/home/xilinx/models/vta/cnn_mnist_tiny/

Date: April 1, 2026
"""
import numpy as np
import tvm
from tvm import te
from tvm.contrib import utils
import vta
import json
import math
import os
import sys
import argparse
import struct
import gzip
import torch


# ---- MNIST loading ----

MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]

def download_mnist(data_dir='./mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    filenames = {
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }
    paths = {}
    for key, filename in filenames.items():
        fname = os.path.join(data_dir, filename)
        if not os.path.exists(fname):
            from urllib.request import urlretrieve
            for mirror in MNIST_MIRRORS:
                try:
                    print(f"  Downloading {key} from {mirror}...", end=" ", flush=True)
                    urlretrieve(mirror + filename, fname)
                    print("OK")
                    break
                except Exception as e:
                    print(f"failed ({e})")
            else:
                raise RuntimeError(f"Could not download {filename}")
        paths[key] = fname
    return paths

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- BN folding ----

def fold_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into preceding Conv2d weights.

    Conv: y = W * x (no bias in Brevitas conv)
    BN:   z = gamma * (y - mean) / sqrt(var + eps) + beta

    Folded: z = W_folded * x + b_folded
      W_folded = gamma / sqrt(var + eps) * W   (per output channel)
      b_folded = -gamma * mean / sqrt(var + eps) + beta
    """
    # conv_weight: (C_out, C_in, kH, kW)
    C_out = conv_weight.shape[0]
    scale = bn_weight / np.sqrt(bn_var + eps)  # (C_out,)

    # Scale each output channel's conv filter
    W_folded = conv_weight * scale.reshape(C_out, 1, 1, 1)
    b_folded = -bn_mean * scale + bn_bias

    return W_folded, b_folded


# ---- im2col ----

def im2col(x, kH, kW, pad=0, stride=1):
    """Extract sliding window patches.

    x: (H, W, C) -> output: (H_out * W_out, kH * kW * C)
    """
    H, W, C = x.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    if pad > 0:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    patches = np.zeros((H_out * W_out, kH * kW * C), dtype=x.dtype)
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+kH, j*stride:j*stride+kW, :]
            patches[idx] = patch.flatten()
            idx += 1
    return patches, H_out, W_out


def maxpool2d(x, pool_size=2):
    """Max pooling on (H, W, C) tensor."""
    H, W, C = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    out = np.zeros((H_out, W_out, C), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            out[i, j] = x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size].max(axis=(0, 1))
    return out


# ---- VTA helpers ----

def tile_weights_2d(W_flat_int8, env):
    """Tile a 2D weight matrix (out_f, in_f) for VTA.
    Both dims must be multiples of BLOCK_OUT and BLOCK_IN respectively.
    """
    out_f, in_f = W_flat_int8.shape
    m = out_f // env.BLOCK_OUT
    n = in_f // env.BLOCK_IN
    return W_flat_int8.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)


def pad_to_block(x, block_size, axis):
    """Pad a dimension to a multiple of block_size."""
    current = x.shape[axis]
    if current % block_size == 0:
        return x, current
    pad_amount = block_size - (current % block_size)
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (0, pad_amount)
    return np.pad(x, pad_widths, mode='constant'), current


def compute_shift_bits(activations_int8, W_int8):
    """Determine right-shift from calibration data.
    activations_int8: (N, in_f), W_int8: (out_f, in_f)
    """
    acc = activations_int8.astype(np.int32) @ W_int8.T.astype(np.int32)
    max_abs = np.max(np.abs(acc))
    if max_abs <= 127:
        return 0
    return int(math.ceil(math.log2(max_abs / 127.0)))


# ---- VTA module compilation ----

def compile_gemm_with_shift(env, o, n, m, shift_bits):
    """Compile VTA GEMM + ALU shift + clip. Returns TVM Module."""
    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
    A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
    B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf")

    ko = te.reduce_axis((0, n), name="ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
    C_buf = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda bo, co, bi, ci: te.sum(
            A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
            B_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki]),
        name="C_buf")

    shr_const = tvm.tir.const(shift_bits, env.acc_dtype)
    C_shr = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_buf(*i) >> shr_const, name="C_shr")

    C_clip_hi = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.min(C_shr(*i), tvm.tir.const(127, env.acc_dtype)),
        name="C_clip_hi")

    C_clip_lo = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.max(C_clip_hi(*i), tvm.tir.const(-128, env.acc_dtype)),
        name="C_clip_lo")

    C = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_clip_lo(*i).astype(env.inp_dtype), name="C")

    s = te.create_schedule(C.op)
    s[A_buf].set_scope(env.inp_scope)
    s[B_buf].set_scope(env.wgt_scope)
    s[C_buf].set_scope(env.acc_scope)
    s[C_shr].set_scope(env.acc_scope)
    s[C_clip_hi].set_scope(env.acc_scope)
    s[C_clip_lo].set_scope(env.acc_scope)

    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
        s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
    s[C_clip_hi].pragma(s[C_clip_hi].op.axis[0], env.alu)
    s[C_clip_lo].pragma(s[C_clip_lo].op.axis[0], env.alu)
    s[C].pragma(s[C].op.axis[0], env.dma_copy)

    mod = vta.build(s, [A, B, C],
                    tvm.target.vta(),
                    tvm.target.arm_cpu("ultra96"),
                    name="my_gemm")
    return mod


# ---- Weight extraction and preparation ----

def load_brevitas_cnn(checkpoint_path):
    """Load Brevitas CNN checkpoint, fold BN, return layer list.

    Returns list of dicts with keys:
      'type': 'conv' or 'dense'
      'W': weight array (for conv: flattened to 2D after im2col reshape)
      'b': bias array
      'kernel_size', 'padding', 'in_channels', 'out_channels' (conv only)
    """
    sd = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    layers = []

    # Conv1: features.0 (conv) + features.1 (BN)
    conv1_w = sd['features.0.weight'].numpy()  # (8, 1, 3, 3)
    bn1_w = sd['features.1.weight'].numpy()
    bn1_b = sd['features.1.bias'].numpy()
    bn1_m = sd['features.1.running_mean'].numpy()
    bn1_v = sd['features.1.running_var'].numpy()
    W1_folded, b1_folded = fold_bn_into_conv(conv1_w, bn1_w, bn1_b, bn1_m, bn1_v)
    # Reshape conv weight to 2D for im2col GEMM: (C_out, kH*kW*C_in)
    # im2col with HWC input produces patches in (kH, kW, C_in) order,
    # so we must transpose weight from (C_out, C_in, kH, kW) to (C_out, kH, kW, C_in)
    C_out1, C_in1, kH1, kW1 = W1_folded.shape
    layers.append({
        'type': 'conv',
        'W': W1_folded.transpose(0, 2, 3, 1).reshape(C_out1, -1),  # (8, 9)
        'b': b1_folded,                       # (8,)
        'kernel_size': kH1,
        'padding': 1,
        'in_channels': C_in1,
        'out_channels': C_out1,
        'pool': 2,
    })

    # Conv2: features.4 (conv) + features.5 (BN)
    conv2_w = sd['features.4.weight'].numpy()  # (16, 8, 3, 3)
    bn2_w = sd['features.5.weight'].numpy()
    bn2_b = sd['features.5.bias'].numpy()
    bn2_m = sd['features.5.running_mean'].numpy()
    bn2_v = sd['features.5.running_var'].numpy()
    W2_folded, b2_folded = fold_bn_into_conv(conv2_w, bn2_w, bn2_b, bn2_m, bn2_v)
    C_out2, C_in2, kH2, kW2 = W2_folded.shape
    layers.append({
        'type': 'conv',
        'W': W2_folded.transpose(0, 2, 3, 1).reshape(C_out2, -1),  # (16, 72)
        'b': b2_folded,                       # (16,)
        'kernel_size': kH2,
        'padding': 1,
        'in_channels': C_in2,
        'out_channels': C_out2,
        'pool': 2,
    })

    # Dense: classifier.1
    W_cls = sd['classifier.1.weight'].numpy()  # (10, 16)
    b_cls = sd['classifier.1.bias'].numpy()    # (10,)
    layers.append({
        'type': 'dense',
        'W': W_cls,
        'b': b_cls,
    })

    return layers


def prepare_vta_layers(layers, env):
    """Pad weights to VTA alignment, quantize, tile.

    For conv layers, im2col flattens the kernel to (C_out, kH*kW*C_in).
    We pad the inner dim to BLOCK_IN multiples and outer to BLOCK_OUT multiples.

    Returns list of dicts with VTA-ready data.
    """
    vta_layers = []
    for i, layer in enumerate(layers):
        W = layer['W'].astype(np.float32)
        b = layer['b'].astype(np.float32)
        out_f, in_f = W.shape

        # Pad output dim to BLOCK_OUT multiple
        real_out = out_f
        if out_f % env.BLOCK_OUT != 0:
            pad_out = env.BLOCK_OUT - (out_f % env.BLOCK_OUT)
            W = np.pad(W, ((0, pad_out), (0, 0)), mode='constant')
            b = np.pad(b, (0, pad_out), mode='constant')
            out_f = W.shape[0]

        # Pad input dim to BLOCK_IN multiple
        real_in = in_f
        if in_f % env.BLOCK_IN != 0:
            pad_in = env.BLOCK_IN - (in_f % env.BLOCK_IN)
            W = np.pad(W, ((0, 0), (0, pad_in)), mode='constant')
            in_f = W.shape[1]

        # Quantize weights
        w_max = np.max(np.abs(W))
        w_scale = w_max / 127.0 if w_max > 0 else 1e-10
        W_int8 = np.clip(np.round(W / w_scale), -128, 127).astype(np.int8)
        W_tiled = tile_weights_2d(W_int8, env)

        n_tiles = in_f // env.BLOCK_IN
        m_tiles = out_f // env.BLOCK_OUT

        info = {
            'W_int8': W_int8,
            'W_tiled': W_tiled,
            'b_float': b,
            'w_scale': float(w_scale),
            'in_f': in_f,       # padded
            'out_f': out_f,     # padded
            'real_in': real_in,
            'real_out': real_out,
            'n_tiles': n_tiles,
            'm_tiles': m_tiles,
            'layer_type': layer['type'],
        }

        if layer['type'] == 'conv':
            info['kernel_size'] = layer['kernel_size']
            info['padding'] = layer['padding']
            info['in_channels'] = layer['in_channels']
            info['out_channels'] = layer['out_channels']
            info['pool'] = layer.get('pool', 0)

        vta_layers.append(info)
        print(f"  Layer {i} ({layer['type']}): {real_in}->{real_out} "
              f"(padded {in_f}->{out_f}), "
              f"w_scale={w_scale:.6f}, tiles n={n_tiles} m={m_tiles}")

    return vta_layers


# ---- Calibration ----

def calibrate_cnn(vta_layers, cal_images, env):
    """Run calibration images through CNN pipeline to determine shift amounts.

    cal_images: (N, 28, 28) float32 [0, 1]
    Returns list of shift amounts and global_x_scale.
    """
    N = len(cal_images)
    shift_amounts = []

    # Quantize input images
    global_x_scale = np.mean([np.max(np.abs(img)) / 127.0 for img in cal_images])
    if global_x_scale < 1e-10:
        global_x_scale = 1e-10

    # Process each calibration image through the pipeline
    # Accumulate max accumulator values per layer across all images
    layer_max_acc = [0.0 for _ in vta_layers]

    for img_idx in range(N):
        img = cal_images[img_idx]  # (28, 28)

        # Quantize input
        x_s = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        current_scale = x_s

        # Process through layers
        h_float = img  # current activation (spatial)

        for i, vl in enumerate(vta_layers):
            if vl['layer_type'] == 'conv':
                # im2col
                if h_float.ndim == 2:
                    # First layer: (H, W) -> (H, W, 1)
                    h_spatial = h_float[:, :, np.newaxis]
                else:
                    h_spatial = h_float  # (H, W, C)

                patches, H_out, W_out = im2col(
                    h_spatial, vl['kernel_size'], vl['kernel_size'],
                    pad=vl['padding'])
                # patches: (H_out*W_out, kH*kW*C_in)

                # Pad patches to BLOCK_IN alignment
                real_patch_dim = patches.shape[1]
                if real_patch_dim < vl['in_f']:
                    patches = np.pad(patches, ((0, 0), (0, vl['in_f'] - real_patch_dim)),
                                     mode='constant')

                # Quantize patches
                p_int8 = np.clip(np.round(patches / current_scale), -128, 127).astype(np.int8)

                # Compute accumulator range
                acc = p_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                max_abs = np.max(np.abs(acc))
                if max_abs > layer_max_acc[i]:
                    layer_max_acc[i] = max_abs

                # Simulate VTA: shift + clip
                shift = int(math.ceil(math.log2(max_abs / 127.0))) if max_abs > 127 else 0
                shifted = acc >> shift
                clipped = np.clip(shifted, -128, 127).astype(np.int8)

                # Dequantize
                combined = current_scale * vl['w_scale'] * (2 ** shift)
                y_float = clipped.astype(np.float32) * combined + vl['b_float'][:vl['out_f']]

                # ReLU
                y_float = np.maximum(y_float, 0)

                # Reshape to spatial and take real output channels
                y_spatial = y_float[:, :vl['real_out']].reshape(H_out, W_out, vl['real_out'])

                # MaxPool
                if vl.get('pool', 0) > 0:
                    y_spatial = maxpool2d(y_spatial, vl['pool'])

                h_float = y_spatial
                next_scale = np.max(np.abs(h_float)) / 127.0
                current_scale = max(next_scale, 1e-10)

            elif vl['layer_type'] == 'dense':
                # GlobalAvgPool: (H, W, C) -> (C,)
                h_vec = h_float.mean(axis=(0, 1))

                # Pad to BLOCK_IN alignment
                if len(h_vec) < vl['in_f']:
                    h_vec = np.pad(h_vec, (0, vl['in_f'] - len(h_vec)), mode='constant')

                h_int8 = np.clip(np.round(h_vec / current_scale), -128, 127).astype(np.int8)

                acc = h_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                max_abs = np.max(np.abs(acc))
                if max_abs > layer_max_acc[i]:
                    layer_max_acc[i] = max_abs

    # Compute shift amounts from accumulated max values
    for i, max_acc in enumerate(layer_max_acc):
        if max_acc <= 127:
            shift = 0
        else:
            shift = int(math.ceil(math.log2(max_acc / 127.0)))
        shift_amounts.append(shift)
        print(f"  Layer {i}: max_acc={max_acc:.0f}, shift={shift}")

    return shift_amounts, global_x_scale


# ---- CPU-side inference verification ----

def verify_cnn(vta_layers, shift_amounts, test_images, test_labels, env, num_verify=100):
    """Run CPU-side VTA-equivalent inference to verify accuracy before compilation."""
    correct = 0
    for img_idx in range(min(num_verify, len(test_labels))):
        img = test_images[img_idx]
        label = test_labels[img_idx]

        x_s = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        current_scale = x_s
        h_float = img

        for i, (vl, shift) in enumerate(zip(vta_layers, shift_amounts)):
            if vl['layer_type'] == 'conv':
                if h_float.ndim == 2:
                    h_spatial = h_float[:, :, np.newaxis]
                else:
                    h_spatial = h_float

                patches, H_out, W_out = im2col(
                    h_spatial, vl['kernel_size'], vl['kernel_size'],
                    pad=vl['padding'])

                real_patch_dim = patches.shape[1]
                if real_patch_dim < vl['in_f']:
                    patches = np.pad(patches, ((0, 0), (0, vl['in_f'] - real_patch_dim)),
                                     mode='constant')

                p_int8 = np.clip(np.round(patches / current_scale), -128, 127).astype(np.int8)

                # VTA-equivalent: GEMM + shift + clip (truncating, not saturating)
                acc = p_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                shifted = acc >> shift
                clipped = shifted.astype(np.int8)  # VTA truncates

                combined = current_scale * vl['w_scale'] * (2 ** shift)
                y_float = clipped.astype(np.float32) * combined + vl['b_float'][:vl['out_f']]

                y_float = np.maximum(y_float, 0)
                y_spatial = y_float[:, :vl['real_out']].reshape(H_out, W_out, vl['real_out'])

                if vl.get('pool', 0) > 0:
                    y_spatial = maxpool2d(y_spatial, vl['pool'])

                h_float = y_spatial
                next_scale = np.max(np.abs(h_float)) / 127.0
                current_scale = max(next_scale, 1e-10)

            elif vl['layer_type'] == 'dense':
                h_vec = h_float.mean(axis=(0, 1))
                if len(h_vec) < vl['in_f']:
                    h_vec = np.pad(h_vec, (0, vl['in_f'] - len(h_vec)), mode='constant')

                h_int8 = np.clip(np.round(h_vec / current_scale), -128, 127).astype(np.int8)
                acc = h_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                shifted = acc >> shift
                clipped = shifted.astype(np.int8)

                combined = current_scale * vl['w_scale'] * (2 ** shift)
                y_float = clipped.astype(np.float32) * combined + vl['b_float'][:vl['out_f']]

                pred = np.argmax(y_float[:vl['real_out']])
                if pred == label:
                    correct += 1

    acc = correct / min(num_verify, len(test_labels))
    return acc


def main():
    parser = argparse.ArgumentParser(description='Export VTA CNN model for board-side inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Brevitas CNN checkpoint (.pth)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for compiled model')
    parser.add_argument('--mnist-dir', default='./mnist_data',
                        help='MNIST data directory')
    parser.add_argument('--cal-samples', type=int, default=200,
                        help='Number of calibration samples')
    parser.add_argument('--verify-samples', type=int, default=500,
                        help='Number of verification samples (0 to skip)')
    args = parser.parse_args()

    env = vta.get_env()
    print(f"VTA env: TARGET={env.TARGET}, BLOCK_IN={env.BLOCK_IN}, BLOCK_OUT={env.BLOCK_OUT}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load and prepare weights ----
    print(f"\nLoading Brevitas checkpoint: {args.checkpoint}")
    raw_layers = load_brevitas_cnn(args.checkpoint)
    for i, l in enumerate(raw_layers):
        print(f"  Layer {i} ({l['type']}): W={l['W'].shape}, b={l['b'].shape}")

    print(f"\nPreparing VTA layers (pad + quantize + tile)...")
    vta_layers = prepare_vta_layers(raw_layers, env)

    # ---- Calibrate shift amounts ----
    print(f"\nLoading MNIST for calibration...")
    mnist = download_mnist(args.mnist_dir)
    test_images = load_mnist_images(mnist['test_images'])  # (10000, 28, 28)
    test_labels = load_mnist_labels(mnist['test_labels'])

    print(f"\nCalibrating shift amounts ({args.cal_samples} samples)...")
    shift_amounts, global_x_scale = calibrate_cnn(
        vta_layers, test_images[:args.cal_samples], env)

    # ---- CPU-side verification ----
    if args.verify_samples > 0:
        print(f"\nCPU-side verification ({args.verify_samples} samples)...")
        acc = verify_cnn(vta_layers, shift_amounts, test_images, test_labels, env,
                         num_verify=args.verify_samples)
        print(f"  VTA-equivalent accuracy: {acc:.4f} ({int(acc * args.verify_samples)}/{args.verify_samples})")
        if acc < 0.85:
            print("  WARNING: Accuracy below 85%. Check weight folding and quantization.")

    # ---- Compile VTA modules ----
    # VTA hardware limitation: when n_tiles > 1, the maximum o dimension is ~96.
    # For larger o (e.g. conv2 with 196 output pixels), we tile o into chunks
    # and call the module multiple times at inference. The module is compiled
    # with o_tile, and the inference code loops over o_total/o_tile chunks.
    MAX_O_WHEN_N_GT1 = 64  # safe margin below empirical ~96 limit

    print(f"\nCompiling VTA modules...")
    module_filenames = []
    for i, vl in enumerate(vta_layers):
        shift = shift_amounts[i]

        if vl['layer_type'] == 'conv':
            if i == 0:
                o_total = 28 * 28  # Conv1: 28x28 output
            elif i == 1:
                o_total = 14 * 14  # Conv2: 14x14 input after maxpool
            else:
                raise ValueError(f"Unexpected conv layer index {i}")
        else:
            o_total = 1  # Dense layer

        n_t = vl['n_tiles']
        m_t = vl['m_tiles']

        # Determine o_tile: tile o when n>1 and o exceeds hardware limit
        if n_t > 1 and o_total > MAX_O_WHEN_N_GT1:
            # Find largest divisor of o_total that's <= MAX_O_WHEN_N_GT1
            o_tile = None
            for candidate in range(MAX_O_WHEN_N_GT1, 0, -1):
                if o_total % candidate == 0:
                    o_tile = candidate
                    break
            assert o_tile is not None
            n_chunks = o_total // o_tile
        else:
            o_tile = o_total
            n_chunks = 1

        fname = f"layer{i}_o{o_tile}_n{n_t}_m{m_t}_s{shift}.o"
        print(f"  Layer {i} ({vl['layer_type']}, o_total={o_total}, o_tile={o_tile}, "
              f"chunks={n_chunks}, n={n_t}, m={m_t}, shift={shift})...", end=" ", flush=True)

        mod = compile_gemm_with_shift(env, o_tile, n_t, m_t, shift)
        out_path = os.path.join(args.output_dir, fname)
        mod.save(out_path)
        module_filenames.append(fname)
        vl['o_total'] = o_total
        vl['o_tile'] = o_tile
        vl['n_chunks'] = n_chunks
        print(f"OK -> {fname}")

    # ---- Save weights ----
    print(f"\nSaving weights...")
    for i, vl in enumerate(vta_layers):
        np.save(os.path.join(args.output_dir, f'W{i}_tiled.npy'), vl['W_tiled'])
        np.save(os.path.join(args.output_dir, f'b{i}.npy'), vl['b_float'])
        print(f"  W{i}_tiled.npy: {vl['W_tiled'].shape}, b{i}.npy: {vl['b_float'].shape}")

    # ---- Save config ----
    config = {
        'model_type': 'cnn',
        'architecture': 'cnn_tiny_8_16_mnist',
        'input_shape': [1, 28, 28],
        'num_layers': len(vta_layers),
        'layers': [],
        'global_x_scale': float(global_x_scale),
        'vta_config': {
            'BATCH': env.BATCH,
            'BLOCK_IN': env.BLOCK_IN,
            'BLOCK_OUT': env.BLOCK_OUT,
        },
        'bitstream': '1x16_i8w8a32_15_15_18_17.bit',
        'calibration_samples': args.cal_samples,
    }

    for i, vl in enumerate(vta_layers):
        layer_config = {
            'index': i,
            'type': vl['layer_type'],
            'in_f': vl['in_f'],
            'out_f': vl['out_f'],
            'real_in': vl['real_in'],
            'real_out': vl['real_out'],
            'n_tiles': vl['n_tiles'],
            'm_tiles': vl['m_tiles'],
            'o_total': vl['o_total'],
            'o_tile': vl['o_tile'],
            'n_chunks': vl['n_chunks'],
            'shift': shift_amounts[i],
            'w_scale': vl['w_scale'],
            'module_file': module_filenames[i],
            'weight_file': f'W{i}_tiled.npy',
            'bias_file': f'b{i}.npy',
        }
        if vl['layer_type'] == 'conv':
            layer_config['kernel_size'] = vl['kernel_size']
            layer_config['padding'] = vl['padding']
            layer_config['in_channels'] = vl['in_channels']
            layer_config['out_channels'] = vl['out_channels']
            layer_config['pool'] = vl.get('pool', 0)
        config['layers'].append(layer_config)

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Export complete: {args.output_dir}/")
    print(f"  Model: CNN tiny [8, 16] MNIST")
    print(f"  Layers: {len(vta_layers)} ({sum(1 for v in vta_layers if v['layer_type']=='conv')} conv, "
          f"{sum(1 for v in vta_layers if v['layer_type']=='dense')} dense)")
    print(f"  Shift amounts: {shift_amounts}")
    for i, vl in enumerate(vta_layers):
        tile_info = f"o_tile={vl['o_tile']}" if vl['n_chunks'] == 1 else f"o_total={vl['o_total']} o_tile={vl['o_tile']} chunks={vl['n_chunks']}"
        print(f"    Layer {i} ({vl['layer_type']}): "
              f"{tile_info} n={vl['n_tiles']} m={vl['m_tiles']} shift={shift_amounts[i]}")
    print(f"  Files:")
    for f_name in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f_name))
        print(f"    {f_name} ({size} bytes)")
    print(f"\nCopy to board:")
    basename = os.path.basename(args.output_dir.rstrip('/'))
    print(f"  scp -r {args.output_dir} xilinx@192.168.3.1:/home/xilinx/models/vta/{basename}/")
    print(f"\nOn board, link .o -> .so:")
    print(f"  cd /home/xilinx/models/vta/{basename}/")
    for fname in module_filenames:
        so_name = fname.replace('.o', '.so')
        print(f"  gcc -shared -o {so_name} {fname} -ltvm_runtime")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
