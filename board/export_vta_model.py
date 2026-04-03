#!/usr/bin/env python3
"""Export pre-compiled VTA modules for board-side inference.

Compiles VTA GEMM+shift+clip modules on the host (cross-compiled for
aarch64) and saves them alongside weights and config as a self-contained
model directory. The board-side benchmark.py loads these directly via
tvm.runtime.load_module() — no RPC needed.

Usage (from Ubuntu host):
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 export_vta_model.py \
            --weights-dir ./vta_mnist_weights/ \
            --output-dir ./vta_export/mlp_mnist_tiny/

Then copy to board:
    scp -r ./vta_export/mlp_mnist_tiny/ xilinx@192.168.3.1:/home/xilinx/vta_models/mlp_mnist_tiny/

Date: March 30, 2026
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


# ---- MNIST loading (same as test_vta_mnist.py) ----

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
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- VTA helpers (same as test_vta_mnist.py) ----

def tile_weights(W_int8, env):
    out_f, in_f = W_int8.shape
    m = out_f // env.BLOCK_OUT
    n = in_f // env.BLOCK_IN
    return W_int8.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)

def pad_to_multiple(W, b, block_size):
    out_f, in_f = W.shape
    if out_f % block_size == 0:
        return W, b, out_f
    pad_out = block_size - (out_f % block_size)
    W_padded = np.pad(W, ((0, pad_out), (0, 0)), mode='constant')
    b_padded = np.pad(b, (0, pad_out), mode='constant')
    return W_padded, b_padded, out_f

def compute_shift_bits(x_int8_samples, W_int8):
    acc = x_int8_samples.astype(np.int32) @ W_int8.T.astype(np.int32)
    max_abs = np.max(np.abs(acc))
    if max_abs <= 127:
        return 0
    return int(math.ceil(math.log2(max_abs / 127.0)))


# ---- Module compilation ----

def compile_gemm_with_shift(env, o, n, m, shift_bits):
    """Compile VTA GEMM + ALU shift + clip. Returns TVM Module (not uploaded)."""
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


def main():
    parser = argparse.ArgumentParser(description='Export pre-compiled VTA model for board-side inference')
    parser.add_argument('--weights-dir', required=True,
                        help='Directory with W0.npy, b0.npy, ... (Brevitas or trained)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for compiled model')
    parser.add_argument('--mnist-dir', default='./mnist_data',
                        help='MNIST data directory (for calibration)')
    parser.add_argument('--cal-samples', type=int, default=100,
                        help='Number of calibration samples for shift amounts')
    parser.add_argument('--architecture', default='784,64,32,10',
                        help='Comma-separated layer dimensions (default: 784,64,32,10)')
    args = parser.parse_args()

    env = vta.get_env()
    print(f"VTA env: TARGET={env.TARGET}, BLOCK_IN={env.BLOCK_IN}, BLOCK_OUT={env.BLOCK_OUT}")

    raw_dims = [int(d) for d in args.architecture.split(',')]
    print(f"Architecture: {' -> '.join(str(d) for d in raw_dims)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load weights ----
    print(f"\nLoading weights from {args.weights_dir}/")
    weight_list = []
    for i in range(len(raw_dims) - 1):
        W = np.load(os.path.join(args.weights_dir, f'W{i}.npy'))
        b = np.load(os.path.join(args.weights_dir, f'b{i}.npy'))
        weight_list.append((W, b))
        print(f"  Layer {i}: {W.shape}")

    # ---- Quantize and pad ----
    print(f"\nQuantizing weights for VTA...")
    vta_layers = []
    for i, (W, b) in enumerate(weight_list):
        W_padded, b_padded, real_out = pad_to_multiple(W, b, env.BLOCK_OUT)
        out_f, in_f = W_padded.shape
        w_scale = np.max(np.abs(W_padded)) / 127.0 if np.max(np.abs(W_padded)) > 0 else 1e-10
        W_int8 = np.clip(np.round(W_padded / w_scale), -128, 127).astype(np.int8)
        W_tiled = tile_weights(W_int8, env)

        vta_layers.append({
            'W_int8': W_int8, 'W_tiled': W_tiled,
            'b_float': b_padded, 'w_scale': w_scale,
            'in_f': in_f, 'out_f': out_f, 'real_out': real_out,
            'n_tiles': in_f // env.BLOCK_IN,
            'm_tiles': out_f // env.BLOCK_OUT,
        })
        print(f"  Layer {i}: {in_f}->{out_f} (real={real_out}), "
              f"w_scale={w_scale:.6f}, tiles n={in_f//env.BLOCK_IN} m={out_f//env.BLOCK_OUT}")

    # ---- Calibrate shift amounts ----
    print(f"\nCalibrating shift amounts ({args.cal_samples} samples)...")
    mnist = download_mnist(args.mnist_dir)
    test_images = load_mnist_images(mnist['test_images'])
    cal_images = test_images[:args.cal_samples]

    x_scales = [np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
                for img in cal_images]
    global_x_scale = float(np.mean(x_scales))

    h_cal = np.clip(np.round(cal_images / global_x_scale), -128, 127).astype(np.int8)

    shift_amounts = []
    cal_layer_scales = [global_x_scale]
    for i, layer in enumerate(vta_layers):
        shift = compute_shift_bits(h_cal, layer['W_int8'])
        shift_amounts.append(shift)

        acc_batch = h_cal.astype(np.int32) @ layer['W_int8'].T.astype(np.int32)
        shifted_batch = acc_batch >> shift
        clipped_batch = np.clip(shifted_batch, -128, 127).astype(np.int8)

        combined = cal_layer_scales[-1] * layer['w_scale'] * (2 ** shift)
        y_float_batch = clipped_batch.astype(np.float32) * combined + layer['b_float']

        if i < len(vta_layers) - 1:
            y_float_batch = np.maximum(y_float_batch, 0)
            next_scale = np.max(np.abs(y_float_batch)) / 127.0 if np.max(np.abs(y_float_batch)) > 0 else 1e-10
            h_cal = np.clip(np.round(y_float_batch / next_scale), -128, 127).astype(np.int8)
            cal_layer_scales.append(next_scale)

        print(f"  Layer {i}: shift={shift}")

    # ---- Compile and save modules ----
    print(f"\nCompiling VTA modules...")
    module_filenames = []
    for i, layer in enumerate(vta_layers):
        shift = shift_amounts[i]
        n_t = layer['n_tiles']
        m_t = layer['m_tiles']
        fname = f"layer{i}_n{n_t}_m{m_t}_s{shift}.o"
        print(f"  Layer {i} (o=1, n={n_t}, m={m_t}, shift={shift})...", end=" ", flush=True)

        mod = compile_gemm_with_shift(env, 1, n_t, m_t, shift)
        out_path = os.path.join(args.output_dir, fname)
        mod.save(out_path)
        module_filenames.append(fname)
        print(f"OK -> {fname}")

    # ---- Save weights ----
    print(f"\nSaving weights...")
    for i, layer in enumerate(vta_layers):
        np.save(os.path.join(args.output_dir, f'W{i}_tiled.npy'), layer['W_tiled'])
        np.save(os.path.join(args.output_dir, f'b{i}.npy'), layer['b_float'])
        print(f"  W{i}_tiled.npy: {layer['W_tiled'].shape}, b{i}.npy: {layer['b_float'].shape}")

    # ---- Save config ----
    config = {
        'architecture': raw_dims,
        'num_layers': len(vta_layers),
        'layers': [],
        'global_x_scale': global_x_scale,
        'vta_config': {
            'BATCH': env.BATCH,
            'BLOCK_IN': env.BLOCK_IN,
            'BLOCK_OUT': env.BLOCK_OUT,
        },
        'bitstream': '1x16_i8w8a32_15_15_18_17.bit',
        'calibration_samples': args.cal_samples,
    }
    for i, layer in enumerate(vta_layers):
        config['layers'].append({
            'index': i,
            'in_f': layer['in_f'],
            'out_f': layer['out_f'],
            'real_out': layer['real_out'],
            'n_tiles': layer['n_tiles'],
            'm_tiles': layer['m_tiles'],
            'shift': shift_amounts[i],
            'w_scale': float(layer['w_scale']),
            'module_file': module_filenames[i],
            'weight_file': f'W{i}_tiled.npy',
            'bias_file': f'b{i}.npy',
        })

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Export complete: {args.output_dir}/")
    print(f"  Architecture: {' -> '.join(str(d) for d in raw_dims)}")
    print(f"  Shift amounts: {shift_amounts}")
    print(f"  Files:")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"    {f} ({size} bytes)")
    print(f"\nCopy to board:")
    print(f"  scp -r {args.output_dir} xilinx@192.168.3.1:/home/xilinx/vta_models/{os.path.basename(args.output_dir)}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
