#!/usr/bin/env python3
"""Board-side VTA CNN MNIST inference test.

Loads pre-compiled VTA CNN modules (from export_vta_cnn.py), runs MNIST
test set, reports accuracy. This is the CNN equivalent of the MLP
benchmark path — run on the AUP-ZU3 board.

CNN pipeline per image:
  1. Quantize input image (float32 -> int8)
  2. Conv1: im2col -> VTA GEMM+shift+clip -> dequant -> ReLU -> MaxPool
  3. Conv2: im2col -> VTA GEMM+shift+clip -> dequant -> ReLU -> MaxPool
  4. Dense: GlobalAvgPool -> VTA GEMM+shift+clip -> dequant -> argmax

im2col, MaxPool, GlobalAvgPool, ReLU, quantization all run on ARM CPU.
GEMM+shift+clip runs on VTA hardware.

Usage (on board):
    export LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python
    python3 test_vta_cnn.py /home/xilinx/models/vta/cnn_mnist_tiny /home/xilinx/MNIST/raw

Date: April 1, 2026
"""
import numpy as np
import tvm
import tvm.runtime
import vta
import ctypes
import json
import os
import sys
import time
import struct
import gzip
import math


# ---- MNIST loading ----

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- CPU helper functions ----

def im2col(x, kH, kW, pad=0, stride=1):
    """x: (H, W, C) -> (H_out*W_out, kH*kW*C)"""
    H, W, C = x.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    if pad > 0:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    # Vectorized im2col using stride tricks
    patches = np.zeros((H_out * W_out, kH * kW * C), dtype=x.dtype)
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            patches[idx] = x[i*stride:i*stride+kH, j*stride:j*stride+kW, :].flatten()
            idx += 1
    return patches, H_out, W_out


def maxpool2d(x, pool_size=2):
    """x: (H, W, C) -> (H//pool, W//pool, C)"""
    H, W, C = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    out = np.zeros((H_out, W_out, C), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            out[i, j] = x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size].max(axis=(0, 1))
    return out


# ---- Main ----

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <mnist_dir> [num_images]")
        sys.exit(1)

    model_dir = sys.argv[1]
    mnist_dir = sys.argv[2]
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

    # ---- Load config ----
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)

    assert config['model_type'] == 'cnn', f"Expected CNN model, got {config['model_type']}"

    env = vta.get_env()
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT
    print(f"VTA env: BATCH={env.BATCH}, BLOCK_IN={BLOCK_IN}, BLOCK_OUT={BLOCK_OUT}")
    print(f"Model: {config['architecture']}")

    # ---- Clear stale PYNQ state ----
    stale_json = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale_json):
            os.remove(stale_json)
    except Exception:
        pass

    # ---- Load bitstream ----
    bitstream_name = config.get('bitstream', '1x16_i8w8a32_15_15_18_17.bit')
    bitstream_path = None
    for candidate in [
        f'/root/.vta_cache/ultra96/0_0_2/{bitstream_name}',
        f'/home/xilinx/.vta_cache/ultra96/0_0_2/{bitstream_name}',
        os.path.join(model_dir, bitstream_name),
    ]:
        if os.path.exists(candidate):
            bitstream_path = candidate
            break

    if bitstream_path is None:
        print(f"ERROR: Bitstream not found: {bitstream_name}")
        sys.exit(1)

    print(f"Loading bitstream: {bitstream_path}")
    from pynq import Overlay
    overlay = Overlay(bitstream_path)
    print(f"  IPs: {list(overlay.ip_dict.keys())}")

    # ---- Load VTA runtime ----
    vta_lib = None
    for candidate in [
        '/home/xilinx/tvm-src/build/libvta.so',
    ]:
        if os.path.exists(candidate):
            vta_lib = candidate
            break
    if vta_lib is None:
        print("ERROR: libvta.so not found")
        sys.exit(1)
    print(f"Loading VTA runtime: {vta_lib}")
    ctypes.CDLL(vta_lib, ctypes.RTLD_GLOBAL)
    ctx = tvm.device("ext_dev", 0)

    # ---- Load modules and weights ----
    layer_info = config['layers']
    num_layers = len(layer_info)
    gemm_modules = []
    W_nds = []
    layer_meta = []

    for lc in layer_info:
        # Module file: .o needs to be linked to .so on board first
        mod_file = lc['module_file']
        so_file = mod_file.replace('.o', '.so')
        mod_path = os.path.join(model_dir, so_file)
        if not os.path.exists(mod_path):
            # Try .o.so (legacy naming)
            mod_path = os.path.join(model_dir, mod_file + '.so')
        if not os.path.exists(mod_path):
            mod_path = os.path.join(model_dir, mod_file)

        print(f"  Loading layer {lc['index']} ({lc['type']}): {mod_path}")
        f = tvm.runtime.load_module(mod_path)
        gemm_modules.append(f)

        W_tiled = np.load(os.path.join(model_dir, lc['weight_file']))
        b_float = np.load(os.path.join(model_dir, lc['bias_file']))
        W_nd = tvm.nd.array(W_tiled, ctx)
        W_nds.append(W_nd)

        meta = {
            'type': lc['type'],
            'in_f': lc['in_f'],
            'out_f': lc['out_f'],
            'real_in': lc['real_in'],
            'real_out': lc['real_out'],
            'n_tiles': lc['n_tiles'],
            'm_tiles': lc['m_tiles'],
            'o_total': lc['o_total'],
            'o_tile': lc['o_tile'],
            'n_chunks': lc['n_chunks'],
            'shift': lc['shift'],
            'w_scale': lc['w_scale'],
            'b_float': b_float,
        }
        if lc['type'] == 'conv':
            meta['kernel_size'] = lc['kernel_size']
            meta['padding'] = lc['padding']
            meta['in_channels'] = lc['in_channels']
            meta['out_channels'] = lc['out_channels']
            meta['pool'] = lc.get('pool', 0)
        layer_meta.append(meta)

    # ---- Pre-allocate VTA buffers (sized for o_tile, not o_total) ----
    A_nds = []
    C_nds = []
    for lm in layer_meta:
        A_nds.append(tvm.nd.array(
            np.zeros((lm['o_tile'], lm['n_tiles'], 1, BLOCK_IN), dtype=np.int8), ctx))
        C_nds.append(tvm.nd.array(
            np.zeros((lm['o_tile'], lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8), ctx))

    # ---- Load MNIST ----
    images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz')
    labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz')
    test_images = load_mnist_images(images_path)  # (10000, 28, 28)
    test_labels = load_mnist_labels(labels_path)
    num_images = min(num_images, len(test_labels))
    print(f"Loaded {num_images} MNIST test images")

    # ---- Inference function ----
    def tile_input_2d(x_int8_2d, n_tiles):
        """Tile a 2D matrix (rows, padded_cols) for VTA.
        Output: (rows, n_tiles, 1, BLOCK_IN)
        """
        rows = x_int8_2d.shape[0]
        return x_int8_2d.reshape(rows, n_tiles, 1, BLOCK_IN)

    def infer_one(img):
        """Run single image through VTA CNN. img: (28, 28) float32 [0,1]."""
        x_s = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        current_scale = x_s
        h_float = img  # spatial activation

        for i, (lm, gm) in enumerate(zip(layer_meta, gemm_modules)):
            if lm['type'] == 'conv':
                # Prepare spatial input
                if h_float.ndim == 2:
                    h_spatial = h_float[:, :, np.newaxis]  # (H, W, 1)
                else:
                    h_spatial = h_float

                # im2col
                patches, H_out, W_out = im2col(
                    h_spatial, lm['kernel_size'], lm['kernel_size'],
                    pad=lm['padding'])
                # patches: (H_out*W_out, kH*kW*C_in)

                # Pad to BLOCK_IN alignment
                real_dim = patches.shape[1]
                if real_dim < lm['in_f']:
                    patches = np.pad(patches, ((0, 0), (0, lm['in_f'] - real_dim)),
                                     mode='constant')

                # Quantize
                p_int8 = np.clip(np.round(patches / current_scale), -128, 127).astype(np.int8)

                # Run VTA GEMM in chunks (tiled o dimension)
                o_total = lm['o_total']
                o_tile = lm['o_tile']
                n_chunks = lm['n_chunks']
                vta_out_full = np.zeros((o_total, lm['out_f']), dtype=np.int8)

                for chunk in range(n_chunks):
                    start = chunk * o_tile
                    end = start + o_tile
                    p_tiled = p_int8[start:end].reshape(o_tile, lm['n_tiles'], 1, BLOCK_IN)
                    A_nds[i].copyfrom(p_tiled)
                    C_nds[i].copyfrom(np.zeros((o_tile, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                    gm(A_nds[i], W_nds[i], C_nds[i])
                    vta_out_full[start:end] = C_nds[i].numpy().reshape(o_tile, lm['out_f'])

                # Dequantize + bias (broadcast bias across spatial positions)
                combined = current_scale * lm['w_scale'] * (2 ** lm['shift'])
                y_float = vta_out_full.astype(np.float32) * combined + lm['b_float'][:lm['out_f']]

                # ReLU
                y_float = np.maximum(y_float, 0)

                # Reshape to spatial, take real channels
                y_spatial = y_float[:, :lm['real_out']].reshape(H_out, W_out, lm['real_out'])

                # MaxPool
                if lm.get('pool', 0) > 0:
                    y_spatial = maxpool2d(y_spatial, lm['pool'])

                h_float = y_spatial
                next_scale = np.max(np.abs(h_float)) / 127.0
                current_scale = max(next_scale, 1e-10)

            elif lm['type'] == 'dense':
                # GlobalAvgPool: (H, W, C) -> (C,)
                h_vec = h_float.mean(axis=(0, 1))

                # Pad to BLOCK_IN alignment
                if len(h_vec) < lm['in_f']:
                    h_vec_padded = np.zeros(lm['in_f'], dtype=np.float32)
                    h_vec_padded[:len(h_vec)] = h_vec
                    h_vec = h_vec_padded

                h_int8 = np.clip(np.round(h_vec / current_scale), -128, 127).astype(np.int8)

                # Tile: (1, n_tiles, 1, BLOCK_IN)
                h_tiled = h_int8.reshape(1, lm['n_tiles'], 1, BLOCK_IN)
                A_nds[i].copyfrom(h_tiled)
                C_nds[i].copyfrom(np.zeros((1, lm['m_tiles'], 1, BLOCK_OUT), dtype=np.int8))
                gm(A_nds[i], W_nds[i], C_nds[i])

                vta_out = C_nds[i].numpy().reshape(lm['out_f'])
                combined = current_scale * lm['w_scale'] * (2 ** lm['shift'])
                y_float = vta_out.astype(np.float32) * combined + lm['b_float'][:lm['out_f']]

                return int(np.argmax(y_float[:lm['real_out']]))

        raise RuntimeError("No dense layer found at end of CNN")

    # ---- Run test set ----
    print(f"\n{'='*60}")
    print(f"Running {num_images} MNIST images on VTA CNN...")
    print(f"{'='*60}")

    correct = 0
    t_start = time.time()

    for img_idx in range(num_images):
        pred = infer_one(test_images[img_idx])
        if pred == test_labels[img_idx]:
            correct += 1

        if (img_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            acc = correct / (img_idx + 1)
            ms_per = elapsed / (img_idx + 1) * 1000
            fps = (img_idx + 1) / elapsed
            print(f"  [{img_idx+1}/{num_images}] acc={acc:.4f}, {ms_per:.1f}ms/img, {fps:.1f} FPS")

    total_time = time.time() - t_start
    final_acc = correct / num_images

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Model: CNN tiny [8, 16] MNIST")
    print(f"  Accuracy:    {final_acc:.4f} ({correct}/{num_images})")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  Throughput:  {num_images/total_time:.1f} FPS")
    print(f"  Latency:     {total_time/num_images*1000:.2f} ms/image")
    print(f"  NOTE: Includes CPU im2col + maxpool + quantization overhead")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
