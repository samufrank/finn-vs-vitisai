"""MNIST MLP accuracy test on VTA.

Trains a simple MLP (784->64->32->10), quantizes weights for VTA,
runs the full MNIST test set via RPC, reports accuracy.

Architecture matches FINN's MLP tiny [64,32] for direct comparison.
Last layer padded from 10 to 16 outputs (BLOCK_OUT=16 constraint).

Two modes:
  1. Train fresh: trains a simple PyTorch MLP on MNIST
  2. Load FINN weights: reuses Brevitas-trained weights from FINN deploy/

Usage:
    cd ~/dev/CEN571-final/tvm-v0.12.0

    # Train fresh (requires torchvision):
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 test_vta_mnist.py --train

    # Load pre-saved weights:
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 test_vta_mnist.py --weights-dir ./vta_mnist_weights/

    # After first run with --train, weights are auto-saved to ./vta_mnist_weights/
"""
import numpy as np
import tvm
from tvm import te, rpc
from tvm.contrib import utils
import vta
import time
import sys
import math
import os
import argparse
import struct
import gzip
from urllib.request import urlretrieve

BOARD_IP = "192.168.3.1"
RPC_PORT = 9091

# ---- MNIST data loading (no torchvision dependency) ----

MNIST_FILENAMES = {
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz',
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
}

# Mirrors in priority order (yann.lecun.com is frequently down)
MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
    'http://yann.lecun.com/exdb/mnist/',
]


def download_mnist(data_dir='./mnist_data'):
    """Load MNIST from local dir, or download if not present."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, filename in MNIST_FILENAMES.items():
        fname = os.path.join(data_dir, filename)
        if not os.path.exists(fname):
            downloaded = False
            for mirror in MNIST_MIRRORS:
                url = mirror + filename
                try:
                    print(f"  Downloading {key} from {mirror}...", end=" ", flush=True)
                    urlretrieve(url, fname)
                    print("OK")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"failed ({e})")
            if not downloaded:
                raise RuntimeError(
                    f"Could not download {filename}. Copy MNIST data manually:\n"
                    f"  scp xilinx@192.168.3.1:/home/xilinx/MNIST/raw/* {data_dir}/")
        paths[key] = fname
    return paths


def load_mnist_images(path):
    """Load MNIST images from gzipped IDX file."""
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def load_mnist_labels(path):
    """Load MNIST labels from gzipped IDX file."""
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- Simple MLP training (numpy-only fallback, or PyTorch if available) ----

def train_mlp_pytorch(train_images, train_labels, layer_dims, epochs=10, lr=0.01, batch_size=128):
    """Train MLP using PyTorch. Returns list of (W, b) numpy arrays."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("PyTorch not available, using numpy training (slower, lower accuracy)")
        return train_mlp_numpy(train_images, train_labels, layer_dims, epochs)

    class MLP(nn.Module):
        def __init__(self, dims):
            super().__init__()
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = MLP(layer_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = torch.tensor(train_images, dtype=torch.float32)
    Y = torch.tensor(train_labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(xb)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/total:.4f}, "
                  f"acc={correct/total:.4f}")

    # Extract weights
    weight_list = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            weight_list.append((W, b))

    return weight_list


def train_mlp_numpy(train_images, train_labels, layer_dims, epochs=10):
    """Minimal numpy MLP training (SGD, no batching). Fallback if no PyTorch."""
    np.random.seed(42)
    weights = []
    for i in range(len(layer_dims) - 1):
        std = np.sqrt(2.0 / layer_dims[i])
        W = np.random.randn(layer_dims[i+1], layer_dims[i]).astype(np.float32) * std
        b = np.zeros(layer_dims[i+1], dtype=np.float32)
        weights.append((W, b))

    lr = 0.001
    for epoch in range(epochs):
        # Simple SGD on random subset
        idx = np.random.permutation(len(train_images))[:1000]
        correct = 0
        for j in idx:
            x = train_images[j]
            y = train_labels[j]

            # Forward
            activations = [x]
            for k, (W, b) in enumerate(weights):
                h = W @ activations[-1] + b
                if k < len(weights) - 1:
                    h = np.maximum(h, 0)
                activations.append(h)

            # Softmax + loss
            logits = activations[-1]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            correct += (np.argmax(logits) == y)

            # Backward (simplified)
            grad = probs.copy()
            grad[y] -= 1

            for k in reversed(range(len(weights))):
                W, b = weights[k]
                a = activations[k]
                dW = np.outer(grad, a)
                db = grad
                weights[k] = (W - lr * dW, b - lr * db)
                if k > 0:
                    grad = W.T @ grad
                    grad = grad * (activations[k] > 0)  # ReLU gradient

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train_acc={correct/len(idx):.4f}")

    return weights


# ---- VTA inference ----

def build_gemm_with_shift(remote, env, o, n, m, shift_bits):
    """Build VTA GEMM + ALU shift + clip module."""
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

    fname = f"gemm_shr_o{o}_n{n}_m{m}_s{shift_bits}.o"
    temp = utils.tempdir()
    mod.save(temp.relpath(fname))
    remote.upload(temp.relpath(fname))
    return remote.load_module(fname)


def tile_input(x_int8, env):
    n = len(x_int8) // env.BLOCK_IN
    return x_int8.reshape(1, n, 1, env.BLOCK_IN)


def tile_weights(W_int8, env):
    out_f, in_f = W_int8.shape
    m = out_f // env.BLOCK_OUT
    n = in_f // env.BLOCK_IN
    return W_int8.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)


def untile_output(C_tiled, out_features):
    return C_tiled.reshape(-1)[:out_features]


def compute_shift_bits(x_int8_samples, W_int8):
    """Determine shift from a batch of calibration samples."""
    acc = x_int8_samples.astype(np.int32) @ W_int8.T.astype(np.int32)
    max_abs = np.max(np.abs(acc))
    if max_abs <= 127:
        return 0
    return int(math.ceil(math.log2(max_abs / 127.0)))


def pad_to_multiple(W, b, block_size):
    """Pad output dimension to multiple of block_size."""
    out_f, in_f = W.shape
    if out_f % block_size == 0:
        return W, b, out_f
    pad_out = block_size - (out_f % block_size)
    W_padded = np.pad(W, ((0, pad_out), (0, 0)), mode='constant')
    b_padded = np.pad(b, (0, pad_out), mode='constant')
    return W_padded, b_padded, out_f  # return original out_f for unpadding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train fresh MLP')
    parser.add_argument('--weights-dir', type=str, default=None,
                        help='Directory with pre-saved numpy weights')
    parser.add_argument('--finn-weights', type=str, default=None,
                        help='FINN deploy directory with mlp_*.npy weights')
    parser.add_argument('--num-test', type=int, default=10000,
                        help='Number of test images (default: full 10K)')
    parser.add_argument('--save-weights', type=str, default='./vta_mnist_weights',
                        help='Save trained weights here')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mnist-dir', type=str, default='./mnist_data',
                        help='Directory with MNIST .gz files (downloads if missing)')
    args = parser.parse_args()

    env = vta.get_env()
    print(f"VTA env: TARGET={env.TARGET}, BLOCK_IN={env.BLOCK_IN}, BLOCK_OUT={env.BLOCK_OUT}")

    # ---- Architecture ----
    # 784 -> 64 -> 32 -> 10 (matching FINN MLP tiny)
    # Last layer padded to 16 for VTA (10 is not divisible by BLOCK_OUT=16)
    raw_dims = [784, 64, 32, 10]
    print(f"MLP architecture: {' -> '.join(str(d) for d in raw_dims)}")

    # ---- Load or train weights ----
    if args.weights_dir and os.path.exists(args.weights_dir):
        print(f"\nLoading weights from {args.weights_dir}/")
        weight_list = []
        for i in range(len(raw_dims) - 1):
            W = np.load(os.path.join(args.weights_dir, f'W{i}.npy'))
            b = np.load(os.path.join(args.weights_dir, f'b{i}.npy'))
            weight_list.append((W, b))
            print(f"  Layer {i+1}: {W.shape}")
    elif args.finn_weights and os.path.exists(args.finn_weights):
        print(f"\nLoading FINN weights from {args.finn_weights}/")
        # FINN deploy saves weights as mlp_0_W.npy, etc.
        # Adjust filenames based on actual FINN output
        weight_list = []
        for i in range(len(raw_dims) - 1):
            # Try common FINN naming patterns
            for pattern in [f'mlp_{i}_W.npy', f'fc{i}_weight.npy', f'W{i}.npy']:
                wpath = os.path.join(args.finn_weights, pattern)
                if os.path.exists(wpath):
                    W = np.load(wpath)
                    break
            else:
                print(f"  Could not find weights for layer {i}")
                return 1
            for pattern in [f'mlp_{i}_b.npy', f'fc{i}_bias.npy', f'b{i}.npy']:
                bpath = os.path.join(args.finn_weights, pattern)
                if os.path.exists(bpath):
                    b = np.load(bpath)
                    break
            else:
                b = np.zeros(W.shape[0], dtype=np.float32)
            weight_list.append((W, b))
            print(f"  Layer {i+1}: {W.shape}")
    else:
        print(f"\nTraining MLP on MNIST ({args.epochs} epochs)...")
        mnist = download_mnist(args.mnist_dir)
        train_images = load_mnist_images(mnist['train_images'])
        train_labels = load_mnist_labels(mnist['train_labels'])
        weight_list = train_mlp_pytorch(train_images, train_labels, raw_dims,
                                         epochs=args.epochs)
        # Save weights
        os.makedirs(args.save_weights, exist_ok=True)
        for i, (W, b) in enumerate(weight_list):
            np.save(os.path.join(args.save_weights, f'W{i}.npy'), W)
            np.save(os.path.join(args.save_weights, f'b{i}.npy'), b)
        print(f"  Weights saved to {args.save_weights}/")

    # ---- CPU accuracy check ----
    print(f"\nLoading MNIST test data...")
    mnist = download_mnist(args.mnist_dir)
    test_images = load_mnist_images(mnist['test_images'])
    test_labels = load_mnist_labels(mnist['test_labels'])
    num_test = min(args.num_test, len(test_labels))
    test_images = test_images[:num_test]
    test_labels = test_labels[:num_test]

    print(f"  {num_test} test images loaded")

    # CPU float32 accuracy
    correct_cpu = 0
    for i in range(num_test):
        h = test_images[i]
        for j, (W, b) in enumerate(weight_list):
            h = h @ W.T + b
            if j < len(weight_list) - 1:
                h = np.maximum(h, 0)
        if np.argmax(h) == test_labels[i]:
            correct_cpu += 1
    cpu_acc = correct_cpu / num_test
    print(f"  CPU float32 accuracy: {cpu_acc:.4f} ({correct_cpu}/{num_test})")

    # ---- Quantize weights for VTA ----
    print(f"\nQuantizing weights for VTA...")
    vta_layers = []
    for i, (W, b) in enumerate(weight_list):
        # Pad output dim to BLOCK_OUT multiple
        W_padded, b_padded, real_out = pad_to_multiple(W, b, env.BLOCK_OUT)
        out_f, in_f = W_padded.shape

        # Quantize
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
        print(f"  Layer {i+1}: {in_f}->{out_f} (real={real_out}), "
              f"w_scale={w_scale:.6f}, tiles n={in_f//env.BLOCK_IN} m={out_f//env.BLOCK_OUT}")

    # ---- Calibrate shift amounts ----
    print(f"\nCalibrating shift amounts (100 samples)...")
    cal_samples = 100
    cal_images = test_images[:cal_samples]

    # Quantize calibration inputs
    x_scales = []
    for img in cal_images:
        xs = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        x_scales.append(xs)
    global_x_scale = np.mean(x_scales)

    # Run calibration through each layer to find shift amounts
    h_cal_batch = np.clip(np.round(cal_images / global_x_scale), -128, 127).astype(np.int8)

    shift_amounts = []
    cal_layer_scales = [global_x_scale]
    for i, layer in enumerate(vta_layers):
        shift = compute_shift_bits(h_cal_batch, layer['W_int8'])
        shift_amounts.append(shift)

        # Propagate through layer for next calibration
        acc_batch = h_cal_batch.astype(np.int32) @ layer['W_int8'].T.astype(np.int32)
        shifted_batch = acc_batch >> shift
        clipped_batch = np.clip(shifted_batch, -128, 127).astype(np.int8)

        combined = cal_layer_scales[-1] * layer['w_scale'] * (2 ** shift)
        y_float_batch = clipped_batch.astype(np.float32) * combined + layer['b_float']

        if i < len(vta_layers) - 1:
            y_float_batch = np.maximum(y_float_batch, 0)
            next_scale = np.max(np.abs(y_float_batch)) / 127.0 if np.max(np.abs(y_float_batch)) > 0 else 1e-10
            h_cal_batch = np.clip(np.round(y_float_batch / next_scale), -128, 127).astype(np.int8)
            cal_layer_scales.append(next_scale)

        print(f"  Layer {i+1}: shift={shift}")

    # ---- Connect to board ----
    print(f"\nConnecting to {BOARD_IP}:{RPC_PORT}...")
    remote = rpc.connect(BOARD_IP, RPC_PORT)
    vta.reconfig_runtime(remote)
    try:
        vta.program_fpga(remote, bitstream=None)
    except Exception:
        pass
    ctx = remote.ext_dev(0)

    # ---- Build VTA modules ----
    gemm_modules = []
    for i, layer in enumerate(vta_layers):
        shift = shift_amounts[i]
        print(f"  Building layer {i+1} (n={layer['n_tiles']}, m={layer['m_tiles']}, "
              f"shift={shift})...", end=" ", flush=True)
        f = build_gemm_with_shift(remote, env, 1, layer['n_tiles'],
                                   layer['m_tiles'], shift)
        gemm_modules.append(f)
        print("OK")

    # ---- Pre-allocate tvm arrays for weights ----
    W_nds = [tvm.nd.array(l['W_tiled'], ctx) for l in vta_layers]

    # ---- Run MNIST test set ----
    print(f"\n{'='*60}")
    print(f"Running {num_test} MNIST images on VTA...")
    print(f"{'='*60}")

    correct_vta = 0
    t_start = time.time()

    for img_idx in range(num_test):
        img = test_images[img_idx]
        label = test_labels[img_idx]

        # Quantize input
        x_s = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        h_int8 = np.clip(np.round(img / x_s), -128, 127).astype(np.int8)
        current_scale = x_s

        for i, (layer, f, shift) in enumerate(zip(vta_layers, gemm_modules, shift_amounts)):
            # Tile and run on VTA
            x_tiled = tile_input(h_int8, env)
            A_nd = tvm.nd.array(x_tiled, ctx)
            C_nd = tvm.nd.array(
                np.zeros((1, layer['m_tiles'], env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)
            f(A_nd, W_nds[i], C_nd)
            vta_out = untile_output(C_nd.numpy(), layer['out_f'])

            # Dequantize on CPU
            combined = current_scale * layer['w_scale'] * (2 ** shift)
            y_float = vta_out.astype(np.float32) * combined + layer['b_float']

            if i < len(vta_layers) - 1:
                y_float = np.maximum(y_float, 0)
                current_scale = np.max(np.abs(y_float)) / 127.0 if np.max(np.abs(y_float)) > 0 else 1e-10
                h_int8 = np.clip(np.round(y_float / current_scale), -128, 127).astype(np.int8)
            else:
                # Final layer: take argmax of real outputs only (not padding)
                pred = np.argmax(y_float[:layer['real_out']])
                if pred == label:
                    correct_vta += 1

        if (img_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            running_acc = correct_vta / (img_idx + 1)
            ms_per = elapsed / (img_idx + 1) * 1000
            print(f"  [{img_idx+1}/{num_test}] acc={running_acc:.4f}, "
                  f"{ms_per:.0f}ms/image")

    total_time = time.time() - t_start
    vta_acc = correct_vta / num_test

    # ---- Results ----
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Architecture: {' -> '.join(str(d) for d in raw_dims)}")
    print(f"  CPU float32 accuracy:  {cpu_acc:.4f} ({correct_cpu}/{num_test})")
    print(f"  VTA int8 accuracy:     {vta_acc:.4f} ({correct_vta}/{num_test})")
    print(f"  Accuracy delta:        {abs(cpu_acc - vta_acc):.4f}")
    print(f"  Total time:            {total_time:.1f}s for {num_test} images")
    n_layers = len(vta_layers)
    sleep_per_layer_ms = 1  # must match usleep() in pynq_driver_xrt.cc
    sleep_total_ms = n_layers * sleep_per_layer_ms
    print(f"  Avg per image:         {total_time/num_test*1000:.0f}ms "
          f"(~{total_time/num_test*1000 - sleep_total_ms:.0f}ms without sleep overhead)")
    print(f"  NOTE: Timing includes adaptive sleep overhead (~insn_count*2 us/layer)")
    print(f"        and RPC round-trip. Board-side execution needed for accurate timing.")
    print(f"  Shift amounts:         {shift_amounts}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
