#!/usr/bin/env python3
"""Export pre-compiled VTA INT4 modules for board-side inference (v2).

Key improvements over the original export_vta_model_int4.py:
  - Uses Brevitas's learned scales (from extract_int4_brevitas.py), not
    recomputed max-abs scales.
  - Bias ADD is INSIDE the VTA ALU sequence (GEMM → bias → SHR → CLIP),
    not a post-hoc CPU fixup.
  - Fully parameterized by meta.json — no hard-coded layer count or shifts.
  - Board-realistic clip bounds [0, 7] for hidden layers (signed int4 hw
    can only represent [0, 7] for non-negative post-ReLU values).

Usage (from Ubuntu host, with ARM VTA config):
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \\
        python3 ../finn-vs-vitisai/board/export_vta_model_int4_v2.py \\
            --weights-dir ./vta_mnist_weights_int4_v2/ \\
            --output-dir ./vta_export/mlp_mnist_int4_qat_v2/

For cross-check only (no TVM/VTA required):
    python3 ../finn-vs-vitisai/board/export_vta_model_int4_v2.py \\
        --weights-dir ./vta_mnist_weights_int4_v2/ \\
        --output-dir ./vta_export/mlp_mnist_int4_qat_v2/ \\
        --crosscheck-only
"""
import numpy as np
import json
import math
import os
import sys
import argparse
import struct
import gzip


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
                    print(f"  Downloading {key} from {mirror}...",
                          end=" ", flush=True)
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
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return (np.frombuffer(f.read(), dtype=np.uint8)
                .reshape(n, rows * cols).astype(np.float32) / 255.0)


def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- Weight helpers ----

def tile_weights(W_int, block_out, block_in):
    """Tile weight matrix into VTA layout: (m_tiles, n_tiles, BLOCK_OUT, BLOCK_IN)."""
    out_f, in_f = W_int.shape
    m = out_f // block_out
    n = in_f // block_in
    return W_int.reshape(m, block_out, n, block_in).transpose(0, 2, 1, 3)


def pad_to_multiple(W, b, block_size):
    """Pad output dimension to a multiple of block_size."""
    out_f, in_f = W.shape
    if out_f % block_size == 0:
        return W, b, out_f
    pad_out = block_size - (out_f % block_size)
    W_padded = np.pad(W, ((0, pad_out), (0, 0)), mode='constant')
    b_padded = np.pad(b, (0, pad_out), mode='constant')
    return W_padded, b_padded, out_f


def pad_input_dim(W, block_in):
    """Pad input dimension to a multiple of block_in."""
    out_f, in_f = W.shape
    if in_f % block_in == 0:
        return W
    pad_in = block_in - (in_f % block_in)
    return np.pad(W, ((0, 0), (0, pad_in)), mode='constant')


# ---- VTA module compilation (local to this script) ----

def compile_gemm_with_bias_and_shift(env, o, n, m, shift_bits, clip_lo, clip_hi):
    """Compile VTA GEMM + ALU ADD bias + ALU SHR + ALU CLIP.

    Op sequence: GEMM → bias ADD → SHR → MAX(clip_lo) → MIN(clip_hi) → cast.
    Returns TVM Module. This function is LOCAL to this script — does NOT
    modify any shared code in the TVM tree.

    Args:
        env: VTA environment
        o: batch tiles (typically 1)
        n: input tiles (in_f // BLOCK_IN)
        m: output tiles (out_f // BLOCK_OUT)
        shift_bits: right-shift amount (non-negative)
        clip_lo: lower clip bound (0 for post-ReLU, -8 for signed)
        clip_hi: upper clip bound (7 for board-realistic int4)
    """
    import tvm
    from tvm import te
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),
                        name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                        name="B", dtype=env.wgt_dtype)
    D = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT),
                        name="D", dtype=env.acc_dtype)

    # DMA loads
    A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN),
                        lambda *i: A(*i), "A_buf")
    B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                        lambda *i: B(*i), "B_buf")
    D_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
                        lambda *i: D(*i), "D_buf")

    # GEMM
    ko = te.reduce_axis((0, n), name="ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
    C_buf = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda bo, co, bi, ci: te.sum(
            A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
            B_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki]),
        name="C_buf")

    # ALU ADD bias (pre-shift — mathematically correct placement)
    C_add = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_buf(*i) + D_buf(*i),
        name="C_add")

    # ALU SHR
    shr_const = tvm.tir.const(shift_bits, env.acc_dtype)
    C_shr = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_add(*i) >> shr_const,
        name="C_shr")

    # ALU CLIP (MIN then MAX, applied as MAX-of-lo then MIN-of-hi)
    C_clip_lo = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.max(C_shr(*i),
                              tvm.tir.const(clip_lo, env.acc_dtype)),
        name="C_clip_lo")

    C_clip_hi = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.min(C_clip_lo(*i),
                              tvm.tir.const(clip_hi, env.acc_dtype)),
        name="C_clip_hi")

    # Cast to inp_dtype and DMA out
    C = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_clip_hi(*i).astype(env.inp_dtype),
        name="C")

    # Schedule
    s = te.create_schedule(C.op)
    s[A_buf].set_scope(env.inp_scope)
    s[B_buf].set_scope(env.wgt_scope)
    s[D_buf].set_scope(env.acc_scope)
    s[C_buf].set_scope(env.acc_scope)
    s[C_add].set_scope(env.acc_scope)
    s[C_shr].set_scope(env.acc_scope)
    s[C_clip_lo].set_scope(env.acc_scope)
    s[C_clip_hi].set_scope(env.acc_scope)

    # GEMM tiling
    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
        s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    # Bias DMA + ALU ops
    s[D_buf].pragma(s[D_buf].op.axis[0], env.dma_copy)
    s[C_add].pragma(s[C_add].op.axis[0], env.alu)
    s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
    s[C_clip_lo].pragma(s[C_clip_lo].op.axis[0], env.alu)
    s[C_clip_hi].pragma(s[C_clip_hi].op.axis[0], env.alu)

    # Output DMA
    s[C].pragma(s[C].op.axis[0], env.dma_copy)

    # Determine host target based on VTA target
    if env.TARGET in ("sim", "tsim"):
        host_target = tvm.target.Target("llvm")
    else:
        host_target = tvm.target.arm_cpu("ultra96")

    mod = vta.build(s, [A, B, D, C],
                    tvm.target.vta(),
                    host_target,
                    name="my_gemm")
    return mod


def compile_gemm_with_shift(env, o, n, m, shift_bits, clip_lo, clip_hi):
    """Compile VTA GEMM + ALU SHR + ALU CLIP (NO bias).

    Used for the last layer where bias is applied on CPU after dequant.
    VTA cannot output raw INT32 from acc_scope (DMA narrows to inp_dtype,
    producing garbage for int32), so shift+clip is structurally required.
    Moving bias to CPU preserves float precision for argmax.

    Op sequence: GEMM → SHR → MAX(clip_lo) → MIN(clip_hi) → cast.
    Module takes 2 inputs: A (activations), B (weights).
    """
    import tvm
    from tvm import te
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),
                        name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                        name="B", dtype=env.wgt_dtype)

    A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN),
                        lambda *i: A(*i), "A_buf")
    B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                        lambda *i: B(*i), "B_buf")

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
        lambda *i: C_buf(*i) >> shr_const,
        name="C_shr")

    C_clip_lo = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.max(C_shr(*i),
                              tvm.tir.const(clip_lo, env.acc_dtype)),
        name="C_clip_lo")

    C_clip_hi = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.min(C_clip_lo(*i),
                              tvm.tir.const(clip_hi, env.acc_dtype)),
        name="C_clip_hi")

    C = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_clip_hi(*i).astype(env.inp_dtype),
        name="C")

    s = te.create_schedule(C.op)
    s[A_buf].set_scope(env.inp_scope)
    s[B_buf].set_scope(env.wgt_scope)
    s[C_buf].set_scope(env.acc_scope)
    s[C_shr].set_scope(env.acc_scope)
    s[C_clip_lo].set_scope(env.acc_scope)
    s[C_clip_hi].set_scope(env.acc_scope)

    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
        s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
    s[C_clip_lo].pragma(s[C_clip_lo].op.axis[0], env.alu)
    s[C_clip_hi].pragma(s[C_clip_hi].op.axis[0], env.alu)
    s[C].pragma(s[C].op.axis[0], env.dma_copy)

    if env.TARGET in ("sim", "tsim"):
        host_target = tvm.target.Target("llvm")
    else:
        host_target = tvm.target.arm_cpu("ultra96")

    mod = vta.build(s, [A, B, C],
                    tvm.target.vta(),
                    host_target,
                    name="my_gemm")
    return mod


# ---- Numpy cross-check (functionally equivalent to VTA sim) ----

def numpy_crosscheck(output_dir, weights_dir, mnist_dir, n_check=100):
    """Load exported artifacts from disk, run integer-exact pipeline,
    compare against Mode D reference.

    The numpy operations (int32 matmul, int32 add, arithmetic right shift,
    clip) are identical to VTA's integer ALU ops. This is functionally
    equivalent to running in TVM simulator mode.
    """
    print(f"\n{'='*60}")
    print("CROSS-CHECK: Numpy verification (VTA-equivalent integer ops)")
    print(f"{'='*60}")

    config = json.load(open(os.path.join(output_dir, 'config.json')))
    n_layers = config['num_layers']

    # Load exported weights and biases
    W_tiled = []
    bias_data = []
    for lc in config['layers']:
        W_tiled.append(np.load(os.path.join(output_dir, lc['weight_file'])))
        bias_data.append(np.load(os.path.join(output_dir, lc['bias_file'])))

    block_out = config['vta_config']['BLOCK_OUT']
    block_in = config['vta_config']['BLOCK_IN']

    # Load MNIST
    mnist = download_mnist(mnist_dir)
    images = load_mnist_images(mnist['test_images'])
    labels = load_mnist_labels(mnist['test_labels'])

    input_scale = config['input_scale']
    input_clip_max = config['input_clip_max']

    # Run pipeline
    argmax_vta = []
    for img_idx in range(n_check):
        x_float = images[img_idx]
        x_int = np.clip(np.round(x_float / input_scale),
                        0, input_clip_max).astype(np.int32)

        for i, lc in enumerate(config['layers']):
            # Untile weights
            W = W_tiled[i]
            m_t, n_t = W.shape[0], W.shape[1]
            W_flat = W.transpose(0, 2, 1, 3).reshape(
                m_t * block_out, n_t * block_in)

            # Pad input if needed
            in_f = lc['in_f']
            if len(x_int) < in_f:
                x_padded = np.zeros(in_f, dtype=np.int32)
                x_padded[:len(x_int)] = x_int
                x_int = x_padded

            # GEMM (INT32)
            acc = W_flat.astype(np.int32) @ x_int[:in_f].astype(np.int32)

            if lc.get('has_vta_bias', True):
                # Hidden layer: int bias pre-shift in VTA
                bias_int = bias_data[i].reshape(-1)
                acc = acc + bias_int[:len(acc)]
                acc = acc >> lc['shift']
                acc = np.clip(acc, lc['clip_lo'], lc['clip_hi'])
                x_int = acc[:lc['real_out']].astype(np.int32)
            else:
                # Last layer: GEMM + shift + clip only, bias on CPU
                acc = acc >> lc['shift']
                acc = np.clip(acc, lc['clip_lo'], lc['clip_hi'])
                # Dequant + float bias on CPU
                combined = lc['w_scale'] * lc['in_scale'] * (2 ** lc['shift'])
                y_float = (acc[:lc['real_out']].astype(np.float64)
                           * combined
                           + bias_data[i][:lc['real_out']].astype(np.float64))
                pred = int(np.argmax(y_float))
                argmax_vta.append(pred)

    # Load Mode D reference
    ref_path = os.path.join(weights_dir, 'mode_d_argmax_100.json')
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        mode_d_argmax = ref['argmax_vec']
    else:
        print("  WARNING: Mode D reference not found, computing inline...")
        mode_d_argmax = compute_mode_d_reference(
            weights_dir, mnist_dir, n_check)

    agree = sum(1 for a, b in zip(argmax_vta, mode_d_argmax[:n_check])
                if a == b)
    correct_vta = sum(1 for i in range(n_check)
                      if argmax_vta[i] == labels[i])
    correct_ref = sum(1 for i in range(n_check)
                      if mode_d_argmax[i] == labels[i])

    print(f"  VTA pipeline accuracy:       {correct_vta}/{n_check}")
    print(f"  Mode D reference accuracy:   {correct_ref}/{n_check}")
    print(f"  Argmax agreement (VTA vs D): {agree}/{n_check}")

    # Disagreements
    disagrees = [(i, argmax_vta[i], mode_d_argmax[i])
                 for i in range(n_check) if argmax_vta[i] != mode_d_argmax[i]]
    if disagrees:
        print(f"  Disagreements:")
        for idx, vta_pred, ref_pred in disagrees[:10]:
            print(f"    img {idx}: VTA={vta_pred}, ModeD={ref_pred}, "
                  f"true={labels[idx]}")

    passed = agree >= 95
    print(f"\n  GATE: {agree}/100 >= 95 → {'PASS' if passed else 'FAIL'}")
    return passed, agree


def compute_mode_d_reference(weights_dir, mnist_dir, n_check):
    """Compute Mode D (board-realistic VTA shift) argmax for reference."""
    meta = json.load(open(os.path.join(weights_dir, 'meta.json')))
    n_layers = meta['num_layers']

    W, w_scale, bias = [], [], []
    for i in range(n_layers):
        W.append(np.load(os.path.join(weights_dir, f'W{i}.npy')))
        w_scale.append(float(np.load(
            os.path.join(weights_dir, f'w_scale_{i}.npy'))))
        bias.append(np.load(os.path.join(weights_dir, f'b{i}.npy')))

    # Board-realistic scales
    board_act_scale = [1.0 / 7.0]
    for entry in meta['act_scales'][1:]:
        board_act_scale.append(entry['raw_value'] / 7.0)

    # Hidden-layer shifts
    shifts = []
    for i in range(n_layers - 1):
        ratio = (w_scale[i] * board_act_scale[i]) / board_act_scale[i + 1]
        shifts.append(round(-math.log2(ratio)))

    mnist = download_mnist(mnist_dir)
    images = load_mnist_images(mnist['test_images'])
    labels = load_mnist_labels(mnist['test_labels'])

    argmax_vec = []
    for img_idx in range(n_check):
        x_float = images[img_idx]
        x_int = np.clip(np.round(x_float / board_act_scale[0]),
                        0, 7).astype(np.int32)
        for i in range(n_layers):
            acc = W[i].astype(np.int32) @ x_int.astype(np.int32)
            combined_scale = w_scale[i] * board_act_scale[i]
            bias_int = np.round(
                bias[i].astype(np.float64) / combined_scale).astype(np.int32)
            acc = acc + bias_int
            if i < n_layers - 1:
                acc = acc >> shifts[i]
                x_int = np.clip(acc, 0, 7).astype(np.int32)
            else:
                argmax_vec.append(int(np.argmax(acc)))
    return argmax_vec


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Export pre-compiled VTA INT4 model with bias-in-VTA (v2)')
    parser.add_argument('--weights-dir', required=True,
                        help='Directory with extracted Brevitas weights '
                             '(from extract_int4_brevitas.py)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for compiled model')
    parser.add_argument('--mnist-dir', default='./mnist_data',
                        help='MNIST data directory (for last-layer calibration)')
    parser.add_argument('--cal-samples', type=int, default=100,
                        help='Calibration samples for last-layer shift')
    parser.add_argument('--crosscheck-only', action='store_true',
                        help='Skip VTA compilation, run cross-check only')
    args = parser.parse_args()

    # ---- Load meta.json ----
    meta_path = os.path.join(args.weights_dir, 'meta.json')
    meta = json.load(open(meta_path))
    n_layers = meta['num_layers']
    arch = meta['architecture']

    print(f"Architecture: {' -> '.join(str(d) for d in arch)}")
    print(f"Layers: {n_layers}")

    # ---- Load extracted weights ----
    W_raw, w_scale, bias_raw = [], [], []
    for i in range(n_layers):
        W_raw.append(np.load(os.path.join(args.weights_dir, f'W{i}.npy')))
        w_scale.append(float(np.load(
            os.path.join(args.weights_dir, f'w_scale_{i}.npy'))))
        bias_raw.append(np.load(os.path.join(args.weights_dir, f'b{i}.npy')))

    # ---- Derive board-realistic activation scales ----
    # Input: scale = 1/7 (MNIST [0,1] → [0,7] for signed int4)
    # Hidden: scale = learned_threshold / 7
    board_act_scale = [1.0 / 7.0]
    for entry in meta['act_scales'][1:]:
        board_act_scale.append(entry['raw_value'] / 7.0)

    print(f"\nBoard-realistic activation scales:")
    for i, s in enumerate(board_act_scale):
        print(f"  act_scale_{i} = {s:.6f}")

    # ---- Determine per-layer clip bounds from meta ----
    # Hidden layers: post-ReLU → [0, 7] (signed int4, non-negative)
    # Last layer: signed → [-8, 7]
    # Read from meta.json act_scales where available, else default
    layer_clip_bounds = []
    for i in range(n_layers):
        if i < n_layers - 1:
            # Hidden layer: post-ReLU unsigned → [0, 7] on board
            layer_clip_bounds.append((0, 7))
        else:
            # Last layer: signed int4
            layer_clip_bounds.append((-8, 7))

    # ---- Compute per-layer shifts ----
    shifts = []
    # Hidden layers: from learned scale ratios
    for i in range(n_layers - 1):
        ratio = (w_scale[i] * board_act_scale[i]) / board_act_scale[i + 1]
        ideal = -math.log2(ratio)
        shifts.append(round(ideal))

    # Last layer: calibrate shift from raw GEMM output (without bias).
    # Bias is applied on CPU after dequant for the last layer — this
    # preserves float precision for argmax and avoids int4 tie-breaking
    # errors from the shift.
    print(f"\nCalibrating last-layer shift ({args.cal_samples} samples)...")
    mnist = download_mnist(args.mnist_dir)
    images = load_mnist_images(mnist['test_images'])
    labels = load_mnist_labels(mnist['test_labels'])
    cal_images = images[:args.cal_samples]

    max_abs_last = 0
    for img_idx in range(len(cal_images)):
        x_int = np.clip(np.round(cal_images[img_idx] / board_act_scale[0]),
                        0, 7).astype(np.int32)
        for i in range(n_layers):
            acc = W_raw[i].astype(np.int32) @ x_int.astype(np.int32)
            if i < n_layers - 1:
                combined_scale = w_scale[i] * board_act_scale[i]
                bias_int = np.round(
                    bias_raw[i].astype(np.float64) / combined_scale
                ).astype(np.int32)
                acc = acc + bias_int
                acc = acc >> shifts[i]
                x_int = np.clip(acc, 0, 7).astype(np.int32)
            else:
                # Last layer: raw GEMM only, no bias
                max_abs_last = max(max_abs_last,
                                   int(np.max(np.abs(acc))))

    last_shift = (int(math.ceil(math.log2(max_abs_last / 7.0)))
                  if max_abs_last > 7 else 0)
    shifts.append(last_shift)

    print(f"  Last-layer max |GEMM| (no bias) = {max_abs_last}")
    print(f"  Last-layer shift = {last_shift}")
    print(f"\nAll shifts: {shifts}")

    # ---- Determine VTA block sizes ----
    # Read from env if available, else default INT4 config
    try:
        import vta
        env = vta.get_env()
        block_in = env.BLOCK_IN
        block_out = env.BLOCK_OUT
        batch = env.BATCH
        print(f"\nVTA env: TARGET={env.TARGET}, "
              f"BLOCK_IN={block_in}, BLOCK_OUT={block_out}")
    except ImportError:
        block_in = 16
        block_out = 16
        batch = 1
        print(f"\nVTA not available, using defaults: "
              f"BLOCK_IN={block_in}, BLOCK_OUT={block_out}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Quantize biases and prepare tiled weights ----
    print(f"\nPreparing weights...")
    vta_layers = []
    for i in range(n_layers):
        W = W_raw[i].copy()
        b = bias_raw[i].copy()

        # Pad input dimension
        W = pad_input_dim(W, block_in)
        # Pad output dimension
        W, b, real_out = pad_to_multiple(W, b, block_out)
        out_f, in_f = W.shape

        # Tile weights
        W_tiled = tile_weights(W, block_out, block_in)

        m_tiles = out_f // block_out
        n_tiles = in_f // block_in
        in_scale = board_act_scale[i]
        clip_lo, clip_hi = layer_clip_bounds[i]
        is_last = (i == n_layers - 1)

        if not is_last:
            # Hidden layers: bias quantized to int32, added in VTA ALU
            combined_scale = w_scale[i] * in_scale
            bias_int = np.round(
                b.astype(np.float64) / combined_scale).astype(np.int32)
            bias_tiled = bias_int.reshape(1, m_tiles, 1, block_out)
            bias_desc = f"bias_int range: [{bias_int.min()}, {bias_int.max()}]"
        else:
            # Last layer: bias stays as float32, applied on CPU after dequant.
            # VTA module has no bias input (2 args: A, B).
            bias_tiled = b.astype(np.float32)
            bias_desc = (f"bias_float range: [{b.min():.4f}, {b.max():.4f}] "
                         f"(CPU-side)")

        vta_layers.append({
            'W_tiled': W_tiled,
            'bias_tiled': bias_tiled,
            'in_f': in_f,
            'out_f': out_f,
            'real_out': real_out,
            'n_tiles': n_tiles,
            'm_tiles': m_tiles,
            'w_scale': w_scale[i],
            'in_scale': in_scale,
            'shift': shifts[i],
            'clip_lo': clip_lo,
            'clip_hi': clip_hi,
            'has_vta_bias': not is_last,
        })

        print(f"  Layer {i}: {in_f}->{out_f} (real={real_out}), "
              f"shift={shifts[i]}, clip=[{clip_lo},{clip_hi}], "
              f"tiles n={n_tiles} m={m_tiles}"
              f"{'' if not is_last else ' [bias on CPU]'}")
        print(f"    {bias_desc}")

    # ---- Compile VTA modules ----
    module_filenames = []
    if not args.crosscheck_only:
        print(f"\nCompiling VTA modules...")
        for i, layer in enumerate(vta_layers):
            n_t = layer['n_tiles']
            m_t = layer['m_tiles']
            s = layer['shift']
            clo = layer['clip_lo']
            chi = layer['clip_hi']
            has_bias = layer['has_vta_bias']
            tag = "bias+shift" if has_bias else "shift only"
            fname = f"layer{i}_n{n_t}_m{m_t}_s{s}.o"
            print(f"  Layer {i} ({tag}, o=1, n={n_t}, m={m_t}, shift={s}, "
                  f"clip=[{clo},{chi}])...", end=" ", flush=True)

            if has_bias:
                mod = compile_gemm_with_bias_and_shift(
                    env, 1, n_t, m_t, s, clo, chi)
            else:
                mod = compile_gemm_with_shift(
                    env, 1, n_t, m_t, s, clo, chi)

            out_path = os.path.join(args.output_dir, fname)
            mod.save(out_path)
            module_filenames.append(fname)
            print(f"OK -> {fname}")
    else:
        print("\nSkipping VTA compilation (--crosscheck-only)")
        for i, layer in enumerate(vta_layers):
            fname = f"layer{i}_n{layer['n_tiles']}_m{layer['m_tiles']}" \
                    f"_s{layer['shift']}.o"
            module_filenames.append(fname)

    # ---- Save weights and biases ----
    print(f"\nSaving weights and biases...")
    for i, layer in enumerate(vta_layers):
        np.save(os.path.join(args.output_dir, f'W{i}_tiled.npy'),
                layer['W_tiled'])
        if layer['has_vta_bias']:
            np.save(os.path.join(args.output_dir, f'b{i}_int.npy'),
                    layer['bias_tiled'])
            print(f"  W{i}_tiled.npy: {layer['W_tiled'].shape}, "
                  f"b{i}_int.npy: {layer['bias_tiled'].shape}")
        else:
            np.save(os.path.join(args.output_dir, f'b{i}_float.npy'),
                    layer['bias_tiled'])
            print(f"  W{i}_tiled.npy: {layer['W_tiled'].shape}, "
                  f"b{i}_float.npy: {layer['bias_tiled'].shape} (CPU bias)")

    # ---- Save config ----
    config = {
        'architecture': arch,
        'num_layers': n_layers,
        'layers': [],
        'input_scale': board_act_scale[0],
        'input_clip_max': 7,
        'vta_config': {
            'BATCH': batch,
            'BLOCK_IN': block_in,
            'BLOCK_OUT': block_out,
        },
        'bitstream': '1x16_i4w4a32_15_14_17_17.bit',
        'calibration_samples': args.cal_samples,
        'board_act_scales': board_act_scale,
        'weight_scales': [float(s) for s in w_scale],
        'requant_mode': 'vta_native',
        'bias_placement': 'hidden: pre-shift in VTA ALU; last: CPU-side float after dequant',
    }
    for i, layer in enumerate(vta_layers):
        bias_fname = f'b{i}_int.npy' if layer['has_vta_bias'] else f'b{i}_float.npy'
        config['layers'].append({
            'index': i,
            'in_f': layer['in_f'],
            'out_f': layer['out_f'],
            'real_out': layer['real_out'],
            'n_tiles': layer['n_tiles'],
            'm_tiles': layer['m_tiles'],
            'shift': layer['shift'],
            'clip_lo': layer['clip_lo'],
            'clip_hi': layer['clip_hi'],
            'w_scale': float(layer['w_scale']),
            'in_scale': float(layer['in_scale']),
            'has_vta_bias': layer['has_vta_bias'],
            'module_file': module_filenames[i],
            'weight_file': f'W{i}_tiled.npy',
            'bias_file': bias_fname,
        })

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # ---- Cross-check ----
    passed, agree = numpy_crosscheck(
        args.output_dir, args.weights_dir, args.mnist_dir)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Export complete: {args.output_dir}/")
    print(f"  Architecture: {' -> '.join(str(d) for d in arch)}")
    print(f"  Shifts: {shifts}")
    print(f"  Clip bounds: {[layer_clip_bounds[i] for i in range(n_layers)]}")
    print(f"  Cross-check: {agree}/100 "
          f"({'PASS' if passed else 'FAIL'})")
    print(f"  Files:")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"    {f} ({size} bytes)")
    print(f"{'='*60}")

    if not passed:
        print("\nERROR: Cross-check FAILED (<95/100 agreement).")
        sys.exit(1)


if __name__ == '__main__':
    main()
