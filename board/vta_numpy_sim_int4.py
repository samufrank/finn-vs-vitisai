#!/usr/bin/env python3
"""Host-side numpy simulator for VTA INT4 MNIST inference.

Uses Brevitas's learned scales (extracted by extract_int4_brevitas.py),
NOT recomputed max-abs scales. Runs four simulation modes:

  Mode A — "ceiling" (float requant, Brevitas-native [0,15] activations)
  Mode B — "VTA-faithful" (power-of-two shift, Brevitas-native [0,15] activations)
  Mode C — "board-realistic ceiling" (float requant, signed int4 [0,7] activations)
  Mode D — "board-realistic VTA" (shift, signed int4 [0,7] activations)

Modes A/B assume unsigned 4-bit activations [0,15] as Brevitas trained.
Modes C/D constrain to [0,7] which is what VTA's signed int4 can represent
for non-negative post-ReLU values. Activation scales are adjusted:
  board_act_scale = brevitas_threshold / 7  (not /15)

Usage:
    cd ~/dev/CEN571-final
    python3 finn-vs-vitisai/board/vta_numpy_sim_int4.py

Requires only numpy and standard library (uses raw MNIST loader, no torchvision).
"""
import os
import sys
import json
import math
import struct
import gzip
import numpy as np

WEIGHTS_DIR = os.path.expanduser(
    '~/dev/CEN571-final/tvm-v0.12.0/vta_mnist_weights_int4_v2')

# ---------- MNIST loader (same as export scripts) ----------

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


# ---------- Load extracted tensors ----------

def load_weights(weights_dir):
    """Load all extracted numpy arrays and meta."""
    meta = json.load(open(os.path.join(weights_dir, 'meta.json')))
    n_layers = meta['num_layers']

    W = []
    w_scale = []
    act_scale = []
    bias = []

    for i in range(n_layers):
        W.append(np.load(os.path.join(weights_dir, f'W{i}.npy')))
        w_scale.append(float(np.load(os.path.join(weights_dir, f'w_scale_{i}.npy'))))
        bias.append(np.load(os.path.join(weights_dir, f'b{i}.npy')))

    # act_scale_0 = input, act_scale_1 = after layer 0's ReLU, act_scale_2 = after layer 1's ReLU
    for i in range(n_layers):
        act_scale.append(float(np.load(os.path.join(weights_dir, f'act_scale_{i}.npy'))))

    return W, w_scale, act_scale, bias, meta


def derive_board_scales(act_scale, meta):
    """Derive board-realistic scales for signed int4 [0,7].

    Brevitas learned thresholds (stored in meta) map to [0,15].
    For VTA signed int4, post-ReLU values fit in [0,7].
    Board scale = threshold / 7 (preserves dynamic range, coarser steps).
    """
    board_act_scale = []
    # act_scale_0 (input): Brevitas had 1/15, board needs 1/7
    board_act_scale.append(1.0 / 7.0)

    # act_scale_1, act_scale_2: threshold / 7 instead of threshold / 15
    for entry in meta.get('act_scales', [])[1:]:
        threshold = entry['raw_value']
        board_act_scale.append(threshold / 7.0)

    return board_act_scale


# ---------- Simulation core ----------

def compute_shift_info(w_scale_i, in_scale, out_scale):
    """Compute shift bits and rounding error for power-of-two approximation."""
    ratio = (w_scale_i * in_scale) / out_scale
    if ratio <= 0:
        return 0, 1.0, True
    ideal_shift = -math.log2(ratio)
    shift = round(ideal_shift)
    approx_ratio = 2.0 ** (-shift)
    rel_error = abs(ratio - approx_ratio) / ratio
    negative = shift < 0
    return shift, rel_error, negative


def simulate(images, labels, W, w_scale, act_scale, bias, clip_max, use_shift):
    """Unified simulation.

    clip_max: upper clip bound for activations (15 for Brevitas-native, 7 for board)
    use_shift: False = Mode A/C (float requant), True = Mode B/D (power-of-two shift)
    """
    n_layers = len(W)
    n_images = len(images)
    correct = 0

    # Precompute shifts for hidden layers
    shifts = []
    for i in range(n_layers - 1):
        shift, _, _ = compute_shift_info(w_scale[i], act_scale[i], act_scale[i + 1])
        shifts.append(shift)

    for img_idx in range(n_images):
        x_float = images[img_idx]

        # Input quantization: [0, clip_max], unsigned for MNIST [0,1]
        x_int = np.clip(np.round(x_float / act_scale[0]), 0, clip_max).astype(np.int32)

        for i in range(n_layers):
            acc = W[i].astype(np.int32) @ x_int.astype(np.int32)

            if use_shift:
                # Mode B/D: integer bias, then shift, then clip
                combined_scale = w_scale[i] * act_scale[i]
                bias_int = np.round(bias[i].astype(np.float64) / combined_scale).astype(np.int32)
                acc = acc + bias_int

                if i < n_layers - 1:
                    s = shifts[i]
                    y_int = acc >> s if s >= 0 else acc << (-s)
                    x_int = np.clip(y_int, 0, clip_max).astype(np.int32)
                else:
                    pred = int(np.argmax(acc))
            else:
                # Mode A/C: float domain bias + requant
                combined_scale = w_scale[i] * act_scale[i]
                float_acc = acc.astype(np.float64) * combined_scale + bias[i].astype(np.float64)

                if i < n_layers - 1:
                    float_acc = np.maximum(float_acc, 0.0)
                    x_int = np.clip(np.round(float_acc / act_scale[i + 1]),
                                    0, clip_max).astype(np.int32)
                else:
                    pred = int(np.argmax(float_acc))

        if pred == labels[img_idx]:
            correct += 1

    return correct, n_images


def print_shift_diagnostics(w_scale, act_scale, label):
    """Print per-layer shift info."""
    n_layers = len(w_scale)
    any_negative = False
    print(f"\n  Per-layer shift diagnostics ({label}):")
    for i in range(n_layers - 1):
        ratio = (w_scale[i] * act_scale[i]) / act_scale[i + 1]
        shift, rel_err, neg = compute_shift_info(w_scale[i], act_scale[i], act_scale[i + 1])
        flag = " *** NEGATIVE SHIFT ***" if neg else ""
        print(f"    Layer {i}: ratio={ratio:.6f}, shift={shift}, "
              f"2^(-shift)={2.0**(-shift):.6f}, error={rel_err*100:.2f}%{flag}")
        if neg:
            any_negative = True
    print(f"    Layer {n_layers-1}: no shift (last layer, argmax of int32 accumulator)")
    return any_negative


def print_signedness_table(act_scale, clip_max, label):
    """Item 3: Print activation signedness, bit width, clip bounds per layer."""
    names = ['input', 'post-ReLU layer 0', 'post-ReLU layer 1']
    print(f"\n  Activation quantizer config ({label}):")
    print(f"  {'Quantizer':<25} {'Signed':<8} {'Bits':<6} {'Clip':<12} {'Scale':<12}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*12} {'-'*12}")
    for i, name in enumerate(names):
        signed = 'no' if clip_max > 7 else 'yes (hw)'
        print(f"  {name:<25} {signed:<8} {'4':<6} {'[0,'+str(clip_max)+']':<12} {act_scale[i]:<12.6f}")
    print(f"  {'output layer 2':<25} {'n/a':<8} {'32':<6} {'int32':<12} {'n/a':<12}")


def main():
    print("=" * 70)
    print("VTA INT4 MNIST Numpy Simulator (Brevitas learned scales)")
    print("=" * 70)

    # Load weights
    print(f"\nLoading weights from {WEIGHTS_DIR}/")
    W, w_scale, act_scale, bias, meta = load_weights(WEIGHTS_DIR)
    board_act_scale = derive_board_scales(act_scale, meta)

    for i in range(len(W)):
        print(f"  Layer {i}: W{W[i].shape}, w_scale={w_scale[i]:.6f}, bias={bias[i].shape}")
    print(f"\n  Brevitas-native act_scales (threshold/15, clip [0,15]):")
    for i, s in enumerate(act_scale):
        print(f"    act_scale_{i} = {s:.6f}")
    print(f"  Board-realistic act_scales (threshold/7, clip [0,7]):")
    for i, s in enumerate(board_act_scale):
        print(f"    act_scale_{i} = {s:.6f}")

    # Load MNIST
    print("\nLoading MNIST test set...")
    mnist_dir = os.path.join(os.path.dirname(WEIGHTS_DIR), 'mnist_data')
    if not os.path.exists(mnist_dir):
        mnist_dir = './mnist_data'
    mnist = download_mnist(mnist_dir)
    images = load_mnist_images(mnist['test_images'])
    labels = load_mnist_labels(mnist['test_labels'])
    print(f"  {len(images)} images, pixel range: [{images.min():.3f}, {images.max():.3f}]")

    # ================================================================
    # Item 3: Signedness table
    # ================================================================
    print("\n" + "=" * 70)
    print("ITEM 3: ACTIVATION SIGNEDNESS PER LAYER")
    print("=" * 70)
    print_signedness_table(act_scale, 15, "Brevitas-native")
    print_signedness_table(board_act_scale, 7, "board-realistic")
    print("\n  Weight quantizers: all signed 4-bit [-7, 7] (unchanged)")

    # ================================================================
    # Modes A & B: Brevitas-native (unsigned [0,15])
    # ================================================================
    print("\n" + "=" * 70)
    print("MODE A: Ceiling / float requant, Brevitas-native [0,15]")
    print("=" * 70)
    correct_a, total_a = simulate(images, labels, W, w_scale, act_scale, bias,
                                   clip_max=15, use_shift=False)
    acc_a = 100.0 * correct_a / total_a
    print(f"  Accuracy: {acc_a:.2f}% ({correct_a}/{total_a})")
    neg_a = print_shift_diagnostics(w_scale, act_scale, "Brevitas-native")

    print("\n" + "=" * 70)
    print("MODE B: VTA shift, Brevitas-native [0,15]")
    print("=" * 70)
    correct_b, total_b = simulate(images, labels, W, w_scale, act_scale, bias,
                                   clip_max=15, use_shift=True)
    acc_b = 100.0 * correct_b / total_b
    print(f"  Accuracy: {acc_b:.2f}% ({correct_b}/{total_b})")

    # ================================================================
    # Modes C & D: Board-realistic (signed int4, [0,7])
    # ================================================================
    print("\n" + "=" * 70)
    print("MODE C: Ceiling / float requant, board-realistic [0,7]")
    print("=" * 70)
    correct_c, total_c = simulate(images, labels, W, w_scale, board_act_scale, bias,
                                   clip_max=7, use_shift=False)
    acc_c = 100.0 * correct_c / total_c
    print(f"  Accuracy: {acc_c:.2f}% ({correct_c}/{total_c})")
    neg_c = print_shift_diagnostics(w_scale, board_act_scale, "board-realistic")

    print("\n" + "=" * 70)
    print("MODE D: VTA shift, board-realistic [0,7]")
    print("=" * 70)
    correct_d, total_d = simulate(images, labels, W, w_scale, board_act_scale, bias,
                                   clip_max=7, use_shift=True)
    acc_d = 100.0 * correct_d / total_d
    print(f"  Accuracy: {acc_d:.2f}% ({correct_d}/{total_d})")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mode A (float requant, [0,15]):     {acc_a:.2f}%")
    print(f"  Mode B (VTA shift, [0,15]):          {acc_b:.2f}%")
    print(f"  Mode C (float requant, [0,7]):       {acc_c:.2f}%")
    print(f"  Mode D (VTA shift, [0,7]):           {acc_d:.2f}%")
    print(f"  A-B gap (shift error, [0,15]):       {acc_a - acc_b:.2f}%")
    print(f"  A-C gap (clip bound, float):         {acc_a - acc_c:.2f}%")
    print(f"  C-D gap (shift error, [0,7]):        {acc_c - acc_d:.2f}%")
    print(f"  A-D gap (total board penalty):       {acc_a - acc_d:.2f}%")

    print(f"\n  Shift values (Brevitas-native [0,15]):")
    for i in range(len(W) - 1):
        s, e, n = compute_shift_info(w_scale[i], act_scale[i], act_scale[i + 1])
        print(f"    Layer {i}: shift={s}, error={e*100:.2f}%{' NEGATIVE' if n else ''}")
    print(f"  Shift values (board-realistic [0,7]):")
    for i in range(len(W) - 1):
        s, e, n = compute_shift_info(w_scale[i], board_act_scale[i], board_act_scale[i + 1])
        print(f"    Layer {i}: shift={s}, error={e*100:.2f}%{' NEGATIVE' if n else ''}")

    any_neg = neg_a or neg_c
    print(f"\n  Negative shifts: {'YES — needs fix' if any_neg else 'None'}")

    print(f"\n  NOTES:")
    print(f"  - Brevitas trains with Uint4 [0,15] activations.")
    print(f"  - VTA signed int4 can only represent [0,7] for non-negative values.")
    print(f"  - Modes C/D use threshold/7 scales to preserve dynamic range in [0,7].")
    print(f"  - Board-side bias is added POST-shift in float (benchmark.py pattern).")
    print(f"    Sim adds bias PRE-shift in int32 (mathematically correct).")
    print(f"    Export script should include bias ADD in VTA ALU before SHR.")
    print("=" * 70)


if __name__ == '__main__':
    main()
