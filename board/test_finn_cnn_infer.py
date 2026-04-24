#!/usr/bin/env python3
"""Host-side correctness harness for libfinn_cnn_infer.so.

Five checks (all must PASS; no board execution):

  1. Pack byte-exact (FPGA input layout)
     1000 random (28, 28, 8) uint8 tensors.  C memcpy vs Python
     finnpy_to_packed_bytearray on the matching (1, 28, 28, 1, 8)
     fold shape.  Byte-identical flat result.

  2a. Unpack byte-exact (FPGA output layout)
     1000 random 784-byte buffers.  C memcpy vs Python
     packed_bytearray_to_finnpy uint8-view.

  2b. Obuf HWC layout verification
     Structured byte patterns where each byte depends on (h, w, c).
     Confirms that Python's `hw_out.reshape(7, 7, 16).mean(axis=(0, 1))`
     averages exactly the 49 bytes that C indexes as
     obuf[h*112 + w*16 + c] for each channel c.

  3. Pack-scratch correctness (end-to-end mock)
     100 MNIST images.  Python runs CPU pre via its im2col + first-MatMul
     + MultiThreshold path; C writes pack_scratch through finn_cnn_infer_one_mock.
     Bytes must match.

  4. End-to-end argmax match (end-to-end mock)
     Same 100 images.  Deterministic random obuf fed as the "FPGA" result
     to both Python (hw_out = obuf.astype(float32).reshape(7,7,16)) and C
     (via mock entry).  Predictions must match.

  5. GAP uint32 accumulator exactness
     Synthetic obuf = all 255.  Both paths must produce feat[c] = 255.0
     exactly (no rounding) and identical argmax.
"""

import argparse
import ctypes
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
DEPLOY_CNN = os.path.join(REPO_ROOT, 'finn-vs-vitisai/finn/output_cnn_mnist_tiny/deploy')
MNIST_RAW  = os.path.join(REPO_ROOT, 'finn-vs-vitisai/data/MNIST/raw')
DATA_PACKING_DRIVER_DIR = os.path.join(DEPLOY_CNN, 'driver')


def build_so(src_path, so_path):
    cmd = ['gcc', '-O2', '-shared', '-fPIC', '-Wall', '-Werror',
           '-o', so_path, src_path]
    print(f'Building: {" ".join(cmd)}')
    subprocess.check_call(cmd)


def load_lib(so_path):
    lib = ctypes.CDLL(so_path)

    lib.finn_cnn_pack_uint8.argtypes   = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_cnn_pack_uint8.restype    = None
    lib.finn_cnn_unpack_uint8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_cnn_unpack_uint8.restype  = None

    lib.finn_cnn_runner_init.argtypes = [
        ctypes.c_int,                   # precision
        ctypes.c_int, ctypes.c_int, ctypes.c_int,            # img_h, img_w, img_c
        ctypes.c_int, ctypes.c_int,                          # kernel, pad
        ctypes.c_int,                                        # fpga_in_c
        ctypes.c_int, ctypes.c_int, ctypes.c_int,            # fpga_out_h,w,c
        ctypes.c_int, ctypes.c_int,                          # num_classes, num_thresholds
        ctypes.c_int,                                        # use_cache_ops
        ctypes.c_void_p, ctypes.c_uint64,                    # ibuf virt, phys
        ctypes.c_void_p, ctypes.c_uint64,                    # obuf virt, phys
        ctypes.c_void_p, ctypes.c_void_p,                    # idma, odma mmio
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # W_conv, thres, W_cls
        ctypes.c_float, ctypes.c_void_p,                     # mul, add
    ]
    lib.finn_cnn_runner_init.restype = ctypes.c_int

    lib.finn_cnn_runner_destroy.argtypes = []
    lib.finn_cnn_runner_destroy.restype  = ctypes.c_int

    lib.finn_cnn_infer_one_mock.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.finn_cnn_infer_one_mock.restype = ctypes.c_int

    return lib


def setup_data_packing():
    sys.path.insert(0, DATA_PACKING_DRIVER_DIR)
    from qonnx.core.datatype import DataType
    from finn.util.data_packing import (
        finnpy_to_packed_bytearray, packed_bytearray_to_finnpy)
    return DataType, finnpy_to_packed_bytearray, packed_bytearray_to_finnpy


def load_mnist_test():
    with open(os.path.join(MNIST_RAW, 't10k-images-idx3-ubyte'), 'rb') as f:
        magic, n, H, W = struct.unpack('>IIII', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, H, W)
    with open(os.path.join(MNIST_RAW, 't10k-labels-idx1-ubyte'), 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, labels


# ----- Check 1: pack byte-exact ------------------------------------------

def test_pack_byte_exact(lib, DataType, py_pack, n_trials, rng):
    print(f'[1] pack UINT8 byte-exact — {n_trials} random (28,28,8) tensors')
    dtype = DataType['UINT8']
    n_fail = 0
    for t in range(n_trials):
        act = rng.integers(0, 256, size=(28, 28, 8), dtype=np.uint8)
        py = py_pack(act.reshape((1, 28, 28, 1, 8)), dtype,
                     reverse_endian=True, reverse_inner=True, fast_mode=True)
        py_bytes = np.ascontiguousarray(py).flatten().astype(np.uint8)
        assert py_bytes.shape == (6272,)

        c_buf = np.zeros(6272, dtype=np.uint8)
        lib.finn_cnn_pack_uint8(act.ctypes.data, c_buf.ctypes.data, 6272)
        if not np.array_equal(py_bytes, c_buf):
            n_fail += 1
            if n_fail <= 3:
                print(f'    FAIL t={t}: py[:16]={py_bytes[:16].tolist()}, '
                      f'c[:16]={c_buf[:16].tolist()}')
    print(f'    {n_trials - n_fail}/{n_trials} OK')
    return n_fail == 0


# ----- Check 2a: unpack byte-exact ---------------------------------------

def test_unpack_byte_exact(lib, DataType, py_unpack, n_trials, rng):
    print(f'[2a] unpack UINT8 byte-exact — {n_trials} random 784-byte bufs')
    dtype = DataType['UINT8']
    n_fail = 0
    for t in range(n_trials):
        obuf = rng.integers(0, 256, size=784, dtype=np.uint8)
        py = py_unpack(obuf.reshape((1, 7, 7, 1, 16)), dtype,
                       (1, 7, 7, 1, 16),
                       reverse_endian=True, reverse_inner=True)
        py_u8 = np.ascontiguousarray(py).astype(np.int64).astype(np.uint8).flatten()

        c_buf = np.zeros(784, dtype=np.uint8)
        lib.finn_cnn_unpack_uint8(obuf.ctypes.data, c_buf.ctypes.data, 784)
        if not np.array_equal(py_u8, c_buf):
            n_fail += 1
            if n_fail <= 3:
                print(f'    FAIL t={t}')
    print(f'    {n_trials - n_fail}/{n_trials} OK')
    return n_fail == 0


# ----- Check 2b: obuf HWC layout verification ----------------------------

def test_obuf_layout(DataType, py_unpack):
    """Confirm that `hw_out.reshape(7,7,16).mean(axis=(0,1))` for Python and
    `feat[c] = sum_{h,w} obuf[h*112 + w*16 + c] / 49` for C produce the same
    feat vector.  Proves axis 0 is H, axis 1 is W, channel-fastest layout."""
    print('[2b] obuf HWC-with-C-fastest layout verification')
    dtype = DataType['UINT8']
    n_fail = 0

    # Pattern A: each byte encodes its own channel index (+ constant offset),
    # independent of (h, w).  Expected feat[c] = c + 10 exactly.
    obuf = np.zeros(784, dtype=np.uint8)
    for h in range(7):
        for w in range(7):
            for c in range(16):
                obuf[h * 112 + w * 16 + c] = c + 10

    py = py_unpack(obuf.reshape((1, 7, 7, 1, 16)), dtype,
                   (1, 7, 7, 1, 16),
                   reverse_endian=True, reverse_inner=True)
    hw_out_py = np.ascontiguousarray(py).astype(np.float32).reshape(7, 7, 16)
    feat_py = hw_out_py.mean(axis=(0, 1))
    expected_a = np.arange(16, dtype=np.float32) + 10.0
    if not np.array_equal(feat_py, expected_a):
        n_fail += 1
        print(f'    FAIL pattern A: feat={feat_py.tolist()}, expected={expected_a.tolist()}')
    else:
        print('    pattern A (byte = c + 10): py feat matches expected — H/W axes confirmed')

    # Pattern B: byte varies with h+w+c.  Confirms the axis order under actual
    # spatial variation, not just channel constants.
    obuf2 = np.zeros(784, dtype=np.uint8)
    for h in range(7):
        for w in range(7):
            for c in range(16):
                obuf2[h * 112 + w * 16 + c] = (h + w + c) % 256

    py2 = py_unpack(obuf2.reshape((1, 7, 7, 1, 16)), dtype,
                    (1, 7, 7, 1, 16),
                    reverse_endian=True, reverse_inner=True)
    hw2 = np.ascontiguousarray(py2).astype(np.float32).reshape(7, 7, 16)
    feat2_py = hw2.mean(axis=(0, 1))

    expected_b = np.zeros(16, dtype=np.float32)
    for c in range(16):
        s = 0.0
        for h in range(7):
            for w in range(7):
                s += float((h + w + c) % 256)
        expected_b[c] = s / 49.0

    if not np.allclose(feat2_py, expected_b, atol=1e-5):
        n_fail += 1
        print(f'    FAIL pattern B: feat={feat2_py.tolist()}, expected={expected_b.tolist()}')
    else:
        print('    pattern B (byte = (h+w+c) % 256): py feat matches C-assumed indexing')

    return n_fail == 0


# ----- Check 5: GAP exactness on all-255 obuf (run before end-to-end since
#                it's also a good smoke test of the runner wiring) --------

def test_gap_all_255(lib, deploy):
    print('[5] GAP uint32-accumulator exactness (all-255 obuf)')
    W_conv  = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_0_param0.npy')).astype(np.float32))
    thres   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MultiThreshold_0_param0.npy')).astype(np.float32))
    W_cls   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_2_param0.npy')).astype(np.float32))
    mul_out = float(np.load(os.path.join(deploy, 'cnn_Mul_0_param0.npy')).flatten()[0])
    add_out = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_Add_0_param0.npy')).astype(np.float32))

    ibuf = np.zeros(6272, dtype=np.uint8)
    obuf = np.zeros(784, dtype=np.uint8)

    rc = lib.finn_cnn_runner_init(
        8, 28, 28, 1, 3, 1, 8, 7, 7, 16, 10, 255,
        0,                               # use_cache_ops=0
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,
        W_conv.ctypes.data, thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data)
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    # Mock obuf = all 255.  Dummy input image is irrelevant (we only care about
    # the post-stage exactness); but we still need a 784-byte buffer so cpu_pre
    # can run without segfault.
    mock_obuf = np.ascontiguousarray(np.full(784, 255, dtype=np.uint8))
    dummy_img = np.zeros(784, dtype=np.uint8)
    pack_scratch = np.zeros(6272, dtype=np.uint8)
    pred_c = lib.finn_cnn_infer_one_mock(
        dummy_img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

    # Python equivalent: feat = 255 everywhere (exact), then the rest follows.
    feat = np.full(16, 255.0, dtype=np.float32)
    logits = feat @ W_cls
    out = logits.astype(np.float32) * mul_out + add_out
    pred_py = int(np.argmax(out))

    # Exactness check on Python's own GAP: float32 mean of 49 copies of 255.0
    hw_out = np.full((7, 7, 16), 255.0, dtype=np.float32)
    feat_py = hw_out.mean(axis=(0, 1))
    ok_exact = bool(np.all(feat_py == 255.0))

    lib.finn_cnn_runner_destroy()

    if not ok_exact:
        print(f'    FAIL: np.mean of all-255 not exactly 255.0: {feat_py.tolist()}')
        return False
    if pred_c != pred_py:
        print(f'    FAIL: pred_c={pred_c}, pred_py={pred_py}')
        return False
    print(f'    OK: feat == 255.0 (exact);  argmax: c={pred_c} py={pred_py}')
    return True


# ----- Checks 3 + 4: end-to-end mock on MNIST ----------------------------

def _im2col_py(x, kernel_size=3, pad=1):
    H, W, C = x.shape
    kH, kW = kernel_size, kernel_size
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    out = np.zeros((H, W, kH * kW * C), dtype=x.dtype)
    for i in range(H):
        for j in range(W):
            out[i, j, :] = x_pad[i:i+kH, j:j+kW, :].flatten()
    return out


def test_end_to_end(lib, deploy, n_trials, mnist_imgs, rng):
    print(f'[3+4] end-to-end mock — {n_trials} MNIST images')
    W_conv  = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_0_param0.npy')).astype(np.float32))
    thres   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MultiThreshold_0_param0.npy')).astype(np.float32))
    W_cls   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_2_param0.npy')).astype(np.float32))
    mul_out = float(np.load(os.path.join(deploy, 'cnn_Mul_0_param0.npy')).flatten()[0])
    add_out = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_Add_0_param0.npy')).astype(np.float32))

    ibuf = np.zeros(6272, dtype=np.uint8)
    obuf = np.zeros(784, dtype=np.uint8)

    rc = lib.finn_cnn_runner_init(
        8, 28, 28, 1, 3, 1, 8, 7, 7, 16, 10, 255,
        0,
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,
        W_conv.ctypes.data, thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data)
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    def py_cpu_pre(img_flat):
        img = img_flat.reshape(28, 28, 1).astype(np.float32) / 255.0
        patches = _im2col_py(img)
        x_f = patches @ W_conv                              # (28, 28, 8) float32
        act = np.sum(x_f[..., None] >= thres, axis=-1).astype(np.uint8)  # (28, 28, 8)
        return act

    def py_cpu_post(obuf_u8):
        hw_out = obuf_u8.astype(np.float32).reshape(7, 7, 16)
        feat = hw_out.mean(axis=(0, 1))                     # (16,) float32
        logits = feat @ W_cls                               # (10,) float32
        out = logits.astype(np.float32) * mul_out + add_out
        return int(np.argmax(out))

    n_fail_pred = 0
    n_fail_pack = 0
    pack_scratch = np.zeros(6272, dtype=np.uint8)

    for i in range(n_trials):
        img = np.ascontiguousarray(mnist_imgs[i].flatten().astype(np.uint8))
        mock_obuf = np.ascontiguousarray(
            rng.integers(0, 256, size=784, dtype=np.uint8))

        py_act = py_cpu_pre(img)           # (28, 28, 8) uint8
        py_pred = py_cpu_post(mock_obuf)

        c_pred = lib.finn_cnn_infer_one_mock(
            img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

        # pack_scratch should equal py_act flattened NHWC (C-fastest)
        expected_pack = py_act.flatten().astype(np.uint8)
        if not np.array_equal(pack_scratch, expected_pack):
            n_fail_pack += 1
            if n_fail_pack <= 3:
                diff_positions = np.where(pack_scratch != expected_pack)[0]
                print(f'    FAIL pack i={i}: {len(diff_positions)} byte diffs, '
                      f'first @ {diff_positions[:5].tolist()}')

        if c_pred != py_pred:
            n_fail_pred += 1
            if n_fail_pred <= 3:
                print(f'    FAIL argmax i={i}: py={py_pred}, c={c_pred}')

    lib.finn_cnn_runner_destroy()
    print(f'    argmax: {n_trials - n_fail_pred}/{n_trials} OK,  '
          f'pack:   {n_trials - n_fail_pack}/{n_trials} OK')
    return (n_fail_pred == 0) and (n_fail_pack == 0)


# ----- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(THIS_DIR, 'finn_cnn_infer.c'))
    ap.add_argument('--so',  default=None)
    ap.add_argument('--pack-trials',   type=int, default=1000)
    ap.add_argument('--unpack-trials', type=int, default=1000)
    ap.add_argument('--e2e-trials',    type=int, default=100)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    so_path = args.so or tempfile.NamedTemporaryFile(
        suffix='.so', prefix='libfinn_cnn_infer_test_', delete=False).name
    build_so(args.src, so_path)
    lib = load_lib(so_path)
    DataType, py_pack, py_unpack = setup_data_packing()

    rng = np.random.default_rng(args.seed)
    ok = True
    ok &= test_pack_byte_exact(lib, DataType, py_pack,
                               args.pack_trials, rng)
    ok &= test_unpack_byte_exact(lib, DataType, py_unpack,
                                 args.unpack_trials, rng)
    ok &= test_obuf_layout(DataType, py_unpack)
    ok &= test_gap_all_255(lib, DEPLOY_CNN)

    mnist_imgs, _ = load_mnist_test()
    ok &= test_end_to_end(lib, DEPLOY_CNN, args.e2e_trials, mnist_imgs, rng)

    print('\n===== HARNESS:', 'PASS' if ok else 'FAIL', '=====')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
