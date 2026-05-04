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
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
DEPLOY_CNN_INT8 = os.path.join(REPO_ROOT, 'finn-vs-vitisai/finn/output_cnn_mnist_tiny/deploy')
DEPLOY_CNN_INT4 = os.path.join(REPO_ROOT, 'finn-vs-vitisai/finn/output_cnn_mnist_tiny_int4/deploy')
DEPLOY_CNN_DEEP3_INT8 = os.path.join(
    REPO_ROOT, 'finn-vs-vitisai/finn/size_sweep_runs/cnn_int8_deep_3/deploy')
DEPLOY_CNN_DEEP3_INT4 = os.path.join(
    REPO_ROOT, 'finn-vs-vitisai/finn/size_sweep_runs/cnn_int4_deep_3/deploy')
DEPLOY_CNN_QI_SMALL_INT8 = os.path.join(
    REPO_ROOT, 'finn-vs-vitisai/finn/size_sweep_runs/cnn_int8_small_qi/deploy')
MNIST_RAW       = os.path.join(REPO_ROOT, 'finn-vs-vitisai/data/MNIST/raw')


def deploy_path(precision):
    return DEPLOY_CNN_INT8 if precision == 8 else DEPLOY_CNN_INT4


# ---- Per-precision constants. Computed once and threaded through tests.
# CNN INT4 packs 2 channels per byte both directions (driver/driver.py:
# ishape_packed=(1,28,28,1,4) → 8 in-ch / 4 bytes; oshape_packed=(1,7,7,1,8)
# → 16 out-ch / 8 bytes). INT8 is 1 channel per byte both ways.
def precision_consts(precision):
    assert precision in (8, 4)
    n_in_elems  = 28 * 28 * 8           # H*W*fpga_in_c
    n_out_elems = 7 * 7 * 16            # OH*OW*fpga_out_c
    if precision == 8:
        return {
            'precision': 8,
            'num_thresholds': 255,
            'max_val': 255,
            'ibuf_packed_bytes': n_in_elems,         # 6272
            'obuf_packed_bytes': n_out_elems,        # 784
            'pack_fold_shape':   (1, 28, 28, 1, 8),  # final dim = ch (1 byte ea)
            'unpack_fold_shape': (1,  7,  7, 1, 16),
            'mock_obuf_high':    256,                # rng range upper bound
            'dtype_name':        'UINT8',
        }
    return {
        'precision': 4,
        'num_thresholds': 15,
        'max_val': 15,
        'ibuf_packed_bytes': n_in_elems  // 2,       # 3136
        'obuf_packed_bytes': n_out_elems // 2,       # 392
        'pack_fold_shape':   (1, 28, 28, 1, 8),      # final dim = elements (FINN packs)
        'unpack_fold_shape': (1,  7,  7, 1, 16),
        'mock_obuf_high':    16,
        'dtype_name':        'UINT4',
    }


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
    lib.finn_cnn_pack_uint4.argtypes   = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_cnn_pack_uint4.restype    = None
    lib.finn_cnn_unpack_uint4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_cnn_unpack_uint4.restype  = None

    lib.finn_cnn_runner_init.argtypes = [
        ctypes.c_int, ctypes.c_int,                          # in_precision, out_precision
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
        ctypes.c_int,                                        # cpu_post_maxpool_k
        ctypes.c_int, ctypes.c_int,                          # ibuf_packed_bytes, obuf_packed_bytes
        ctypes.c_int,                                        # partition (0=classic, 1=qi)
        ctypes.c_int,                                        # idt_signed (0=unsigned, 1=signed)
    ]
    lib.finn_cnn_runner_init.restype = ctypes.c_int

    lib.finn_cnn_runner_destroy.argtypes = []
    lib.finn_cnn_runner_destroy.restype  = ctypes.c_int

    lib.finn_cnn_infer_one_mock.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.finn_cnn_infer_one_mock.restype = ctypes.c_int

    return lib


def setup_data_packing(precision):
    """Inject FINN's data_packing module from the matching deploy/driver/.
    The driver bundles qonnx + finn.util.data_packing; both INT8 and INT4
    deploys ship identical helpers, so either works for both precisions —
    we still pick the matching one to stay consistent with the deploy folder."""
    drv_dir = os.path.join(deploy_path(precision), 'driver')
    sys.path.insert(0, drv_dir)
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


# ----- Check 1 (INT4): pack 2-per-byte UINT4 ----------------------------

def test_pack_uint4(lib, DataType, py_pack, n_trials, rng):
    """C finn_cnn_pack_uint4 must match FINN's finnpy_to_packed_bytearray.
    Values 0..15 in each (28, 28, 8) tensor; packed result is 3136 bytes
    (8 channels × 4 bits = 4 bytes/pixel; 28*28*4 = 3136)."""
    print(f'[1] pack UINT4 byte-exact — {n_trials} random (28,28,8) tensors of 0..15')
    dtype = DataType['UINT4']
    n_fail = 0
    for t in range(n_trials):
        act = rng.integers(0, 16, size=(28, 28, 8), dtype=np.uint8)
        py = py_pack(act.reshape((1, 28, 28, 1, 8)), dtype,
                     reverse_endian=True, reverse_inner=True, fast_mode=False)
        py_bytes = np.ascontiguousarray(py).flatten().astype(np.uint8)
        assert py_bytes.shape == (3136,), f'unexpected packed length {py_bytes.shape}'

        c_buf = np.zeros(3136, dtype=np.uint8)
        # C pack takes element count, not byte count.
        lib.finn_cnn_pack_uint4(act.ctypes.data, c_buf.ctypes.data, 28 * 28 * 8)
        if not np.array_equal(py_bytes, c_buf):
            n_fail += 1
            if n_fail <= 3:
                diffs = np.where(py_bytes != c_buf)[0]
                print(f'    FAIL t={t}: {len(diffs)} byte diffs, '
                      f'first @ {diffs[:5].tolist()}, '
                      f'py[0:8]={py_bytes[:8].tolist()}, c[0:8]={c_buf[:8].tolist()}')
    print(f'    {n_trials - n_fail}/{n_trials} OK')
    return n_fail == 0


# ----- Check 2a (INT4): unpack 2-per-byte UINT4 -------------------------

def test_unpack_uint4(lib, DataType, py_unpack, n_trials, rng):
    """C finn_cnn_unpack_uint4 must match FINN's packed_bytearray_to_finnpy.
    Random 392-byte packed buffers; each unpacks to 784 bytes (values 0..15
    per channel, low nibble = even-index, high nibble = odd-index)."""
    print(f'[2a] unpack UINT4 byte-exact — {n_trials} random 392-byte bufs')
    dtype = DataType['UINT4']
    n_fail = 0
    for t in range(n_trials):
        obuf = rng.integers(0, 256, size=392, dtype=np.uint8)
        py = py_unpack(obuf.reshape((1, 7, 7, 1, 8)), dtype,
                       (1, 7, 7, 1, 16),
                       reverse_endian=True, reverse_inner=True)
        py_u8 = np.ascontiguousarray(py).astype(np.int64).astype(np.uint8).flatten()
        assert py_u8.shape == (784,)

        c_buf = np.zeros(784, dtype=np.uint8)
        lib.finn_cnn_unpack_uint4(obuf.ctypes.data, c_buf.ctypes.data, 7 * 7 * 16)
        if not np.array_equal(py_u8, c_buf):
            n_fail += 1
            if n_fail <= 3:
                diffs = np.where(py_u8 != c_buf)[0]
                print(f'    FAIL t={t}: {len(diffs)} byte diffs, '
                      f'first @ {diffs[:5].tolist()}, '
                      f'py[0:8]={py_u8[:8].tolist()}, c[0:8]={c_buf[:8].tolist()}')
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


# ----- Check 5: GAP exactness on all-max obuf (run before end-to-end since
#                it's also a good smoke test of the runner wiring) --------

def test_gap_all_max(lib, precision):
    """At INT8: mock_obuf = all 0xFF (=255 each); after unpack→memcpy → feat=255.
    At INT4: mock_obuf = all 0xFF (392 bytes); each byte unpacks to two 0xF
    nibbles → feat=15. Both check that the GAP uint32 accumulator and the
    float divide are exact at the synthetic max."""
    pc = precision_consts(precision)
    deploy = deploy_path(precision)
    print(f'[5] GAP uint32-accumulator exactness (all-{pc["max_val"]} obuf, INT{precision})')

    W_conv  = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_0_param0.npy')).astype(np.float32))
    thres   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MultiThreshold_0_param0.npy')).astype(np.float32))
    W_cls   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_2_param0.npy')).astype(np.float32))
    mul_out = float(np.load(os.path.join(deploy, 'cnn_Mul_0_param0.npy')).flatten()[0])
    add_out = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_Add_0_param0.npy')).astype(np.float32))

    ibuf = np.zeros(pc['ibuf_packed_bytes'], dtype=np.uint8)
    obuf = np.zeros(pc['obuf_packed_bytes'], dtype=np.uint8)

    rc = lib.finn_cnn_runner_init(
        precision, precision,            # in_precision, out_precision (same for classic)
        28, 28, 1, 3, 1, 8, 7, 7, 16, 10, pc['num_thresholds'],
        0,                               # use_cache_ops=0
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,
        W_conv.ctypes.data, thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data,
        0,                               # cpu_post_maxpool_k=0 (tiny: FPGA includes final MaxPool)
        pc['ibuf_packed_bytes'],         # 6272 INT8 / 3136 INT4 (2-per-byte for tiny)
        pc['obuf_packed_bytes'],         # 784 INT8 / 392 INT4
        0, 0)                            # partition=classic, idt_signed=unsigned
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    # All-0xFF packed obuf. INT8 = 784 bytes of 255. INT4 = 392 bytes that
    # unpack to 784 bytes of 15. Either way every channel hits max_val.
    mock_obuf = np.ascontiguousarray(np.full(pc['obuf_packed_bytes'], 0xFF, dtype=np.uint8))
    dummy_img = np.zeros(28 * 28, dtype=np.uint8)
    pack_scratch = np.zeros(pc['ibuf_packed_bytes'], dtype=np.uint8)
    pred_c = lib.finn_cnn_infer_one_mock(
        dummy_img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

    # Python equivalent: feat = max_val everywhere (exact), then classifier.
    mv = float(pc['max_val'])
    feat = np.full(16, mv, dtype=np.float32)
    logits = feat @ W_cls
    out = logits.astype(np.float32) * mul_out + add_out
    pred_py = int(np.argmax(out))

    # Exactness check on Python's own GAP: float32 mean of 49 copies of mv.
    hw_out = np.full((7, 7, 16), mv, dtype=np.float32)
    feat_py = hw_out.mean(axis=(0, 1))
    ok_exact = bool(np.all(feat_py == mv))

    lib.finn_cnn_runner_destroy()

    if not ok_exact:
        print(f'    FAIL: np.mean of all-{mv} not exactly {mv}: {feat_py.tolist()}')
        return False
    if pred_c != pred_py:
        print(f'    FAIL: pred_c={pred_c}, pred_py={pred_py}')
        return False
    print(f'    OK: feat == {mv} (exact);  argmax: c={pred_c} py={pred_py}')
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


def test_end_to_end(lib, precision, n_trials, mnist_imgs, rng, DataType, py_pack, py_unpack):
    """End-to-end mock at the given precision.

    INT8: mock_obuf is 784 bytes of 0..255; pack_scratch holds 6272 bytes
    (1 byte/elem); pack expected = py_act flattened.
    INT4: mock_obuf is 392 bytes (any byte values); pack_scratch holds 3136
    bytes (2 elems/byte); pack expected = finnpy_to_packed_bytearray(py_act).
    For the GAP/argmax compare, Python first unpacks mock_obuf the same way
    the C runner would, so both paths see the same dense 784-byte view."""
    pc = precision_consts(precision)
    deploy = deploy_path(precision)
    print(f'[3+4] end-to-end mock — INT{precision}, {n_trials} MNIST images')

    W_conv  = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_0_param0.npy')).astype(np.float32))
    thres   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MultiThreshold_0_param0.npy')).astype(np.float32))
    W_cls   = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_MatMul_2_param0.npy')).astype(np.float32))
    mul_out = float(np.load(os.path.join(deploy, 'cnn_Mul_0_param0.npy')).flatten()[0])
    add_out = np.ascontiguousarray(np.load(os.path.join(
        deploy, 'cnn_Add_0_param0.npy')).astype(np.float32))

    ibuf = np.zeros(pc['ibuf_packed_bytes'], dtype=np.uint8)
    obuf = np.zeros(pc['obuf_packed_bytes'], dtype=np.uint8)

    rc = lib.finn_cnn_runner_init(
        precision, precision,            # in_precision, out_precision
        28, 28, 1, 3, 1, 8, 7, 7, 16, 10, pc['num_thresholds'],
        0,
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,
        W_conv.ctypes.data, thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data,
        0,                               # cpu_post_maxpool_k=0 (tiny)
        pc['ibuf_packed_bytes'],
        pc['obuf_packed_bytes'],
        0, 0)                            # partition=classic, idt_signed=unsigned
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    dtype = DataType[pc['dtype_name']]

    def py_cpu_pre(img_flat):
        img = img_flat.reshape(28, 28, 1).astype(np.float32) / 255.0
        patches = _im2col_py(img)
        x_f = patches @ W_conv                              # (28, 28, 8) float32
        act = np.sum(x_f[..., None] >= thres, axis=-1).astype(np.uint8)  # (28, 28, 8)
        return act

    def py_cpu_post_from_packed(packed_obuf):
        # Mirror what the C runner does: unpack packed_obuf to a dense
        # (7,7,16) uint8 view, then GAP + classifier + argmax.
        py = py_unpack(packed_obuf.reshape(pc['unpack_fold_shape'][:-1] +
                                           (pc['obuf_packed_bytes'] // (7 * 7),)),
                       dtype, pc['unpack_fold_shape'],
                       reverse_endian=True, reverse_inner=True)
        hw_out = np.ascontiguousarray(py).astype(np.int64).astype(np.float32).reshape(7, 7, 16)
        feat = hw_out.mean(axis=(0, 1))
        logits = feat @ W_cls
        out = logits.astype(np.float32) * mul_out + add_out
        return int(np.argmax(out))

    def py_pack_expected(py_act):
        # INT8: flatten NHWC, 1 byte/elem (memcpy parity).
        # INT4: finnpy_to_packed_bytearray on (1,28,28,1,8) → 3136 bytes.
        if precision == 8:
            return py_act.flatten().astype(np.uint8)
        py = py_pack(py_act.reshape((1, 28, 28, 1, 8)), dtype,
                     reverse_endian=True, reverse_inner=True, fast_mode=False)
        return np.ascontiguousarray(py).flatten().astype(np.uint8)

    n_fail_pred = 0
    n_fail_pack = 0
    pack_scratch = np.zeros(pc['ibuf_packed_bytes'], dtype=np.uint8)

    for i in range(n_trials):
        img = np.ascontiguousarray(mnist_imgs[i].flatten().astype(np.uint8))
        mock_obuf = np.ascontiguousarray(
            rng.integers(0, pc['mock_obuf_high'], size=pc['obuf_packed_bytes'], dtype=np.uint8))

        py_act = py_cpu_pre(img)           # (28, 28, 8) uint8 with values 0..max_val
        py_pred = py_cpu_post_from_packed(mock_obuf)

        c_pred = lib.finn_cnn_infer_one_mock(
            img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

        expected_pack = py_pack_expected(py_act)
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


# ----- Check 6: deep_3 (3-conv) end-to-end mock with CPU MaxPool stage ----

def _maxpool2x2_valid_nhwc(x):
    """2x2 stride-2 valid-pad MaxPool over an NHWC array (matches PyTorch
    MaxPool2d(2) and the C cpu_maxpool_2x2 helper). Drops the last row/col
    when H or W is odd."""
    H, W, C = x.shape
    OH, OW = (H - 2) // 2 + 1, (W - 2) // 2 + 1
    return x[:OH * 2, :OW * 2].reshape(OH, 2, OW, 2, C).max(axis=(1, 3))


def test_deep3_e2e_mock(lib, deploy_dir, n_trials, mnist_imgs, rng):
    """End-to-end mock for the deep_3 [16,32,64] 3-conv topology.

    Validates the new CPU 2x2 MaxPool stage that sits between unpack and GAP
    when cpu_post_maxpool_k > 0. Geometry:
        CPU pre  : Conv1 (1->16) at 28x28x16
        FPGA     : MaxPool->Conv2->MaxPool->Conv3 (opaque)
        CPU post : MaxPool (7x7x64 -> 3x3x64) -> GAP -> classifier (64->10)

    Only INT8 is exercised here; INT4 deep_3 follows the same pattern but is
    out of scope for this work item. mock_obuf is random uint8 bytes — the
    test only validates the post-FPGA pipeline, not the FPGA itself."""
    print(f'[6] deep_3 end-to-end mock — INT8, {n_trials} MNIST images')

    cfg_path = os.path.join(deploy_dir, 'cpu_config.json')
    if not os.path.exists(cfg_path):
        print(f'    SKIP: {cfg_path} not found '
              f'(run extract_finn_cpu_weights.py first)')
        return True
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get('cpu_post_maxpool_k', 0) != 2:
        print(f'    SKIP: cpu_post_maxpool_k={cfg.get("cpu_post_maxpool_k")}, '
              f'expected 2 for deep_3')
        return True
    if cfg.get('precision') != 8:
        print(f'    SKIP: precision={cfg.get("precision")}, this test is INT8-only')
        return True

    files = cfg['weight_files']
    W_conv  = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['W_conv'])).astype(np.float32))
    thres   = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['thres'])).astype(np.float32))
    W_cls   = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['W_cls'])).astype(np.float32))
    mul_out = float(np.load(os.path.join(deploy_dir, files['mul'])).flatten()[0])
    add_out = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['add'])).astype(np.float32))

    fpga_in_c   = int(W_conv.shape[1])              # 16
    fpga_out_h  = int(cfg['oshape_normal'][1])      # 7
    fpga_out_w  = int(cfg['oshape_normal'][2])      # 7
    fpga_out_c  = int(cfg['oshape_normal'][3])      # 64
    num_classes = int(W_cls.shape[1])               # 10
    nthres      = int(thres.shape[1])               # 255

    # INT8 packed buffers: 1 byte per element regardless of folding.
    ibuf_packed_bytes = 28 * 28 * fpga_in_c
    obuf_packed_bytes = fpga_out_h * fpga_out_w * fpga_out_c

    ibuf = np.zeros(ibuf_packed_bytes, dtype=np.uint8)
    obuf = np.zeros(obuf_packed_bytes, dtype=np.uint8)

    rc = lib.finn_cnn_runner_init(
        8, 8,                            # in_precision, out_precision (INT8 classic)
        28, 28, 1, 3, 1, fpga_in_c,
        fpga_out_h, fpga_out_w, fpga_out_c,
        num_classes, nthres,
        0,
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,
        W_conv.ctypes.data, thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data,
        2,                               # cpu_post_maxpool_k=2
        ibuf_packed_bytes,               # INT8: == n_in_elements
        obuf_packed_bytes,               # INT8: == n_out_elements
        0, 0)                            # partition=classic, idt_signed=unsigned
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    def py_cpu_pre(img_flat):
        img = img_flat.reshape(28, 28, 1).astype(np.float32) / 255.0
        patches = _im2col_py(img)
        x_f = patches @ W_conv
        return np.sum(x_f[..., None] >= thres, axis=-1).astype(np.uint8)

    def py_cpu_post_from_obuf(mock_obuf):
        # INT8: mock_obuf is the dense (fpga_out_h, fpga_out_w, fpga_out_c)
        # uint8 view (memcpy unpack). Apply CPU MaxPool, GAP, classifier.
        hw_out = mock_obuf.reshape(fpga_out_h, fpga_out_w, fpga_out_c)
        pooled = _maxpool2x2_valid_nhwc(hw_out)              # (3, 3, 64)
        feat   = pooled.astype(np.float32).mean(axis=(0, 1)) # (64,)
        logits = feat @ W_cls
        out = logits.astype(np.float32) * mul_out + add_out
        return int(np.argmax(out))

    n_fail_pred = 0
    n_fail_pack = 0
    pack_scratch = np.zeros(ibuf_packed_bytes, dtype=np.uint8)

    for i in range(n_trials):
        img = np.ascontiguousarray(mnist_imgs[i].flatten().astype(np.uint8))
        mock_obuf = np.ascontiguousarray(
            rng.integers(0, 256, size=obuf_packed_bytes, dtype=np.uint8))

        py_act  = py_cpu_pre(img)                            # (28, 28, 16)
        py_pred = py_cpu_post_from_obuf(mock_obuf)
        c_pred  = lib.finn_cnn_infer_one_mock(
            img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

        # Pack scratch at INT8 is just the activations flattened.
        expected_pack = py_act.flatten().astype(np.uint8)
        if not np.array_equal(pack_scratch, expected_pack):
            n_fail_pack += 1
            if n_fail_pack <= 3:
                diffs = np.where(pack_scratch != expected_pack)[0]
                print(f'    FAIL pack i={i}: {len(diffs)} byte diffs, '
                      f'first @ {diffs[:5].tolist()}')

        if c_pred != py_pred:
            n_fail_pred += 1
            if n_fail_pred <= 3:
                print(f'    FAIL argmax i={i}: py={py_pred}, c={c_pred}')

    lib.finn_cnn_runner_destroy()
    print(f'    argmax: {n_trials - n_fail_pred}/{n_trials} OK,  '
          f'pack:   {n_trials - n_fail_pack}/{n_trials} OK')
    return (n_fail_pred == 0) and (n_fail_pack == 0)


# ----- Check 7: deep_3 INT4 — exercises 1-per-byte output unpack -----------

def test_deep3_int4_e2e_mock(lib, deploy_dir, n_trials, mnist_imgs, rng):
    """End-to-end mock for the deep_3 INT4 topology.

    Critical regression: deep_3 INT4 has asymmetric packing — ishape_packed
    is (1, 28, 28, 1, 8) → input is 2-per-byte (4 bytes/pixel), but
    oshape_packed is (1, 7, 7, 64, 1) → output is 1-per-byte (64 bytes/pixel,
    only low nibble carries the 4-bit value). Tiny INT4 has SIMD=16 on the
    output → 2-per-byte both directions, so the original C unpack assumption
    (always 2-per-byte at INT4) only broke on deep_3 INT4. This test
    catches that regression by verifying Python ↔ C argmax agreement on
    100 random obuf bytes."""
    print(f'[7] deep_3 INT4 end-to-end mock — {n_trials} MNIST images')

    cfg_path = os.path.join(deploy_dir, 'cpu_config.json')
    if not os.path.exists(cfg_path):
        print(f'    SKIP: {cfg_path} not found '
              f'(run extract_finn_cpu_weights.py first)')
        return True
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get('cpu_post_maxpool_k', 0) != 2:
        print(f'    SKIP: cpu_post_maxpool_k={cfg.get("cpu_post_maxpool_k")}, '
              f'expected 2 for deep_3')
        return True
    if cfg.get('precision') != 4:
        print(f'    SKIP: precision={cfg.get("precision")}, this test is INT4-only')
        return True

    files = cfg['weight_files']
    W_conv  = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['W_conv'])).astype(np.float32))
    thres   = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['thres'])).astype(np.float32))
    W_cls   = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['W_cls'])).astype(np.float32))
    mul_out = float(np.load(os.path.join(deploy_dir, files['mul'])).flatten()[0])
    add_out = np.ascontiguousarray(np.load(os.path.join(
        deploy_dir, files['add'])).astype(np.float32))

    fpga_in_c   = int(W_conv.shape[1])              # 16
    fpga_out_h  = int(cfg['oshape_normal'][1])      # 7
    fpga_out_w  = int(cfg['oshape_normal'][2])      # 7
    fpga_out_c  = int(cfg['oshape_normal'][3])      # 64
    num_classes = int(W_cls.shape[1])               # 10
    nthres      = int(thres.shape[1])               # 15

    n_in_elems  = 28 * 28 * fpga_in_c                # 12544
    n_out_elems = fpga_out_h * fpga_out_w * fpga_out_c  # 3136
    # deep_3 INT4 packing: input 2-per-byte, output 1-per-byte (per driver.py).
    ibuf_packed_bytes = n_in_elems // 2              # 6272
    obuf_packed_bytes = n_out_elems                  # 3136 (1-per-byte)

    ibuf = np.zeros(ibuf_packed_bytes, dtype=np.uint8)
    obuf = np.zeros(obuf_packed_bytes, dtype=np.uint8)

    rc = lib.finn_cnn_runner_init(
        4, 4,                            # in_precision, out_precision (INT4 classic)
        28, 28, 1, 3, 1, fpga_in_c,
        fpga_out_h, fpga_out_w, fpga_out_c,
        num_classes, nthres,
        0,
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,
        W_conv.ctypes.data, thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data,
        2,                               # cpu_post_maxpool_k=2
        ibuf_packed_bytes,
        obuf_packed_bytes,
        0, 0)                            # partition=classic, idt_signed=unsigned
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    def py_cpu_pre(img_flat):
        img = img_flat.reshape(28, 28, 1).astype(np.float32) / 255.0
        patches = _im2col_py(img)
        x_f = patches @ W_conv
        return np.sum(x_f[..., None] >= thres, axis=-1).astype(np.uint8)

    def py_cpu_post_from_obuf(mock_obuf):
        # 1-per-byte INT4: low nibble is the 4-bit value, high nibble masked off.
        # Mirrors finn_cnn_unpack_uint4_1pb in C.
        obuf_unpacked = (mock_obuf & 0x0F).reshape(fpga_out_h, fpga_out_w, fpga_out_c)
        pooled = _maxpool2x2_valid_nhwc(obuf_unpacked)
        feat   = pooled.astype(np.float32).mean(axis=(0, 1))
        logits = feat @ W_cls
        out = logits.astype(np.float32) * mul_out + add_out
        return int(np.argmax(out))

    def py_pack_2pb_expected(py_act):
        # 2-per-byte INT4 input pack: byte[i] = (act[2i] & 0xF) | ((act[2i+1] & 0xF) << 4).
        flat = py_act.flatten().astype(np.uint8)
        return ((flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)).astype(np.uint8)

    n_fail_pred = 0
    n_fail_pack = 0
    pack_scratch = np.zeros(ibuf_packed_bytes, dtype=np.uint8)

    for i in range(n_trials):
        img = np.ascontiguousarray(mnist_imgs[i].flatten().astype(np.uint8))
        # Random obuf bytes (full 0..255) — C's 1pb unpack masks to low nibble.
        mock_obuf = np.ascontiguousarray(
            rng.integers(0, 256, size=obuf_packed_bytes, dtype=np.uint8))

        py_act  = py_cpu_pre(img)
        py_pred = py_cpu_post_from_obuf(mock_obuf)
        c_pred  = lib.finn_cnn_infer_one_mock(
            img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

        expected_pack = py_pack_2pb_expected(py_act)
        if not np.array_equal(pack_scratch, expected_pack):
            n_fail_pack += 1
            if n_fail_pack <= 3:
                diffs = np.where(pack_scratch != expected_pack)[0]
                print(f'    FAIL pack i={i}: {len(diffs)} byte diffs, '
                      f'first @ {diffs[:5].tolist()}')

        if c_pred != py_pred:
            n_fail_pred += 1
            if n_fail_pred <= 3:
                print(f'    FAIL argmax i={i}: py={py_pred}, c={c_pred}')

    lib.finn_cnn_runner_destroy()
    print(f'    argmax: {n_trials - n_fail_pred}/{n_trials} OK,  '
          f'pack:   {n_trials - n_fail_pack}/{n_trials} OK')
    return (n_fail_pred == 0) and (n_fail_pack == 0)


# ----- Check 8: QI partition end-to-end mock -----------------------------

def test_qi_e2e_mock(lib, deploy_dir, n_trials, mnist_imgs, rng):
    """End-to-end mock for the QI partition (CNN small INT8 with input
    QuantIdentity). The CPU pre-stage is just per-channel input
    MultiThreshold on the float-normalized image — no im2col, no Conv1
    MatMul. The post-stage (optional MaxPool, GAP, classifier MatMul,
    dequant, argmax) is identical to the classic path. Asserts c
    argmax == numpy reference over n_trials random mock obufs.
    """
    print(f'[qi-e2e] CNN QI mock — {n_trials} MNIST images, deploy={deploy_dir}')
    if not os.path.isdir(deploy_dir):
        print(f'    SKIP: deploy not found: {deploy_dir}')
        return True
    cfg_path = os.path.join(deploy_dir, 'cpu_config.json')
    if not os.path.isfile(cfg_path):
        print(f'    SKIP: no cpu_config.json')
        return True
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get('partition') != 'qi':
        print(f'    SKIP: cpu_config.json partition={cfg.get("partition")!r}')
        return True

    wf = cfg['weight_files']
    input_thres = np.ascontiguousarray(np.load(os.path.join(deploy_dir, wf['input_thres'])).astype(np.float32))
    W_cls       = np.ascontiguousarray(np.load(os.path.join(deploy_dir, wf['W_cls'])).astype(np.float32))
    add_out     = np.ascontiguousarray(np.load(os.path.join(deploy_dir, wf['add'])).astype(np.float32))
    mul_out     = float(np.load(os.path.join(deploy_dir, wf['mul'])).flatten()[0])

    fpga_in_c = int(input_thres.shape[0])     # 1 for MNIST
    fpga_out_h, fpga_out_w, fpga_out_c = (int(x) for x in cfg['oshape_normal'][1:])
    cpu_post_maxpool_k = int(cfg.get('cpu_post_maxpool_k', 0))
    num_classes = int(add_out.shape[0])
    num_thresholds = int(input_thres.shape[1])
    n_in_elems  = 28 * 28 * fpga_in_c
    n_out_elems = fpga_out_h * fpga_out_w * fpga_out_c

    ibuf = np.zeros(n_in_elems, dtype=np.uint8)
    obuf = np.zeros(n_out_elems, dtype=np.uint8)

    # QI: W_conv NULL (Conv1 is on FPGA). Pass 0 for the pointer.
    rc = lib.finn_cnn_runner_init(
        8, 8,                               # in_precision, out_precision (INT8 QI)
        28, 28, 1,                          # img_h,w,c
        3, 1,                               # kernel, pad (unused on QI)
        fpga_in_c,
        fpga_out_h, fpga_out_w, fpga_out_c,
        num_classes, num_thresholds,
        0,                                  # use_cache_ops=0 (mock — no DMA)
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,                               # idma, odma mmio (unused)
        0,                                  # W_conv = NULL
        input_thres.ctypes.data, W_cls.ctypes.data,
        mul_out, add_out.ctypes.data,
        cpu_post_maxpool_k,
        n_in_elems, n_out_elems,
        1,                                  # partition=qi
        1 if cfg.get('idt_signed', True) else 0)  # idt_signed (default True for QI deploys)
    if rc != 0:
        print(f'    runner_init rc={rc}')
        return False

    # idt_signed default True for QI deploys (QuantIdentity(Int8...)). For
    # signed IDT the C runner shifts the count by 2^(N-1) so the byte
    # reinterpreted as int8 matches the MultiThreshold's signed output.
    idt_signed = bool(cfg.get('idt_signed', True))
    bias = (num_thresholds + 1) // 2 if idt_signed else 0
    def py_cpu_pre_qi(img_flat):
        img = img_flat.reshape(28, 28, 1).astype(np.float32) / 255.0
        # input_thres shape (C_in, T), broadcast over H,W positions.
        # >= matches the FINN convention (qonnx multithreshold:78).
        count = np.sum(img[..., np.newaxis] >= input_thres, axis=-1).astype(np.int32)
        return (count - bias).astype(np.uint8)   # wraps for negative values

    def py_cpu_post(obuf_dense):
        feat = obuf_dense.reshape(fpga_out_h, fpga_out_w, fpga_out_c).astype(np.float32)
        if cpu_post_maxpool_k == 2:
            H, Wd, Cd = feat.shape
            OH, OW = (H - 2) // 2 + 1, (Wd - 2) // 2 + 1
            feat = feat[:OH * 2, :OW * 2].reshape(OH, 2, OW, 2, Cd).max(axis=(1, 3))
        feat = feat.mean(axis=(0, 1))
        logits = feat @ W_cls
        out = logits.astype(np.float32) * mul_out + add_out
        return int(np.argmax(out))

    pack_scratch = np.zeros(n_in_elems, dtype=np.uint8)
    n_fail_pred = 0
    n_fail_pack = 0
    for i in range(n_trials):
        img = np.ascontiguousarray(mnist_imgs[i].flatten().astype(np.uint8))
        mock_obuf = np.ascontiguousarray(
            rng.integers(0, 256, size=n_out_elems, dtype=np.uint8))
        py_act = py_cpu_pre_qi(img)
        py_pred = py_cpu_post(mock_obuf)
        c_pred = lib.finn_cnn_infer_one_mock(
            img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)
        # INT8 pack is memcpy, so pack_scratch must equal py_act flattened.
        if not np.array_equal(pack_scratch, py_act.reshape(-1)):
            n_fail_pack += 1
            if n_fail_pack <= 3:
                print(f'    FAIL pack i={i}')
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
    ap.add_argument('--precision', type=int, choices=[8, 4], default=8,
                    help='Test the INT8 path (default) or the INT4 path. '
                         'Both must pass.')
    ap.add_argument('--deep3-deploy', default=DEPLOY_CNN_DEEP3_INT8,
                    help='Path to the deep_3 INT8 FINN deploy/.')
    ap.add_argument('--deep3-int4-deploy', default=DEPLOY_CNN_DEEP3_INT4,
                    help='Path to the deep_3 INT4 FINN deploy/.')
    ap.add_argument('--skip-deep3', action='store_true',
                    help='Skip the deep_3 mock tests (both INT8 and INT4).')
    ap.add_argument('--qi-deploy', default=DEPLOY_CNN_QI_SMALL_INT8,
                    help='Path to a CNN QI partition deploy/ for the QI mock test.')
    ap.add_argument('--skip-qi', action='store_true',
                    help='Skip the QI partition mock test.')
    args = ap.parse_args()

    so_path = args.so or tempfile.NamedTemporaryFile(
        suffix='.so', prefix='libfinn_cnn_infer_test_', delete=False).name
    build_so(args.src, so_path)
    lib = load_lib(so_path)
    DataType, py_pack, py_unpack = setup_data_packing(args.precision)

    rng = np.random.default_rng(args.seed)
    ok = True
    if args.precision == 8:
        ok &= test_pack_byte_exact(lib, DataType, py_pack,
                                   args.pack_trials, rng)
        ok &= test_unpack_byte_exact(lib, DataType, py_unpack,
                                     args.unpack_trials, rng)
        ok &= test_obuf_layout(DataType, py_unpack)
    else:
        ok &= test_pack_uint4(lib, DataType, py_pack,
                              args.pack_trials, rng)
        ok &= test_unpack_uint4(lib, DataType, py_unpack,
                                args.unpack_trials, rng)
        # test_obuf_layout proves Python's reshape(7,7,16).mean axes match
        # the C GAP loop's indexing. That's a property of the dense (1
        # byte/channel) layout the GAP reads — same at INT8 and INT4 since
        # the unpack step always produces the dense layout — so the INT8
        # check covers both. Skipping at INT4 to avoid duplicating it.

    ok &= test_gap_all_max(lib, args.precision)

    mnist_imgs, _ = load_mnist_test()
    ok &= test_end_to_end(lib, args.precision, args.e2e_trials, mnist_imgs, rng,
                          DataType, py_pack, py_unpack)

    # Deep_3 tests exercise the new CPU MaxPool stage. INT8 was the
    # original coverage; INT4 was added after a regression where the C
    # runner's hardcoded "INT4 = always 2-per-byte" assumption broke for
    # deep_3 (which has 1-per-byte output due to SIMD=1 folding).
    if not args.skip_deep3:
        if args.precision == 8:
            ok &= test_deep3_e2e_mock(lib, args.deep3_deploy, args.e2e_trials,
                                      mnist_imgs, rng)
        else:
            ok &= test_deep3_int4_e2e_mock(lib, args.deep3_int4_deploy,
                                            args.e2e_trials, mnist_imgs, rng)

    # QI partition mock — INT8 only (the only QI deploy we currently ship).
    # Skipped when the deploy isn't built or --skip-qi is passed.
    if not args.skip_qi and args.precision == 8:
        ok &= test_qi_e2e_mock(lib, args.qi_deploy, args.e2e_trials,
                                mnist_imgs, rng)

    print('\n===== HARNESS (INT{}):'.format(args.precision),
          'PASS' if ok else 'FAIL', '=====')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
