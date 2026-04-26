#!/usr/bin/env python3
"""Host-side correctness harness for libfinn_mlp_infer.so.

Builds the .so locally (host gcc, x86_64 OK because the ARM cache-op
asm is gated). No board execution. Two test groups:

  1. Byte-exact pack/unpack vs finn.util.data_packing reference,
     N_PACK random inputs per precision per direction.
  2. End-to-end argmax match against numpy-stubbed Python flow,
     N_E2E MNIST images per precision. The "FPGA" is replaced by a
     deterministic random byte sequence; both Python and C decode the
     SAME bytes with their respective unpack routines and feed
     identical hw[] integers into the dequant + argmax tail. Tests
     CPU pre, pack-to-scratch, unpack, and CPU post in one call.

Note: qonnx multithreshold uses '>=' (inclusive); this aligns the C
runner's CPU pre with the FPGA-side semantics. No board action needed.
"""

import argparse
import ctypes
import gzip
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
DEPLOY_INT8 = os.path.join(REPO_ROOT, 'finn-vs-vitisai/finn/output_mlp_mnist_tiny/deploy')
DEPLOY_INT4 = os.path.join(REPO_ROOT, 'finn-vs-vitisai/finn/output_mlp_mnist_tiny_int4/deploy')
MNIST_RAW   = os.path.join(REPO_ROOT, 'finn-vs-vitisai/data/MNIST/raw')

# data_packing + qonnx live bundled with the deploy.
DATA_PACKING_DRIVER_DIR = os.path.join(DEPLOY_INT8, 'driver')


# ----- build / load -------------------------------------------------------

def build_so(src_path, so_path):
    cmd = ['gcc', '-O2', '-shared', '-fPIC', '-Wall', '-Werror',
           '-o', so_path, src_path]
    print(f'Building: {" ".join(cmd)}')
    subprocess.check_call(cmd)


def load_lib(so_path):
    lib = ctypes.CDLL(so_path)

    lib.finn_mlp_pack_uint8.argtypes            = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_mlp_pack_uint4.argtypes            = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_mlp_pack_uint4_2perbyte.argtypes   = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_mlp_unpack_int24_le.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.finn_mlp_unpack_int16_le.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    for fn in (lib.finn_mlp_pack_uint8, lib.finn_mlp_pack_uint4,
               lib.finn_mlp_pack_uint4_2perbyte,
               lib.finn_mlp_unpack_int24_le, lib.finn_mlp_unpack_int16_le):
        fn.restype = None

    lib.finn_mlp_runner_init.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,                            # ibuf_bytes (NEW)
        ctypes.c_int,                            # use_cache_ops
        ctypes.c_void_p, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.finn_mlp_runner_init.restype = ctypes.c_int

    lib.finn_mlp_runner_destroy.argtypes = []
    lib.finn_mlp_runner_destroy.restype  = ctypes.c_int

    lib.finn_mlp_infer_one_mock.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.finn_mlp_infer_one_mock.restype = ctypes.c_int

    return lib


def setup_data_packing():
    sys.path.insert(0, DATA_PACKING_DRIVER_DIR)
    from qonnx.core.datatype import DataType
    from finn.util.data_packing import (
        finnpy_to_packed_bytearray, packed_bytearray_to_finnpy)
    return DataType, finnpy_to_packed_bytearray, packed_bytearray_to_finnpy


# ----- MNIST loader (matches benchmark.py:load_mnist on raw idx files) ----

def load_mnist_test():
    with open(os.path.join(MNIST_RAW, 't10k-images-idx3-ubyte'), 'rb') as f:
        magic, n, H, W = struct.unpack('>IIII', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, H, W)
    with open(os.path.join(MNIST_RAW, 't10k-labels-idx1-ubyte'), 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, labels


# ----- pack / unpack tests ------------------------------------------------

def test_pack(lib, DataType, py_pack, *, precision, mode, fold_shape, ibuf_bytes,
              n_trials, mid_dim, rng):
    """Pack-byte-exact test for one (precision, mode) combination.

    mode ∈ {'uint8', 'uint4_1perbyte', 'uint4_2perbyte'}.
    fold_shape is the FINN-style folded shape passed to finnpy_to_packed_bytearray.
    ibuf_bytes is the expected packed byte count (= prod(ishape_packed[1:])).
    """
    if mode == 'uint8':
        dtype, max_val = DataType['UINT8'], 255
        c_fn = lib.finn_mlp_pack_uint8
    elif mode == 'uint4_1perbyte':
        dtype, max_val = DataType['UINT4'], 15
        c_fn = lib.finn_mlp_pack_uint4
    elif mode == 'uint4_2perbyte':
        dtype, max_val = DataType['UINT4'], 15
        c_fn = lib.finn_mlp_pack_uint4_2perbyte
    else:
        raise ValueError(f'unknown mode {mode!r}')

    label = f'INT{precision} {mode}'
    print(f'[pack {label}] {n_trials} trials, fold={fold_shape}, ibuf={ibuf_bytes} B')
    n_fail = 0
    for t in range(n_trials):
        act = rng.integers(0, max_val + 1, size=mid_dim, dtype=np.uint8)

        py = py_pack(act.reshape(fold_shape), dtype,
                     reverse_endian=True, reverse_inner=True, fast_mode=False)
        py_bytes = np.ascontiguousarray(py).flatten().astype(np.uint8)
        assert py_bytes.shape == (ibuf_bytes,), (
            f'unexpected python pack shape {py_bytes.shape} '
            f'(expected ({ibuf_bytes},)) for fold={fold_shape}')

        c_buf = np.zeros(ibuf_bytes, dtype=np.uint8)
        c_fn(act.ctypes.data, c_buf.ctypes.data, mid_dim)

        if not np.array_equal(py_bytes, c_buf):
            n_fail += 1
            if n_fail <= 3:
                print(f'  FAIL trial {t}:')
                print(f'    act head: {act[:8].tolist()}')
                print(f'    py  head: {py_bytes[:16].tolist()}')
                print(f'    c   head: {c_buf[:16].tolist()}')
    print(f'[pack {label}] {n_trials - n_fail}/{n_trials} OK')
    return n_fail == 0


def test_unpack(lib, DataType, py_unpack, *, precision, n_trials, num_classes, rng):
    if precision == 8:
        dtype = DataType['INT24']
        out_be = 3
        c_fn = lib.finn_mlp_unpack_int24_le
    else:
        dtype = DataType['INT16']
        out_be = 2
        c_fn = lib.finn_mlp_unpack_int16_le
    obuf_total = num_classes * out_be
    print(f'[unpack INT{precision} -> {dtype.name if hasattr(dtype, "name") else dtype}]'
          f' {n_trials} trials, obuf={obuf_total} bytes')
    n_fail = 0
    for t in range(n_trials):
        obuf = rng.integers(0, 256, size=obuf_total, dtype=np.uint8)
        # NB: the bundled finn.util.data_packing has fast_mode= here, but the
        # venv version (finn-plus) drops it. The slow path produces identical
        # output — fast_mode is a memcpy short-circuit only.
        py = py_unpack(obuf.reshape((1, num_classes, out_be)), dtype,
                       (1, num_classes, 1),
                       reverse_endian=True, reverse_inner=True)
        py_int32 = py.flatten().astype(np.int64).astype(np.int32)

        c_out = np.zeros(num_classes, dtype=np.int32)
        c_fn(obuf.ctypes.data, c_out.ctypes.data, num_classes)

        if not np.array_equal(py_int32, c_out):
            n_fail += 1
            if n_fail <= 3:
                print(f'  FAIL trial {t}:')
                print(f'    obuf: {obuf.tolist()}')
                print(f'    py  : {py_int32.tolist()}')
                print(f'    c   : {c_out.tolist()}')
    print(f'[unpack INT{precision}] {n_trials - n_fail}/{n_trials} OK')
    return n_fail == 0


# ----- end-to-end mock ----------------------------------------------------

def python_mock_infer(img_u8, W0, thres, mul, add, mock_hw):
    """Python equivalent of the C path. Returns (pred, act_uint8)."""
    acc = (img_u8.astype(np.float32) / 255.0) @ W0      # (mid_dim,)
    act = np.sum(acc[:, None] >= thres, axis=-1).astype(np.uint8)   # >= matches C
    out = mock_hw.astype(np.float32) * mul + add
    return int(np.argmax(out)), act


def mock_obuf_to_hw(obuf_bytes, precision, num_classes):
    """Decode mock obuf bytes into int32 hw[] the same way C unpack does.
    Used so both Python and C feed identical integers to the post-stage."""
    hw = np.zeros(num_classes, dtype=np.int32)
    if precision == 8:
        for c in range(num_classes):
            b = obuf_bytes[c*3:(c+1)*3]
            v = int(b[0]) | (int(b[1]) << 8) | (int(b[2]) << 16)
            if v & 0x800000:
                v -= 0x1000000      # sign-extend 24-bit
            hw[c] = v
    else:
        for c in range(num_classes):
            b = obuf_bytes[c*2:(c+1)*2]
            v = int(b[0]) | (int(b[1]) << 8)
            if v & 0x8000:
                v -= 0x10000        # sign-extend 16-bit
            hw[c] = v
    return hw


def test_end_to_end(lib, *, deploy, precision, n_trials, mnist_imgs, rng):
    print(f'\n[end2end INT{precision}] {deploy}')
    W0    = np.load(os.path.join(deploy, 'mlp_MatMul_0_param0.npy')).astype(np.float32)
    thres = np.load(os.path.join(deploy, 'mlp_MultiThreshold_0_param0.npy')).astype(np.float32)
    mul   = float(np.load(os.path.join(deploy, 'mlp_Mul_0_param0.npy')).flatten()[0])
    add   = np.load(os.path.join(deploy, 'mlp_Add_0_param0.npy')).astype(np.float32)
    in_dim, mid_dim = W0.shape
    num_classes = add.shape[0]
    nthres = thres.shape[1]
    out_be = 3 if precision == 8 else 2
    obuf_total = num_classes * out_be

    W0c    = np.ascontiguousarray(W0)
    thresc = np.ascontiguousarray(thres)
    addc   = np.ascontiguousarray(add)

    # ibuf/obuf live in regular host memory — mock entry doesn't DMA.
    ibuf = np.zeros(mid_dim,    dtype=np.uint8)
    obuf = np.zeros(obuf_total, dtype=np.uint8)

    rc = lib.finn_mlp_runner_init(
        precision, in_dim, mid_dim, num_classes, nthres,
        mid_dim,                                # ibuf_bytes — baseline deploys are 1-per-byte
        0,                                      # use_cache_ops=0
        ibuf.ctypes.data, 0,
        obuf.ctypes.data, 0,
        0, 0,                                   # idma/odma MMIO NULL — mock skips DMA
        W0c.ctypes.data, thresc.ctypes.data, mul, addc.ctypes.data)
    if rc != 0:
        print(f'  finn_mlp_runner_init returned {rc}')
        return False

    n_fail_pred = 0
    n_fail_pack = 0
    pack_scratch = np.zeros(mid_dim, dtype=np.uint8)

    for i in range(n_trials):
        img = np.ascontiguousarray(mnist_imgs[i].flatten().astype(np.uint8))
        mock_obuf = rng.integers(0, 256, size=obuf_total, dtype=np.uint8)
        mock_obuf = np.ascontiguousarray(mock_obuf)

        mock_hw = mock_obuf_to_hw(mock_obuf, precision, num_classes)
        py_pred, py_act = python_mock_infer(img, W0, thres, mul, add, mock_hw)

        c_pred = lib.finn_mlp_infer_one_mock(
            img.ctypes.data, mock_obuf.ctypes.data, pack_scratch.ctypes.data)

        if c_pred != py_pred:
            n_fail_pred += 1
            if n_fail_pred <= 3:
                print(f'  FAIL pred i={i}: py={py_pred} c={c_pred}')
                print(f'    py_act head: {py_act[:8].tolist()}')

        # pack scratch should equal the python-side act bytes:
        #   INT8: act fits in uint8 -> equal
        #   INT4: pack masks low nibble; py_act is in [0,15] so & 0x0F is identity
        expected = py_act if precision == 8 else (py_act & 0x0F)
        if not np.array_equal(pack_scratch, expected):
            n_fail_pack += 1
            if n_fail_pack <= 3:
                print(f'  FAIL pack i={i}:')
                print(f'    py: {expected[:8].tolist()}')
                print(f'    c : {pack_scratch[:8].tolist()}')

    lib.finn_mlp_runner_destroy()

    print(f'[end2end INT{precision}] argmax: {n_trials - n_fail_pred}/{n_trials} OK,'
          f' pack-act: {n_trials - n_fail_pack}/{n_trials} OK')
    return n_fail_pred == 0 and n_fail_pack == 0


# ----- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',  default=os.path.join(THIS_DIR, 'finn_mlp_infer.c'))
    ap.add_argument('--so',   default=None)
    ap.add_argument('--pack-trials',   type=int, default=1000)
    ap.add_argument('--unpack-trials', type=int, default=1000)
    ap.add_argument('--e2e-trials',    type=int, default=100)
    ap.add_argument('--seed',          type=int, default=12345)
    args = ap.parse_args()

    so_path = args.so or tempfile.NamedTemporaryFile(
        suffix='.so', prefix='libfinn_mlp_infer_test_', delete=False).name
    build_so(args.src, so_path)
    lib = load_lib(so_path)
    DataType, py_pack, py_unpack = setup_data_packing()

    rng = np.random.default_rng(args.seed)
    ok = True
    # mid_dim=64 mirrors the deployed mlp_mnist_tiny ([784, 64, 32, 10] hidden=[64,32]
    # → first FPGA layer's mid_dim is 64). Three pack modes:
    #   uint8 1-per-byte:        baseline INT8 deploy + mlp_int8_fps500000 (SIMD=4 still
    #                            byte-aligned, memcpy works).
    #   uint4 1-per-byte:        baseline INT4 deploy (SIMD=1).
    #   uint4 2-per-byte:        mlp_int4_fps500000 (SIMD=16, FINN packs 16 INT4
    #                            elements per chunk into 8 bytes; same convention as
    #                            CNN INT4 baseline).
    ok &= test_pack(lib, DataType, py_pack, precision=8, mode='uint8',
                    fold_shape=(1, 64, 1), ibuf_bytes=64,
                    n_trials=args.pack_trials, mid_dim=64, rng=rng)
    ok &= test_pack(lib, DataType, py_pack, precision=4, mode='uint4_1perbyte',
                    fold_shape=(1, 64, 1), ibuf_bytes=64,
                    n_trials=args.pack_trials, mid_dim=64, rng=rng)
    ok &= test_pack(lib, DataType, py_pack, precision=4, mode='uint4_2perbyte',
                    fold_shape=(1, 4, 16), ibuf_bytes=32,
                    n_trials=args.pack_trials, mid_dim=64, rng=rng)
    ok &= test_unpack(lib, DataType, py_unpack, precision=8,
                      n_trials=args.unpack_trials, num_classes=10, rng=rng)
    ok &= test_unpack(lib, DataType, py_unpack, precision=4,
                      n_trials=args.unpack_trials, num_classes=10, rng=rng)

    mnist_imgs, _ = load_mnist_test()
    ok &= test_end_to_end(lib, deploy=DEPLOY_INT8, precision=8,
                          n_trials=args.e2e_trials, mnist_imgs=mnist_imgs, rng=rng)
    ok &= test_end_to_end(lib, deploy=DEPLOY_INT4, precision=4,
                          n_trials=args.e2e_trials, mnist_imgs=mnist_imgs, rng=rng)

    print('\n===== HARNESS:', 'PASS' if ok else 'FAIL', '=====')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
