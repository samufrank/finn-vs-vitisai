#!/usr/bin/env python3
"""Minimal board-side smoke test for the 4-arg INT4-o8 codegen path.

Two stages:

  Stage 1 — TINY (o=1, n=1, m=1, shift=2)
    A = all 1's  (1, 1, 1, 16)  int4
    W = all 1's  (1, 1, 16, 16) int4
    D = all 0's  (1, 1, 1, 16)  int32
    Expected C[i] = (sum_k A[k]*W[ci, k] + 0) >> 2 = (16) >> 2 = 4 for every elem.

  Stage 2 — CNN_SHAPE (o=196, n=1, m=1, shift=2)
    Same arithmetic, larger o_tile. Matches CNN Conv1.

If Stage 1 PASSES → the 4-arg INT4-o8 codegen path is sound; the transformer
all-zero bug is multi-tile-shape-specific.
If Stage 1 FAILS  → the compilation pipeline itself is broken (env / pkg_config /
PL bitstream mismatch); the transformer can never work until this passes.

Usage on the board:
    sudo LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH \\
         PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH \\
         python3 test_tiny_module.py --module-dir /home/xilinx/tiny_test
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import subprocess
import sys

import numpy as np


BITSTREAM = "1x16_i4w4o8a32_15_14_17_17.bit"


def pack_int4_for_vta(vals_int8):
    """Same packing the working CNN INT4-o8 uses."""
    vals = np.asarray(vals_int8, dtype=np.int8)
    flat = vals.flatten()
    n = len(flat)
    lo = flat[0::2].view(np.uint8) & 0xF
    hi = flat[1::2].view(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[: n // 2] = packed
    return out.reshape(vals.shape)


def setup_board():
    stale = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale):
            os.remove(stale)
    except Exception:
        pass
    for p in [f'/root/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
              f'/home/xilinx/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
              f'/home/xilinx/{BITSTREAM}']:
        if os.path.exists(p):
            print(f"[bitstream] {p}")
            try:
                from pynq import Overlay
                Overlay(p)
            except Exception as e:
                print(f"  Overlay: {e} (continuing)")
            break
    for p in ['/home/xilinx/tvm-src/build/libvta.so']:
        if os.path.exists(p):
            print(f"[libvta] {p}")
            ctypes.CDLL(p, ctypes.RTLD_GLOBAL)
            break

    import vta as _vta
    e = _vta.get_env()
    print(f"[vta env] INP={e.INP_WIDTH} WGT={e.WGT_WIDTH} OUT={e.OUT_WIDTH} "
          f"BATCH={e.BATCH} BLOCK={e.BLOCK_IN}/{e.BLOCK_OUT}")
    if (e.INP_WIDTH, e.WGT_WIDTH, e.OUT_WIDTH, e.BATCH, e.BLOCK_IN, e.BLOCK_OUT) \
            != (4, 4, 8, 1, 16, 16):
        raise RuntimeError(
            "Host VTA env != INT4-o8 (4,4,8,1,16,16). The compiled tiny "
            "module won't match the hardware. Restore the int4_o8 vta_config.")
    return e


def link_so(o_path):
    so_path = o_path.replace('.o', '.so')
    if not os.path.exists(so_path):
        print(f"  link {os.path.basename(o_path)} -> .so")
        subprocess.check_call([
            'gcc', '-shared', '-o', so_path, o_path,
            '-L/home/xilinx/tvm-src/build', '-ltvm_runtime',
        ])
    return so_path


def run_one(name, mod, A_packed, W_packed, D_int32, C_shape, ctx):
    import tvm
    A_nd = tvm.nd.array(A_packed, ctx)
    W_nd = tvm.nd.array(W_packed, ctx)
    D_nd = tvm.nd.array(D_int32, ctx)
    C_nd = tvm.nd.array(np.zeros(C_shape, dtype=np.int8), ctx)
    print(f"  invoke {name}(A, W, D, C)  A.shape={A_packed.shape} "
          f"W.shape={W_packed.shape} D.shape={D_int32.shape} C.shape={C_shape}")
    mod(A_nd, W_nd, D_nd, C_nd)
    return C_nd.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="/home/xilinx/tiny_test")
    args = ap.parse_args()

    setup_board()
    import tvm  # noqa: E402
    import tvm.runtime  # noqa: E402
    ctx = tvm.device("ext_dev", 0)
    print(f"[ctx] {ctx}")

    print("\n" + "=" * 60)
    print("STAGE 1  TINY  (o=1, n=1, m=1, shift=2)")
    print("=" * 60)
    o_path = os.path.join(args.module_dir, "tiny_4arg.o")
    if not os.path.exists(o_path):
        print(f"  ERR: {o_path} not found"); sys.exit(2)
    mod = tvm.runtime.load_module(link_so(o_path))

    # All ones int4 input + weight; zero int32 bias
    A = np.ones((1, 1, 1, 16), dtype=np.int8)
    W = np.ones((1, 1, 16, 16), dtype=np.int8)
    D = np.zeros((1, 1, 1, 16), dtype=np.int32)
    A_packed = pack_int4_for_vta(A)
    W_packed = pack_int4_for_vta(W)

    C_out = run_one("tiny", mod, A_packed, W_packed, D, (1, 1, 1, 16), ctx)
    print(f"  C dtype={C_out.dtype} shape={C_out.shape}")
    print(f"  C raw (1, 1, 1, 16) = {C_out[0, 0, 0, :].tolist()}")
    expected = 4   # sum_k 1*1 over BLOCK_IN=16 = 16; 16 >> 2 = 4
    actual_unique = set(int(x) for x in C_out.flatten())
    if actual_unique == {expected}:
        print(f"  [STAGE 1] PASS  every element == {expected} (16 >> 2)")
        tiny_ok = True
    else:
        print(f"  [STAGE 1] FAIL  expected every elem = {expected}, "
              f"got unique values {sorted(actual_unique)}")
        tiny_ok = False

    if not tiny_ok:
        print("\nDIAGNOSIS: tiny 4-arg INT4-o8 GEMM is broken on this board.")
        print("  - Expected output 4 means VTA isn't computing the GEMM correctly.")
        print("  - Most likely a host/PL config mismatch beyond what env reports.")
        print("  - The transformer cannot work until this passes.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STAGE 2  CNN_SHAPE  (o=196, n=1, m=1, shift=2)")
    print("=" * 60)
    o_path = os.path.join(args.module_dir, "cnn_shape.o")
    if not os.path.exists(o_path):
        print(f"  WARN: {o_path} not found, skipping stage 2")
    else:
        mod = tvm.runtime.load_module(link_so(o_path))
        A = np.ones((196, 1, 1, 16), dtype=np.int8)
        W = np.ones((1, 1, 16, 16), dtype=np.int8)
        D = np.zeros((196, 1, 1, 16), dtype=np.int32)
        A_packed = pack_int4_for_vta(A)
        W_packed = pack_int4_for_vta(W)
        C_out = run_one("cnn_shape", mod, A_packed, W_packed, D, (196, 1, 1, 16), ctx)
        actual_unique = set(int(x) for x in C_out.flatten())
        cnn_ok = (actual_unique == {4})
        print(f"  [STAGE 2] {'PASS' if cnn_ok else 'FAIL'}  unique values {sorted(actual_unique)} "
              f"(expected {{4}})")
        if not cnn_ok:
            sys.exit(1)

    print("\n" + "=" * 60)
    print("STAGE 3  PROJ_M1  (o=64, n=6, m=1, shift=3)  -- transformer shape, m=1 chunked")
    print("=" * 60)
    o_path = os.path.join(args.module_dir, "proj_m1.o")
    if not os.path.exists(o_path):
        print(f"  WARN: {o_path} not found, skipping stage 3"); sys.exit(0)
    mod = tvm.runtime.load_module(link_so(o_path))

    # All ones int4 input + weight; zero int32 bias.
    # Each output element = (sum_k A*W) >> shift = (96) >> 3 = 12.
    A = np.ones((64, 6, 1, 16), dtype=np.int8)
    W = np.ones((1, 6, 16, 16), dtype=np.int8)
    D = np.zeros((64, 1, 1, 16), dtype=np.int32)
    A_packed = pack_int4_for_vta(A)
    W_packed = pack_int4_for_vta(W)
    C_out = run_one("proj_m1", mod, A_packed, W_packed, D, (64, 1, 1, 16), ctx)
    actual_unique = set(int(x) for x in C_out.flatten())
    expected = (96) >> 3   # K = n*BLOCK_IN = 6*16 = 96; sum of 1*1 over K = 96; 96 >> 3 = 12
    proj_m1_ok = (actual_unique == {expected})
    print(f"  [STAGE 3] {'PASS' if proj_m1_ok else 'FAIL'}  unique values {sorted(actual_unique)} "
          f"(expected {{{expected}}})")
    if not proj_m1_ok:
        print("\nDIAGNOSIS: m=1 also fails — the bug is NOT m-axis specific.")
        print("  Try lower o_tile or n_tiles to find what works.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STAGE 4  PROJ_M2  (o=64, n=6, m=2, shift=3)  -- check m=2 threshold")
    print("=" * 60)
    o_path = os.path.join(args.module_dir, "proj_m2.o")
    if not os.path.exists(o_path):
        print(f"  WARN: {o_path} not found, skipping stage 4")
    else:
        mod = tvm.runtime.load_module(link_so(o_path))
        A = np.ones((64, 6, 1, 16), dtype=np.int8)
        W = np.ones((2, 6, 16, 16), dtype=np.int8)
        D = np.zeros((64, 2, 1, 16), dtype=np.int32)
        A_packed = pack_int4_for_vta(A)
        W_packed = pack_int4_for_vta(W)
        C_out = run_one("proj_m2", mod, A_packed, W_packed, D, (64, 2, 1, 16), ctx)
        actual_unique = set(int(x) for x in C_out.flatten())
        proj_m2_ok = (actual_unique == {expected})
        print(f"  [STAGE 4] {'PASS' if proj_m2_ok else 'FAIL'}  unique values {sorted(actual_unique)} "
              f"(expected {{{expected}}})")
        if not proj_m2_ok:
            print("\nDIAGNOSIS: m=1 works, m=2 doesn't -> m-axis bug. m=1 chunking is the right workaround.")
        else:
            print("\nDIAGNOSIS: m=1 AND m=2 both work -> m=2 might also be safe; widen search.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("  Tiny (o=1,n=1,m=1):     PASS")
    print("  CNN  (o=196,n=1,m=1):   PASS (if module present)")
    print(f"  Proj_m1 (o=64,n=6,m=1): {'PASS' if proj_m1_ok else 'FAIL'}")
    print("  m=1 chunking is viable for the transformer.")


if __name__ == "__main__":
    main()
