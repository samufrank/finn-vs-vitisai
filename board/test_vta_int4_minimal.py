#!/usr/bin/env python3
"""Minimal VTA INT4 GEMM+bias+shift+clip test — board-side runner.

Two-phase design:
  Phase 1 (HOST):  Run with --compile to cross-compile .o modules.
  Phase 2 (BOARD): Run without --compile to load .o modules and test.

Host usage:
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 ../finn-vs-vitisai/board/test_vta_int4_minimal.py \
            --compile --output-dir ./vta_export/int4_minimal_test/
    scp -r ./vta_export/int4_minimal_test/ xilinx@192.168.3.1:/home/xilinx/int4_minimal_test/

Board usage (as root):
    python3 test_vta_int4_minimal.py --test-dir /home/xilinx/int4_minimal_test/

Test matrix (all single-tile 16×16):
    T1: GEMM(16) + bias(5)  >> 2, clip[0,7]  → 5    (normal path)
    T2: GEMM(16) + bias(30) >> 2, clip[0,7]  → 7    (upper clip)
    T3: GEMM(16) + bias(-20)>> 2, clip[0,7]  → 0    (lower clip / ReLU)
    T4: GEMM(16) >> 1,           clip[0,7]   → 7    (no-bias 3-arg module)
    T5: GEMM(16) + bias(5),  s=0, clip[-8,7] → 7    (probe: 21 clipped)
    T6: GEMM(16) + bias(-10),s=0, clip[-8,7] → 6    (probe: in range)
    T7: GEMM(16) + bias(-25),s=0, clip[-8,7] → -8   (probe: neg clip)
    T8: Repeat T1 ×3 for determinism
"""
import sys
import os
import argparse
import json
import numpy as np


# ============================================================
# INT4 nibble packing for VTA
# ============================================================

def pack_int4_for_vta(vals_int8):
    """Pack int8 array of int4 values into VTA nibble format.

    Packing is contiguous across the entire flattened tensor.
    Two values per byte: lo nibble = even flat index, hi = odd.

    Input:  int8 array, any shape.
    Output: int8 array, same shape.  First half of flat buffer holds
            packed nibble pairs; second half is zero-padded.
    """
    vals = np.asarray(vals_int8, dtype=np.int8)
    flat = vals.flatten()
    n = len(flat)
    lo = flat[0::2].view(np.uint8) & 0xF
    hi = flat[1::2].view(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[:n // 2] = packed
    return out.reshape(vals.shape)


def unpack_int4_from_vta(packed_int8):
    """Unpack VTA nibble-packed int4 output to one-value-per-element int8.

    Inverse of pack_int4_for_vta.  First half of flat buffer holds
    packed nibble pairs.

    Input:  int8 array, any shape.
    Output: int8 array, same shape.  Sign-extended int4 in [-8, 7].
    """
    raw = np.asarray(packed_int8, dtype=np.int8)
    flat = raw.flatten()
    n = len(flat)
    packed_bytes = flat[:n // 2].view(np.uint8)
    lo = (packed_bytes & 0xF).astype(np.int8)
    hi = ((packed_bytes >> 4) & 0xF).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.int8)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out.reshape(raw.shape)


# ============================================================
# Phase 1: Cross-compile on host
# ============================================================

def compile_modules(output_dir):
    """Cross-compile VTA modules on host for board deployment."""
    import tvm
    from tvm import te
    import vta

    env = vta.get_env()
    print(f"VTA env: inp={env.inp_dtype}, wgt={env.wgt_dtype}, "
          f"acc={env.acc_dtype}, BLOCK={env.BLOCK_IN}")
    print(f"TARGET={env.TARGET}")

    os.makedirs(output_dir, exist_ok=True)

    if env.TARGET in ("sim", "tsim"):
        host_target = tvm.target.Target("llvm")
    else:
        host_target = tvm.target.arm_cpu("ultra96")

    o, n, m = 1, 1, 1

    def _build_bias(shift, clip_lo, clip_hi, name):
        """GEMM + bias ADD + SHR + CLIP. 4-arg: A, B, D, C."""
        A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),
                           name='A', dtype=env.inp_dtype)
        B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                           name='B', dtype=env.wgt_dtype)
        D = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT),
                           name='D', dtype=env.acc_dtype)
        A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN),
                           lambda *i: A(*i), 'A_buf')
        B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                           lambda *i: B(*i), 'B_buf')
        D_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
                           lambda *i: D(*i), 'D_buf')
        ko = te.reduce_axis((0, n), 'ko')
        ki = te.reduce_axis((0, env.BLOCK_IN), 'ki')
        C_buf = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT),
            lambda bo, co, bi, ci: te.sum(
                A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
                B_buf[co, ko, ci, ki].astype(env.acc_dtype),
                axis=[ko, ki]), name='C_buf')
        C_add = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: C_buf(*i) + D_buf(*i), name='C_add')
        C_shr = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: C_add(*i) >> tvm.tir.const(shift, env.acc_dtype),
            name='C_shr')
        C_clo = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm.te.max(C_shr(*i),
                tvm.tir.const(clip_lo, env.acc_dtype)), name='C_clo')
        C_chi = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm.te.min(C_clo(*i),
                tvm.tir.const(clip_hi, env.acc_dtype)), name='C_chi')
        C = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: C_chi(*i).astype(env.inp_dtype), name='C')
        s = te.create_schedule(C.op)
        s[A_buf].set_scope(env.inp_scope)
        s[B_buf].set_scope(env.wgt_scope)
        s[D_buf].set_scope(env.acc_scope)
        s[C_buf].set_scope(env.acc_scope)
        s[C_add].set_scope(env.acc_scope)
        s[C_shr].set_scope(env.acc_scope)
        s[C_clo].set_scope(env.acc_scope)
        s[C_chi].set_scope(env.acc_scope)
        s[C_buf].reorder(ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
            s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
        s[A_buf].compute_at(s[C_buf], ko)
        s[B_buf].compute_at(s[C_buf], ko)
        s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
        s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
        s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)
        s[D_buf].pragma(s[D_buf].op.axis[0], env.dma_copy)
        s[C_add].pragma(s[C_add].op.axis[0], env.alu)
        s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
        s[C_clo].pragma(s[C_clo].op.axis[0], env.alu)
        s[C_chi].pragma(s[C_chi].op.axis[0], env.alu)
        s[C].pragma(s[C].op.axis[0], env.dma_copy)
        mod = vta.build(s, [A, B, D, C], tvm.target.vta(),
                        host_target, name='my_gemm')
        out = os.path.join(output_dir, f'{name}.o')
        mod.save(out)
        print(f"  {name}.o saved ({os.path.getsize(out)} bytes)")

    def _build_nobias(shift, clip_lo, clip_hi, name):
        """GEMM + SHR + CLIP only. 3-arg: A, B, C."""
        A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),
                           name='A', dtype=env.inp_dtype)
        B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                           name='B', dtype=env.wgt_dtype)
        A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN),
                           lambda *i: A(*i), 'A_buf')
        B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                           lambda *i: B(*i), 'B_buf')
        ko = te.reduce_axis((0, n), 'ko')
        ki = te.reduce_axis((0, env.BLOCK_IN), 'ki')
        C_buf = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT),
            lambda bo, co, bi, ci: te.sum(
                A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
                B_buf[co, ko, ci, ki].astype(env.acc_dtype),
                axis=[ko, ki]), name='C_buf')
        C_shr = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: C_buf(*i) >> tvm.tir.const(shift, env.acc_dtype),
            name='C_shr')
        C_clo = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm.te.max(C_shr(*i),
                tvm.tir.const(clip_lo, env.acc_dtype)), name='C_clo')
        C_chi = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm.te.min(C_clo(*i),
                tvm.tir.const(clip_hi, env.acc_dtype)), name='C_chi')
        C = te.compute((o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: C_chi(*i).astype(env.inp_dtype), name='C')
        s = te.create_schedule(C.op)
        s[A_buf].set_scope(env.inp_scope)
        s[B_buf].set_scope(env.wgt_scope)
        s[C_buf].set_scope(env.acc_scope)
        s[C_shr].set_scope(env.acc_scope)
        s[C_clo].set_scope(env.acc_scope)
        s[C_chi].set_scope(env.acc_scope)
        s[C_buf].reorder(ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
            s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
        s[A_buf].compute_at(s[C_buf], ko)
        s[B_buf].compute_at(s[C_buf], ko)
        s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
        s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
        s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)
        s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
        s[C_clo].pragma(s[C_clo].op.axis[0], env.alu)
        s[C_chi].pragma(s[C_chi].op.axis[0], env.alu)
        s[C].pragma(s[C].op.axis[0], env.dma_copy)
        mod = vta.build(s, [A, B, C], tvm.target.vta(),
                        host_target, name='my_gemm')
        out = os.path.join(output_dir, f'{name}.o')
        mod.save(out)
        print(f"  {name}.o saved ({os.path.getsize(out)} bytes)")

    print("Compiling modules...")
    _build_bias(2, 0, 7, 'mod_bias')      # T1-T3
    _build_nobias(1, 0, 7, 'mod_nobias')   # T4
    _build_bias(0, -8, 7, 'mod_probe')     # T5-T7

    # Save test config so board runner knows tile sizes
    test_config = {
        'o': o, 'n': n, 'm': m,
        'BLOCK_IN': env.BLOCK_IN, 'BLOCK_OUT': env.BLOCK_OUT,
        'BATCH': env.BATCH,
        'modules': {
            'mod_bias':   {'file': 'mod_bias.o',   'n_args': 4},
            'mod_nobias': {'file': 'mod_nobias.o', 'n_args': 3},
            'mod_probe':  {'file': 'mod_probe.o',  'n_args': 4},
        },
    }
    cfg_path = os.path.join(output_dir, 'test_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    print(f"  test_config.json saved")
    print(f"\nDone. Copy to board:")
    print(f"  scp -r {output_dir} xilinx@192.168.3.1:/home/xilinx/int4_minimal_test/")


# ============================================================
# Phase 2: Run on board
# ============================================================

def run_tests(test_dir):
    """Load pre-compiled .o modules and run test matrix on VTA hardware."""
    import ctypes

    # ---- Board setup ----
    BITSTREAM = '1x16_i4w4a32_15_14_17_17.bit'

    stale = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale):
            os.remove(stale)
    except Exception:
        pass

    bitstream_path = None
    for candidate in [
        f'/root/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
        f'/home/xilinx/.vta_cache/ultra96/0_0_2/{BITSTREAM}',
    ]:
        if os.path.exists(candidate):
            bitstream_path = candidate
            break

    if bitstream_path is None:
        print(f"ERROR: Bitstream {BITSTREAM} not found")
        sys.exit(1)

    print(f"Loading bitstream: {bitstream_path}")
    try:
        from pynq import Overlay
        overlay = Overlay(bitstream_path)
        print(f"  Overlay loaded, IPs: {list(overlay.ip_dict.keys())}")
    except Exception as e:
        print(f"  Overlay load failed: {e}")
        print(f"  Continuing (may already be loaded)...")

    vta_lib = None
    for candidate in [
        '/home/xilinx/tvm-src/build/libvta.so',
        os.path.join(os.environ.get('TVM_HOME', ''), 'build/libvta.so'),
    ]:
        if os.path.exists(candidate):
            vta_lib = candidate
            break
    if vta_lib is None:
        print("ERROR: libvta.so not found")
        sys.exit(1)
    print(f"Loading VTA runtime: {vta_lib}")
    ctypes.CDLL(vta_lib, ctypes.RTLD_GLOBAL)

    import tvm
    import tvm.runtime

    ctx = tvm.device("ext_dev", 0)

    # ---- Load test config ----
    cfg_path = os.path.join(test_dir, 'test_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    BLOCK_IN = cfg['BLOCK_IN']
    BLOCK_OUT = cfg['BLOCK_OUT']
    o, n, m = cfg['o'], cfg['n'], cfg['m']

    shape_a = (o, n, cfg['BATCH'], BLOCK_IN)
    shape_b = (m, n, BLOCK_OUT, BLOCK_IN)
    shape_d = (o, m, cfg['BATCH'], BLOCK_OUT)
    shape_c = (o, m, cfg['BATCH'], BLOCK_OUT)

    # ---- Load modules ----
    print("Loading modules...")
    mods = {}
    for name, info in cfg['modules'].items():
        mod_file = info['file']
        # Try .so then .o
        so_file = mod_file.replace('.o', '.so')
        mod_path = os.path.join(test_dir, so_file)
        if not os.path.exists(mod_path):
            mod_path = os.path.join(test_dir, mod_file)
        print(f"  {name}: {mod_path}")
        loaded = tvm.runtime.load_module(mod_path)
        mods[name] = (loaded, info['n_args'])

    # ---- Common inputs (packed for VTA int4 nibble format) ----
    # All-ones in int4: each byte must be 0x11 (lo=1, hi=1), not 0x01
    a_ones_unpacked = np.ones(shape_a, dtype=np.int8)
    b_ones_unpacked = np.ones(shape_b, dtype=np.int8)
    a_ones = pack_int4_for_vta(a_ones_unpacked)
    b_ones = pack_int4_for_vta(b_ones_unpacked)

    # Pre-load weights to VTA (packed, same for all tests)
    b_nd = tvm.nd.array(b_ones, ctx)

    # ---- Test runner ----
    def run_one(name, mod_name, bias_val, expected_val):
        mod, n_args = mods[mod_name]
        a_nd = tvm.nd.array(a_ones.copy(), ctx)
        c_nd = tvm.nd.array(np.zeros(shape_c, dtype=np.int8), ctx)

        if n_args == 4:
            d_np = np.full(shape_d, bias_val, dtype=np.int32)
            d_nd = tvm.nd.array(d_np, ctx)
            mod(a_nd, b_nd, d_nd, c_nd)
        else:
            mod(a_nd, b_nd, c_nd)

        got_packed = c_nd.numpy().flatten()
        got_unpacked = unpack_int4_from_vta(got_packed)[:16]

        expected = np.full(16, expected_val, dtype=np.int8)
        match = np.all(got_unpacked == expected)
        status = "PASS" if match else "FAIL"

        print(f"\n{'=' * 60}")
        print(f"TEST: {name}  [{status}]")
        print(f"{'=' * 60}")
        print(f"  Raw packed:  {got_packed[:8].tolist()} (first 8 bytes)")
        print(f"  Unpacked:    {got_unpacked.tolist()}")
        print(f"  Expected:    {expected.tolist()}")

        if not match:
            diff = got_unpacked.astype(np.int32) - expected.astype(np.int32)
            n_wrong = int(np.sum(diff != 0))
            print(f"  MISMATCH: {n_wrong}/16 wrong")
            print(f"  Diff:    {diff.tolist()}")

        return match

    results = []

    # ---- Cold-start isolation: T1 run twice before anything else ----
    results.append(run_one(
        "T1a (first VTA call ever): GEMM(16)+bias(5) >>2 → 5",
        'mod_bias', 5, 5))

    results.append(run_one(
        "T1b (second call, same params): GEMM(16)+bias(5) >>2 → 5",
        'mod_bias', 5, 5))

    # T2-T7
    results.append(run_one(
        "T2: GEMM(16)+bias(30) >>2, clip[0,7] → 7 (clipped)",
        'mod_bias', 30, 7))

    results.append(run_one(
        "T3: GEMM(16)+bias(-20) >>2, clip[0,7] → 0 (ReLU)",
        'mod_bias', -20, 0))

    results.append(run_one(
        "T4: GEMM(16) >>1, clip[0,7] → 7 (no bias)",
        'mod_nobias', 0, 7))

    results.append(run_one(
        "T5: GEMM(16)+bias(5), s=0, clip[-8,7] → 7 (probe, clipped)",
        'mod_probe', 5, 7))

    results.append(run_one(
        "T6: GEMM(16)+bias(-10), s=0, clip[-8,7] → 6 (probe, in range)",
        'mod_probe', -10, 6))

    results.append(run_one(
        "T7: GEMM(16)+bias(-25), s=0, clip[-8,7] → -8 (probe, neg clip)",
        'mod_probe', -25, -8))

    # ---- T1 again at the very end, after all other tests ----
    results.append(run_one(
        "T1c (after T2-T7): GEMM(16)+bias(5) >>2 → 5",
        'mod_bias', 5, 5))

    # T8: Determinism — repeat T1 three more times
    print(f"\n{'=' * 60}")
    print("TEST: T8 — Determinism (repeat T1 three times)")
    print(f"{'=' * 60}")
    mod, _ = mods['mod_bias']
    for rep in range(3):
        a_nd = tvm.nd.array(a_ones.copy(), ctx)
        d_nd = tvm.nd.array(np.full(shape_d, 5, dtype=np.int32), ctx)
        c_nd = tvm.nd.array(np.zeros(shape_c, dtype=np.int8), ctx)
        mod(a_nd, b_nd, d_nd, c_nd)
        got_packed = c_nd.numpy().flatten()
        got_unpacked = unpack_int4_from_vta(got_packed)[:16]
        ok = np.all(got_unpacked == 5)
        print(f"  Rep {rep}: {got_unpacked.tolist()}  {'OK' if ok else 'WRONG'}")
        if not ok:
            results.append(False)

    # ---- Diagnosis ----
    print(f"\n{'=' * 60}")
    print("COLD-START ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  T1a (1st call):     {'PASS' if results[0] else 'FAIL'}")
    print(f"  T1b (2nd call):     {'PASS' if results[1] else 'FAIL'}")
    print(f"  T1c (after T2-T7):  {'PASS' if results[8] else 'FAIL'}")
    if not results[0] and results[1] and results[8]:
        print("  → Cold-start: only first VTA call affected.")
        print("    Workaround: one dummy VTA call before real inference.")
    elif not results[0] and not results[1]:
        print("  → NOT cold-start: repeatable error on T1.")
        print("    This is a real bug in the bias=5,shift=2 path.")
    elif results[0]:
        print("  → All T1 calls passed. No cold-start issue.")

    # ---- Summary ----
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    print(f"{'=' * 60}")

    if n_pass == n_total:
        print("All tests passed. VTA INT4 pipeline is functional.")
    else:
        n_fail = n_total - n_pass
        print(f"FAILURES: {n_fail}/{n_total}.")
        if n_fail == 1 and not results[0] and all(results[1:]):
            print("Only the very first VTA call failed (cold-start).")
            print("All subsequent calls are correct.")
        sys.exit(1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Minimal VTA INT4 GEMM+bias+shift+clip test')
    parser.add_argument('--compile', action='store_true',
                        help='Phase 1: cross-compile modules on host')
    parser.add_argument('--output-dir', default='./int4_minimal_test/',
                        help='Output dir for compiled modules (host phase)')
    parser.add_argument('--test-dir',
                        default='/home/xilinx/int4_minimal_test/',
                        help='Dir with compiled modules (board phase)')
    args = parser.parse_args()

    print("=" * 60)
    print("VTA INT4 Minimal Pipeline Test")
    print("=" * 60)

    if args.compile:
        compile_modules(args.output_dir)
    else:
        run_tests(args.test_dir)


if __name__ == '__main__':
    main()
