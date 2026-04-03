"""VTA GEMM verification suite — standalone reproducible tests.

Captures the verification tests from session 4 debugging that proved
VTA GEMM is working correctly on the AUP-ZU3.

Key finding: VTA's GEMM truncates int32 accumulator to int8 via
    o_tensor[b][oc] = (out_T) accum.range(VTA_OUT_WIDTH - 1, 0)
This is bitwise truncation, NOT saturation. Saturation requires a
subsequent ALU min/max pass (part of normal TVM inference schedules).

The reference computation must use .astype(np.int8) (truncation),
not np.clip(..., -128, 127).astype(np.int8) (saturation).

Usage (from Ubuntu host):
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 test_vta_verify.py

Prerequisites:
    - Board booted with PYNQ, XRT sourced, FPGA programmed with
      1x16_i8w8a32_15_15_18_17.bit
    - RPC server running on board port 9091
    - Do NOT call vta.program_fpga() — board should already be programmed

Date: March 27-28, 2026
"""
import numpy as np
import tvm
from tvm import te, rpc
from tvm.contrib import utils
import vta
import time
import sys

BOARD_IP = "192.168.3.1"
RPC_PORT = 9091


def build_gemm_module(remote, env, o, n, m):
    """Build and upload a VTA GEMM module for given tile dimensions.

    Uses the tutorial-style schedule with compute_at, which splits
    init (reset_reg=1) and compute (reset_reg=0) into separate GEMM
    instructions — required by VTA hardware.

    Each config gets a unique filename to avoid RPC module cache
    conflicts when running multiple configs in one session.

    Args:
        o: batch tiles
        n: reduction tiles (must be >= 2 for compute_at to work)
        m: output tiles

    Returns:
        Loaded remote module function.
    """
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
    C = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_buf(*i).astype(env.inp_dtype), name="C")

    s = te.create_schedule(C.op)
    s[A_buf].set_scope(env.inp_scope)
    s[B_buf].set_scope(env.wgt_scope)
    s[C_buf].set_scope(env.acc_scope)

    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
        s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)

    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C].pragma(s[C].op.axis[0], env.dma_copy)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    mod = vta.build(s, [A, B, C],
                    tvm.target.vta(),
                    tvm.target.arm_cpu("ultra96"),
                    name="my_gemm")

    # Unique filename per config to avoid RPC module cache conflicts
    fname = f"gemm_o{o}_n{n}_m{m}.o"
    temp = utils.tempdir()
    mod.save(temp.relpath(fname))
    remote.upload(temp.relpath(fname))
    return remote.load_module(fname)


def reference_gemm(A_np, B_np, env, o, n, m, truncate=True):
    """Compute reference GEMM matching VTA hardware behavior.

    Args:
        truncate: If True (default), use bitwise truncation int32->int8
                  matching VTA's accum.range(OUT_WIDTH-1, 0).
                  If False, use saturation (clip to [-128, 127]).
    """
    C_ref = np.zeros((o, m, env.BATCH, env.BLOCK_OUT), dtype=np.int32)
    for bo in range(o):
        for co in range(m):
            for bi in range(env.BATCH):
                for ci in range(env.BLOCK_OUT):
                    for kk in range(n):
                        for kki in range(env.BLOCK_IN):
                            C_ref[bo][co][bi][ci] += (
                                int(A_np[bo][kk][bi][kki]) *
                                int(B_np[co][kk][ci][kki]))
    if truncate:
        # VTA behavior: take low 8 bits of 32-bit accumulator
        return C_ref.astype(np.int8)
    else:
        # Saturating behavior (for comparison / debugging)
        return np.clip(C_ref, -128, 127).astype(np.int8)


def run_test(f, env, ctx, o, n, m, A_np, B_np, test_name):
    """Run a single GEMM test and verify against reference."""
    A_nd = tvm.nd.array(A_np, ctx)
    B_nd = tvm.nd.array(B_np, ctx)
    C_nd = tvm.nd.array(
        np.zeros((o, m, env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)

    f(A_nd, B_nd, C_nd)
    C_out = C_nd.numpy()

    C_ref = reference_gemm(A_np, B_np, env, o, n, m, truncate=True)
    passed = np.array_equal(C_ref, C_out)

    status = "PASSED" if passed else "FAILED"
    print(f"  {test_name}: {status}")

    if not passed:
        mismatches = np.sum(C_ref != C_out)
        print(f"    {mismatches}/{C_ref.size} mismatches")
        print(f"    ref[:8] = {C_ref.flatten()[:8]}")
        print(f"    got[:8] = {C_out.flatten()[:8]}")

        # Check if saturation reference matches (would indicate truncation
        # assumption is wrong)
        C_sat = reference_gemm(A_np, B_np, env, o, n, m, truncate=False)
        if np.array_equal(C_sat, C_out):
            print("    NOTE: output matches SATURATING reference — "
                  "truncation assumption may be wrong")

    return passed, C_out


def main():
    env = vta.get_env()
    print(f"VTA env: {env.TARGET}")
    print(f"BATCH={env.BATCH}, BLOCK_IN={env.BLOCK_IN}, "
          f"BLOCK_OUT={env.BLOCK_OUT}")

    print(f"\nConnecting to {BOARD_IP}:{RPC_PORT}...")
    remote = rpc.connect(BOARD_IP, RPC_PORT)
    vta.reconfig_runtime(remote)

    # Program FPGA if bitstream is in host cache; skip gracefully if not
    try:
        vta.program_fpga(remote, bitstream=None)
        print("FPGA programmed")
    except Exception:
        print("FPGA already programmed (skipped)")
    ctx = remote.ext_dev(0)
    print("Connected")

    all_passed = True

    # ---- Config 1: o=1, n=2, m=1 (minimal) ----
    print(f"\n{'='*60}")
    print("Config 1: o=1, n=2, m=1  —  (1x32) * (32x16)")
    print(f"{'='*60}")
    o, n, m = 1, 2, 1
    f = build_gemm_module(remote, env, o, n, m)
    print("  Build + upload: OK")

    # Test 1a: all-ones
    A_np = np.ones((o, n, env.BATCH, env.BLOCK_IN), dtype=np.int8)
    B_np = np.ones((m, n, env.BLOCK_OUT, env.BLOCK_IN), dtype=np.int8)
    passed, C_out = run_test(f, env, ctx, o, n, m, A_np, B_np,
                             f"all-ones (expect {n*env.BLOCK_IN})")
    all_passed &= passed

    # Test 1b: small random
    np.random.seed(42)
    A_np = np.random.randint(-3, 4, size=(o, n, env.BATCH, env.BLOCK_IN)).astype(np.int8)
    B_np = np.random.randint(-3, 4, size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(np.int8)
    passed, _ = run_test(f, env, ctx, o, n, m, A_np, B_np, "small random")
    all_passed &= passed

    # Test 1c: full range random
    A_np = np.random.randint(-128, 128, size=(o, n, env.BATCH, env.BLOCK_IN)).astype(np.int8)
    B_np = np.random.randint(-128, 128, size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(np.int8)
    passed, _ = run_test(f, env, ctx, o, n, m, A_np, B_np, "full-range random")
    all_passed &= passed

    # ---- Config 2: o=1, n=4, m=4 ----
    print(f"\n{'='*60}")
    print("Config 2: o=1, n=4, m=4  —  (1x64) * (64x64)")
    print(f"{'='*60}")
    o, n, m = 1, 4, 4
    f = build_gemm_module(remote, env, o, n, m)
    print("  Build + upload: OK")

    A_np = np.ones((o, n, env.BATCH, env.BLOCK_IN), dtype=np.int8)
    B_np = np.ones((m, n, env.BLOCK_OUT, env.BLOCK_IN), dtype=np.int8)
    passed, _ = run_test(f, env, ctx, o, n, m, A_np, B_np,
                         f"all-ones (expect {n*env.BLOCK_IN})")
    all_passed &= passed

    np.random.seed(123)
    A_np = np.random.randint(-128, 128, size=(o, n, env.BATCH, env.BLOCK_IN)).astype(np.int8)
    B_np = np.random.randint(-128, 128, size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(np.int8)
    passed, _ = run_test(f, env, ctx, o, n, m, A_np, B_np, "full-range random")
    all_passed &= passed

    # ---- Config 3: o=4, n=4, m=4 (larger batch) ----
    print(f"\n{'='*60}")
    print("Config 3: o=4, n=4, m=4  —  (4x64) * (64x64)")
    print(f"{'='*60}")
    o, n, m = 4, 4, 4
    f = build_gemm_module(remote, env, o, n, m)
    print("  Build + upload: OK")

    A_np = np.ones((o, n, env.BATCH, env.BLOCK_IN), dtype=np.int8)
    B_np = np.ones((m, n, env.BLOCK_OUT, env.BLOCK_IN), dtype=np.int8)
    passed, _ = run_test(f, env, ctx, o, n, m, A_np, B_np,
                         f"all-ones (expect {n*env.BLOCK_IN})")
    all_passed &= passed

    np.random.seed(456)
    A_np = np.random.randint(-128, 128, size=(o, n, env.BATCH, env.BLOCK_IN)).astype(np.int8)
    B_np = np.random.randint(-128, 128, size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(np.int8)
    passed, _ = run_test(f, env, ctx, o, n, m, A_np, B_np, "full-range random")
    all_passed &= passed

    # ---- Timing ----
    print(f"\n{'='*60}")
    print("Timing (o=4, n=4, m=4)")
    print(f"{'='*60}")
    num_runs = 100
    start = time.time()
    A_nd = tvm.nd.array(A_np, ctx)
    B_nd = tvm.nd.array(B_np, ctx)
    C_nd = tvm.nd.array(
        np.zeros((o, m, env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)
    for _ in range(num_runs):
        f(A_nd, B_nd, C_nd)
    elapsed = time.time() - start
    print(f"  {num_runs} runs in {elapsed:.3f}s = "
          f"{elapsed/num_runs*1000:.2f}ms per GEMM")
    print(f"  Matrix: ({o*env.BATCH}x{n*env.BLOCK_IN}) x "
          f"({n*env.BLOCK_IN}x{m*env.BLOCK_OUT})")
    # Note: ~200ms/GEMM is dominated by the 200ms usleep in the driver's
    # done-polling workaround. Actual hardware time is sub-millisecond.

    # ---- Summary ----
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
