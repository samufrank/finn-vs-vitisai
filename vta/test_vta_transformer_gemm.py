"""VTA Transformer GEMM verification — tests attention-sized matmuls.

Follows test_vta_verify.py patterns exactly. Tests whether transformer-
sized GEMMs work on VTA hardware before attempting full inference.

Key concern: the attention matmul (Q @ K^T) has TWO runtime activation
inputs, unlike MLP GEMMs where one input is a static weight tensor.
VTA was designed around the weight-times-activation pattern. This script
verifies that loading one activation tensor into wgt_scope still works
correctly — VTA hardware doesn't care about the semantic difference.

Raw GEMM only (no ALU shift+clip) — sufficient for dimension verification.
Shift+clip already works from the MLP/CNN pipeline.

Dimensions tested (d_model=64, nhead=4, d_head=16, seq_len=32, d_ff=128):
  QKV_proj:  (32, 64) @ (64, 16)  — standard weight GEMM
  Q_KT:     (32, 16) @ (16, 32)  — two runtime activations
  attn_V:   (32, 32) @ (32, 16)  — two runtime activations
  out_proj: (32, 64) @ (64, 64)  — standard weight GEMM
  FFN_1:    (32, 64) @ (64, 128) — standard weight GEMM
  FFN_2:    (32, 128) @ (128, 64) — standard weight GEMM

Usage (from Ubuntu host):
    cd ~/dev/CEN571-final/finn-vs-vitisai
    PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python \
        python3 vta/test_vta_transformer_gemm.py

    CPU-only dimension check (no board needed):
        python3 vta/test_vta_transformer_gemm.py --cpu-only

Prerequisites (for board tests):
    - Board booted with PYNQ, bitstream loaded
    - RPC server running: python3 -m vta.exec.rpc_server
"""
import numpy as np
import sys
import time

BOARD_IP = "192.168.3.1"
RPC_PORT = 9091

# VTA tile sizes (must match bitstream config)
BLOCK_IN = 16
BLOCK_OUT = 16


def pad_to_multiple(dim, block):
    return ((dim + block - 1) // block) * block


# ============================================================
# Test case definitions
# ============================================================

# (name, input_rows, reduction_dim, output_cols, is_act_x_act, description)
TRANSFORMER_TESTS = [
    ("QKV_proj",  32,  64,  16, False,
     "Q/K/V projection per head: (seq, d_model) @ (d_model, d_head)"),
    ("Q_KT",     32,  16,  32, True,
     "Attention Q@K^T per head: (seq, d_head) @ (d_head, seq)"),
    ("attn_V",   32,  32,  16, True,
     "Attention weights @ V per head: (seq, seq) @ (seq, d_head)"),
    ("out_proj",  32,  64,  64, False,
     "Output projection: (seq, d_model) @ (d_model, d_model)"),
    ("FFN_1",    32,  64, 128, False,
     "FFN layer 1: (seq, d_model) @ (d_model, d_ff)"),
    ("FFN_2",    32, 128,  64, False,
     "FFN layer 2: (seq, d_ff) @ (d_ff, d_model)"),
]


def compute_tile_dims(input_rows, reduction_dim, output_cols):
    """Compute VTA tile dimensions with padding."""
    o = pad_to_multiple(input_rows, BLOCK_OUT) // BLOCK_OUT
    n = pad_to_multiple(reduction_dim, BLOCK_IN) // BLOCK_IN
    m = pad_to_multiple(output_cols, BLOCK_OUT) // BLOCK_OUT
    return o, n, m


# ============================================================
# CPU-only dimension check
# ============================================================

def cpu_dimension_check():
    """Verify all transformer GEMM dimensions are valid for VTA tiling."""
    print("=" * 60)
    print("CPU-Only Dimension Check")
    print("=" * 60)

    # Known hardware limit from session 9
    MAX_O_WHEN_N_GT1 = 64

    all_ok = True
    for name, rows, red, cols, is_act, desc in TRANSFORMER_TESTS:
        o, n, m = compute_tile_dims(rows, red, cols)
        o_total = o * BLOCK_OUT
        n_total = n * BLOCK_IN

        print(f"\n  {name}: ({rows},{red}) @ ({red},{cols})")
        print(f"    {desc}")
        print(f"    Padded: ({o*BLOCK_OUT},{n*BLOCK_IN}) @ ({n*BLOCK_IN},{m*BLOCK_OUT})")
        print(f"    Tiles:  o={o}, n={n}, m={m}")

        if is_act:
            print(f"    NOTE: Both inputs are runtime activations")

        # Check o-dimension limit
        if n > 1 and o_total > MAX_O_WHEN_N_GT1:
            print(f"    *** WARNING: o={o_total} with n_tiles={n} exceeds "
                  f"hardware limit (~{MAX_O_WHEN_N_GT1})")
            print(f"    *** Will need o-tiling (chunked execution) on hardware")
        else:
            print(f"    o-dimension OK")

        # Quick reference computation sanity check
        np.random.seed(42)
        A_np = np.random.randint(-3, 4,
            size=(o, n, 1, BLOCK_IN)).astype(np.int8)
        B_np = np.random.randint(-3, 4,
            size=(m, n, BLOCK_OUT, BLOCK_IN)).astype(np.int8)
        ref = reference_gemm(A_np, B_np, o, n, m)
        print(f"    Reference range: [{ref.min()}, {ref.max()}]")

    print(f"\n{'=' * 60}")
    print("Dimension check complete")
    print("=" * 60)
    return True


def reference_gemm(A_np, B_np, o, n, m):
    """Reference GEMM matching VTA truncation behavior.

    Matches test_vta_verify.py — explicit nested loops, truncating cast.
    """
    BATCH = 1  # VTA config
    C_ref = np.zeros((o, m, BATCH, BLOCK_OUT), dtype=np.int32)
    for bo in range(o):
        for co in range(m):
            for bi in range(BATCH):
                for ci in range(BLOCK_OUT):
                    for kk in range(n):
                        for kki in range(BLOCK_IN):
                            C_ref[bo][co][bi][ci] += (
                                int(A_np[bo][kk][bi][kki]) *
                                int(B_np[co][kk][ci][kki]))
    # VTA truncation: take low 8 bits of int32 accumulator
    return C_ref.astype(np.int8)


# ============================================================
# Board verification
# ============================================================

def build_gemm_module(remote, env, o, n, m, name_prefix="xfmr"):
    """Build and upload a VTA GEMM module.

    Follows test_vta_verify.py schedule pattern exactly.
    """
    from tvm import te
    from tvm.contrib import utils
    import vta

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN),
                       name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN),
                       name="B", dtype=env.wgt_dtype)

    # Buffer copies (identity) — required for scope assignment
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

    # Cast int32 accumulator to int8 output (truncation)
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

    import tvm
    mod = vta.build(s, [A, B, C],
                    tvm.target.vta(),
                    tvm.target.arm_cpu("ultra96"),
                    name="my_gemm")

    fname = f"{name_prefix}_o{o}_n{n}_m{m}.o"
    temp = utils.tempdir()
    mod.save(temp.relpath(fname))
    remote.upload(temp.relpath(fname))
    return remote.load_module(fname)


def run_board_tests():
    """Run transformer GEMM tests on VTA hardware via RPC."""
    import tvm
    from tvm import te, rpc
    import vta

    env = vta.get_env()
    print(f"VTA env: {env.TARGET}")
    print(f"BATCH={env.BATCH}, BLOCK_IN={env.BLOCK_IN}, "
          f"BLOCK_OUT={env.BLOCK_OUT}")

    print(f"\nConnecting to {BOARD_IP}:{RPC_PORT}...")
    remote = rpc.connect(BOARD_IP, RPC_PORT)
    vta.reconfig_runtime(remote)

    try:
        vta.program_fpga(remote, bitstream=None)
        print("FPGA programmed")
    except Exception:
        print("FPGA already programmed (skipped)")

    ctx = remote.ext_dev(0)
    print("Connected")

    all_passed = True

    for name, rows, red, cols, is_act, desc in TRANSFORMER_TESTS:
        o, n, m = compute_tile_dims(rows, red, cols)

        print(f"\n{'=' * 60}")
        print(f"{name}: ({rows},{red}) @ ({red},{cols}) "
              f"-> tiles o={o} n={n} m={m}"
              f"{'  [act×act]' if is_act else ''}")
        print(f"  {desc}")
        print(f"{'=' * 60}")

        # Check o-dimension limit before building
        MAX_O_WHEN_N_GT1 = 64
        if n > 1 and o * BLOCK_OUT > MAX_O_WHEN_N_GT1:
            print(f"  SKIPPED: o={o * BLOCK_OUT} exceeds hardware limit "
                  f"with n={n} tiles (needs o-tiling)")
            continue

        f = build_gemm_module(remote, env, o, n, m, name_prefix=name)
        print("  Build + upload: OK")

        # Test with small random values (avoid int32 overflow in
        # large reduction dimensions)
        np.random.seed(hash(name) % 2**31)
        A_np = np.random.randint(-3, 4,
            size=(o, n, env.BATCH, env.BLOCK_IN)).astype(np.int8)
        B_np = np.random.randint(-3, 4,
            size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(np.int8)

        A_nd = tvm.nd.array(A_np, ctx)
        B_nd = tvm.nd.array(B_np, ctx)
        C_nd = tvm.nd.array(
            np.zeros((o, m, env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)

        f(A_nd, B_nd, C_nd)
        C_out = C_nd.numpy()

        C_ref = reference_gemm(A_np, B_np, o, n, m)
        passed = np.array_equal(C_ref, C_out)

        status = "PASSED" if passed else "FAILED"
        print(f"  small random: {status}")

        if not passed:
            mismatches = np.sum(C_ref != C_out)
            print(f"    {mismatches}/{C_ref.size} mismatches")
            print(f"    ref[:8] = {C_ref.flatten()[:8]}")
            print(f"    got[:8] = {C_out.flatten()[:8]}")
            all_passed = False
        else:
            # Also test full-range random
            A_np2 = np.random.randint(-128, 128,
                size=(o, n, env.BATCH, env.BLOCK_IN)).astype(np.int8)
            B_np2 = np.random.randint(-128, 128,
                size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(np.int8)

            A_nd2 = tvm.nd.array(A_np2, ctx)
            B_nd2 = tvm.nd.array(B_np2, ctx)
            C_nd2 = tvm.nd.array(
                np.zeros((o, m, env.BATCH, env.BLOCK_OUT), dtype=np.int8), ctx)

            f(A_nd2, B_nd2, C_nd2)
            C_out2 = C_nd2.numpy()
            C_ref2 = reference_gemm(A_np2, B_np2, o, n, m)
            passed2 = np.array_equal(C_ref2, C_out2)

            print(f"  full-range random: {'PASSED' if passed2 else 'FAILED'}")
            if not passed2:
                all_passed = False
                mismatches = np.sum(C_ref2 != C_out2)
                print(f"    {mismatches}/{C_ref2.size} mismatches")

    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'=' * 60}")

    return all_passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("VTA Transformer GEMM Verification")
    print(f"Transformer: d_model=64, nhead=4, d_head=16, seq=32, d_ff=128")
    print()

    if "--cpu-only" in sys.argv:
        cpu_dimension_check()
    else:
        cpu_dimension_check()
        print()
        ok = run_board_tests()
        sys.exit(0 if ok else 1)
