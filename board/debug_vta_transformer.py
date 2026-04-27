#!/usr/bin/env python3
"""Board-side staged diagnostic for the VTA INT4-o8 transformer.

Validates each piece of the inference pipeline on sample 0 against a host-
generated reference (debug_reference_sample0.npz). Prints PASS/FAIL at four
stages, narrowing down where on-board behavior diverges from the numpy sim.

Stage 1: CPU pre-VTA (patch embed, positional add, BN_attn + INT4 quant)
         Compares emb_q, after_pos, pre_norm bit-for-bit against the reference.
Stage 2: Full VTA Q-projection (proj_k96_s3_m1, 6 m=1 chunked calls)
         Compares the concatenated int8 DMA output against
         (pre_norm @ W_q.T) >> 3 clipped.
Stage 3: Numpy reference of the SAME GEMM (using the un-tiled weight)
         Sanity check that our notion of "what VTA should compute" matches.
Stage 4: Numpy vs board comparison (cross-check)

If Stage 1 fails       -> CPU pre-VTA pipeline (numpy) diverges from sim.
If Stage 1 passes,
   Stage 2 fails       -> bug is in VTA call (packing, tiling, buffer alloc, untiling).
If 1 & 2 pass          -> VTA Q is bit-exact; bug is downstream (K/V/attn/MLP).

The script imports its packing / tiling / quant helpers FROM
benchmark_vta_transformer.py so the test exercises exactly what the production
script does, not a parallel implementation.

Usage on the board:
    sudo LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH \\
         PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH \\
         python3 debug_vta_transformer.py \\
             --export-dir /home/xilinx/transformer_export \\
             --reference  /home/xilinx/transformer_export/debug_reference_sample0.npz
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import tvm
import tvm.runtime

# Pull the EXACT helpers from the production script.
# Assumes benchmark_vta_transformer.py is in the same directory or /home/xilinx.
HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, "/home/xilinx"):
    if os.path.isfile(os.path.join(cand, "benchmark_vta_transformer.py")):
        sys.path.insert(0, cand)
        break
from benchmark_vta_transformer import (  # noqa: E402
    pack_int4_for_vta, setup_board, tile_for_wgt,
    quant_signed, quant_unsigned, BLOCK,
)


def _eq_report(name, actual, expected):
    """Print PASS/FAIL line and return the boolean."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        print(f"  [{name:<26s}] FAIL  shape mismatch  actual={actual.shape} "
              f"expected={expected.shape}")
        return False
    diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    if diff.max() == 0:
        print(f"  [{name:<26s}] PASS  shape={actual.shape}  bit-exact")
        return True
    n_mismatch = int(np.sum(diff > 0))
    print(f"  [{name:<26s}] FAIL  max|diff|={int(diff.max())}  "
          f"mismatches={n_mismatch}/{actual.size}")
    print(f"     actual range   [{actual.min()}, {actual.max()}]")
    print(f"     expected range [{expected.min()}, {expected.max()}]")
    # Show a small slice for inspection
    print(f"     actual[0, :8]   = {actual.flatten()[:8].tolist()}")
    print(f"     expected[0, :8] = {expected.flatten()[:8].tolist()}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="/home/xilinx/transformer_export")
    ap.add_argument("--reference",  default="/home/xilinx/transformer_export/"
                                            "debug_reference_sample0.npz")
    args = ap.parse_args()

    print(f"[ref]    {args.reference}")
    ref = np.load(args.reference)
    print(f"[sample] label={int(ref['label'])}  "
          f"mode_e_pred={int(ref['mode_e_pred'])}")

    print(f"[init]   loading config / scales from {args.export_dir}")
    cfg = json.load(open(os.path.join(args.export_dir, "config.json")))
    S   = json.load(open(os.path.join(args.export_dir, "scales.json")))

    setup_board()
    import tvm  # noqa: E402
    import tvm.runtime  # noqa: E402
    ctx = tvm.device("ext_dev", 0)
    print(f"[ctx]    {ctx}")

    # Load CPU-side parameters
    cpu = lambda fn: np.load(os.path.join(args.export_dir, "cpu_params", fn))
    W_emb        = cpu("emb_conv_W_int.npy").astype(np.int8)
    b_emb        = cpu("emb_conv_bias.npy").astype(np.float32)
    bn_emb_mean  = cpu("bn_emb_mean.npy").astype(np.float32)
    bn_emb_var   = cpu("bn_emb_var.npy").astype(np.float32)
    pos_enc      = cpu("pos_enc.npy")[0].astype(np.float32)        # (64, 96)
    bn_attn_mean = cpu("bn_attn_mean.npy").astype(np.float32)
    bn_attn_var  = cpu("bn_attn_var.npy").astype(np.float32)
    W_emb_flat = W_emb.reshape(96, 32).astype(np.int32)
    bias_emb_int32 = np.round(b_emb / (S["emb_w"] * S["emb_in"])).astype(np.int32)

    signal = ref["signal"]                                          # (1, 1024, 2)

    # ============================================================
    # Stage 1 — CPU pre-VTA pipeline
    # ============================================================
    print("\n" + "=" * 66)
    print("STAGE 1  CPU pre-VTA  (patch embed -> positional -> BN_attn)")
    print("=" * 66)

    # Patch embed
    x = signal.transpose(2, 0, 1).astype(np.float32)
    x_int8 = quant_signed(x, S["emb_in"], -128, 127).astype(np.int8)
    x_patches = x_int8.reshape(2, 1, 64, 16).transpose(2, 0, 1, 3).reshape(64, 32)
    acc = x_patches.astype(np.int32) @ W_emb_flat.T + bias_emb_int32[None, :]
    emb_float = acc.astype(np.float64) * (S["emb_w"] * S["emb_in"])
    emb_bn = (emb_float - bn_emb_mean.astype(np.float64)[None, :]) / \
             np.sqrt(bn_emb_var.astype(np.float64)[None, :] + 1e-5)
    emb_relu = np.maximum(emb_bn, 0.0)
    emb_q = quant_unsigned(emb_relu, S["emb_out"], 0, 255)          # (64, 96), [0,255]
    s1a = _eq_report("1.1 patch embed", emb_q.astype(np.int16), ref["emb_q"])

    # Positional encoding
    emb_for_pos = emb_q.astype(np.float64) * S["emb_out"]
    emb_q2 = quant_signed(emb_for_pos, S["pos_in"], -128, 127)
    pos_q  = quant_signed(pos_enc,    S["pos_in"], -128, 127)
    sum_float = (emb_q2.astype(np.float64) + pos_q.astype(np.float64)) * S["pos_in"]
    after_pos = quant_signed(sum_float, S["pos_out"], -128, 127).astype(np.int8)
    s1b = _eq_report("1.2 positional add", after_pos, ref["after_pos"])

    # BN_attn + INT4 quant
    x_pos_float = after_pos.astype(np.float64) * S["pos_out"]
    bn_attn = (x_pos_float - bn_attn_mean.astype(np.float64)[None, :]) / \
              np.sqrt(bn_attn_var.astype(np.float64)[None, :] + 1e-5)
    pre_int4 = quant_signed(bn_attn, S["attn_pre_out"], -8, 7).astype(np.int8)
    s1c = _eq_report("1.3 BN_attn + INT4 quant", pre_int4, ref["pre_norm"])

    stage1_ok = s1a and s1b and s1c

    # ============================================================
    # Stage 2 — VTA Q-projection via 6 m=1 chunked calls
    # ============================================================
    print("\n" + "=" * 66)
    print("STAGE 2  VTA proj_k96_s3_m1 (Q projection)  6 m=1 chunked calls")
    print("=" * 66)

    # Use the production script's pre_norm if Stage 1 passed; otherwise still
    # exercise the VTA path against the reference's pre_norm so we can isolate.
    pre_for_vta = pre_int4 if stage1_ok else ref["pre_norm"].astype(np.int8)

    o_path  = os.path.join(args.export_dir, "modules/proj_k96_s3_m1.o")
    so_path = o_path.replace(".o", ".so")
    if not os.path.exists(so_path):
        print(f"  linking {os.path.basename(o_path)} -> .so")
        subprocess.check_call(["gcc", "-shared", "-o", so_path, o_path,
                               "-L/home/xilinx/tvm-src/build", "-ltvm_runtime"])
    mod = tvm.runtime.load_module(so_path)
    print(f"  module: {os.path.basename(so_path)}")

    # Q weight: disk shape (m_full=6, n=6, BLOCK, BLOCK). With m=1 codegen,
    # we issue 6 calls, each with weight slice (1, 6, BLOCK, BLOCK), and
    # concatenate int8 outputs on the m axis to form the full (64, 96) result.
    W_q_tiled = np.load(os.path.join(args.export_dir, "weights/q_proj_W_tiled.npy")).astype(np.int8)
    print(f"  W_q tiled shape={W_q_tiled.shape}  range=[{W_q_tiled.min()},{W_q_tiled.max()}]")

    A_tiled = pre_for_vta.reshape(64, 6, 1, BLOCK).astype(np.int8)
    print(f"  A tiled shape={A_tiled.shape}     range=[{A_tiled.min()},{A_tiled.max()}]")
    A_packed = pack_int4_for_vta(A_tiled)
    print(f"  A packed shape={A_packed.shape}   bytes={A_packed.nbytes}")
    # Pre-allocate W only (its data is fresh per call); A/D/C are allocated
    # fresh inside the m-loop to dodge a DMA address reuse race that returns
    # all-zero output on later calls of the same module.
    W_nd   = tvm.nd.array(np.zeros((1, 6, BLOCK, BLOCK), dtype=np.int8).copy(), ctx)
    zero_C = np.zeros((64, 1, 1, BLOCK), dtype=np.int8)
    zero_D = np.zeros((64, 1, 1, BLOCK), dtype=np.int32)

    n_calls = W_q_tiled.shape[0]   # 6
    Q_int8_board = np.empty((64, n_calls * BLOCK), dtype=np.int8)
    for m in range(n_calls):
        W_slice  = np.ascontiguousarray(W_q_tiled[m:m+1])         # (1, 6, BLOCK, BLOCK)
        W_packed = pack_int4_for_vta(W_slice)
        W_nd.copyfrom(W_packed.copy())
        A_nd = tvm.nd.array(A_packed.copy(), ctx)
        D_nd = tvm.nd.array(zero_D.copy(),   ctx)
        C_nd = tvm.nd.array(zero_C.copy(),   ctx)
        print(f"  invoke proj_k96_s3_m1(A, W[{m}:{m+1}], D=0, C)  m_call={m}/{n_calls-1}")
        mod(A_nd, W_nd, D_nd, C_nd)
        # Drain VTA command queue between calls.
        ctx.sync()
        C_raw = C_nd.numpy()
        nz = int(np.count_nonzero(C_raw))
        print(f"    C raw  shape={C_raw.shape}  min={int(C_raw.min())} max={int(C_raw.max())}  "
              f"nonzero={nz}/{C_raw.size}")
        Q_int8_board[:, m*BLOCK:(m+1)*BLOCK] = C_raw[:, 0, 0, :]

    print(f"  Q_int8_board concatenated shape={Q_int8_board.shape}  "
          f"min={int(Q_int8_board.min())} max={int(Q_int8_board.max())}  "
          f"nonzero={int(np.count_nonzero(Q_int8_board))}/{Q_int8_board.size}")
    s2 = _eq_report("2.1 VTA Q int8 output",
                     Q_int8_board, ref["Q_int8_expected"])

    # ============================================================
    # Stage 3 — Numpy reference for the same GEMM (using untiled weight)
    # ============================================================
    print("\n" + "=" * 66)
    print("STAGE 3  Numpy reference Q proj  (untile -> int32 GEMM -> >>3 -> clip)")
    print("=" * 66)

    # Untile the saved tiled weight back to the (96, 96) PyTorch [out, in] layout
    m_t, n_t, b_o, b_i = W_q_tiled.shape                   # (6, 6, 16, 16)
    W_q_untiled = W_q_tiled.transpose(0, 2, 1, 3).reshape(m_t * b_o, n_t * b_i)
    print(f"  W_q untiled shape={W_q_untiled.shape}")

    shift_q = int(ref["q_shift"])
    acc_np = pre_for_vta.astype(np.int32) @ W_q_untiled.astype(np.int32).T
    Q_int8_numpy = np.clip(np.right_shift(acc_np, shift_q), -128, 127).astype(np.int8)
    s3 = _eq_report("3.1 numpy GEMM vs ref",
                     Q_int8_numpy, ref["Q_int8_expected"])

    # ============================================================
    # Stage 4 — Numpy GEMM vs board GEMM cross-check
    # ============================================================
    print("\n" + "=" * 66)
    print("STAGE 4  Cross-check  (board VTA  vs  numpy reference)")
    print("=" * 66)
    s4 = _eq_report("4.1 board vs numpy", Q_int8_board, Q_int8_numpy)

    # ============================================================
    # Diagnosis
    # ============================================================
    print("\n" + "=" * 66)
    print("DIAGNOSIS")
    print("=" * 66)
    c_all_zero = (Q_int8_board == 0).all()
    if not stage1_ok:
        print("  CPU pre-VTA pipeline diverges from the sim reference.")
        print("  Root cause is in the script's CPU code (BN, quant, im2col, etc.),")
        print("  NOT in VTA. Compare numbers above to find where Stage 1 first fails.")
    elif c_all_zero:
        print("  VTA output is *literally zero* for every element. The module ran but")
        print("  produced no values. Most likely causes (in order):")
        print("    1. WRONG BITSTREAM. The currently-loaded .bit may not be the")
        print("       INT4-o8 build (1x16_i4w4o8a32). Check setup_board()'s [vta env]")
        print("       line above — INP and WGT must be 4, OUT must be 8. If those")
        print("       are correct and you still see all-zeros, the live PL bitstream")
        print("       may differ from what vta_config.json says. The fix is loading")
        print("       /home/xilinx/.vta_cache/ultra96/0_0_2/1x16_i4w4o8a32_15_14_17_17.bit.")
        print("    2. PYNQ overlay was loaded but PL_state file is stale; libvta is")
        print("       talking to a different bitstream than PYNQ thinks it loaded.")
        print("       Re-run after `rm /home/xilinx/pynq/pl_server/global_pl_state_.json`.")
        print("    3. VTA HW error (rare). Check `dmesg | tail` for xlnk errors.")
    elif not s2 and s3:
        print("  CPU pre-VTA matches and the numpy GEMM matches the reference,")
        print("  but the VTA module's int8 DMA output is wrong (non-zero garbage).")
        print("  Probable culprits, in order of likelihood:")
        print("    (a) int4 nibble packing of A or W (compare pack_int4_for_vta)")
        print("    (b) tiled weight layout produced by the export script")
        print("    (c) compiled module signature / argument order")
        print("    (d) DMA output reading / untiling order")
    elif not s2 and not s3:
        print("  Numpy reference itself diverges from the saved expected output.")
        print("  Re-run generate_debug_reference.py on the host with matching shifts.")
    elif s2 and not s3:
        print("  Board VTA matches reference, but our numpy reference is wrong.")
        print("  Check that ref was generated with the same q_shift and untiling.")
    else:
        print("  CPU pre-VTA matches AND VTA Q matches AND numpy ref matches AND")
        print("  board==numpy. Q-projection path is bit-exact.")
        print("  The 3.88% accuracy bug is downstream of Q proj — instrument K, V,")
        print("  Q@K^T, softmax, attn@V, O proj, residuals, fc1/fc2 in turn.")

    # Exit nonzero if any stage failed (useful for scripted runs)
    sys.exit(0 if (stage1_ok and s2 and s3 and s4) else 1)


if __name__ == "__main__":
    main()
