#!/usr/bin/env python3
"""Stage-by-stage diagnostic for the VTA INT4-o8 transformer board pipeline.

Runs sample 0 through every stage of benchmark_vta_transformer.py's infer()
inline, printing intermediate values, and bit-exactly comparing each stage to
the host-generated reference (debug_full_reference_sample0.npz). On FAIL,
substitutes the reference value into the next stage so downstream stages
aren't corrupted by upstream divergence — this isolates the FIRST point of
divergence between host sim and board.

Usage on the board:
    sudo LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH \\
         PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH \\
         python3 debug_full_pipeline.py \\
             --export-dir /home/xilinx/transformer_export \\
             --reference  /home/xilinx/debug_full_reference_sample0.npz \\
             --data       /home/xilinx/radioml2018_eval_snr_filtered.npz
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import subprocess
import sys

import numpy as np
import tvm
import tvm.runtime

# Reuse pure-numpy helpers from the production script (no DMA, safe to import).
from benchmark_vta_transformer import (
    BLOCK,
    SQRT_96,
    pack_int4_for_vta,
    tile_for_wgt,
    quant_signed,
    quant_unsigned,
    cpu_requant_after_vta,
    setup_board,
)


# ============================================================
# VTA call wrapper.
#
# Per-call protocol on this PL (verified empirically):
#   - Fresh tvm.nd.array A/W/D/C with .copy() on every numpy input
#   - ctx.sync() BEFORE and AFTER mod() to keep the queue clean
#   - Detect all-zero output and retry up to 3× with fresh buffers
#     (intermittent VTA glitch); log each retry to stderr
#   - DO NOT del + gc.collect after the call — that forces premature
#     CMA release and destabilises the in-flight DMA state
# ============================================================
def vta_chunked(mod, A_packed, W_slices, D_per_call, ctx, o_t, mod_id=""):
    """Issue n_calls m=1 VTA calls, concat int8 outputs on the m axis.

    A_packed:    pre-packed int4 input, shape (o_t, n, 1, BLOCK).
    W_slices:    list of pre-packed (1, n, BLOCK, BLOCK) numpy int8 slices.
    D_per_call:  list of int32 (o_t, 1, 1, BLOCK) per-call bias broadcasts,
                 or None for zero bias.
    mod_id:      label used in retry log lines (cosmetic).
    Returns: int8 (o_t, n_calls * BLOCK).
    """
    n_calls = len(W_slices)
    full = np.empty((o_t, n_calls * BLOCK), dtype=np.int8)
    zero_D = np.zeros((o_t, 1, 1, BLOCK), dtype=np.int32)
    out_shape = (o_t, 1, 1, BLOCK)
    for m in range(n_calls):
        D_arr = D_per_call[m] if D_per_call is not None else zero_D
        out_slice = None
        for attempt in range(4):     # 1 initial + up to 3 retries
            A_nd = tvm.nd.array(A_packed.copy(),    ctx)
            W_nd = tvm.nd.array(W_slices[m].copy(), ctx)
            D_nd = tvm.nd.array(D_arr.copy(),       ctx)
            C_nd = tvm.nd.array(np.zeros(out_shape, dtype=np.int8), ctx)
            ctx.sync()
            mod(A_nd, W_nd, D_nd, C_nd)
            ctx.sync()
            out_slice = C_nd.numpy()[:, 0, 0, :].copy()
            if (out_slice != 0).any():
                break
            if attempt < 3:
                print(f"[retry] {mod_id or '?'} m_call={m}: zero output, "
                      f"retry {attempt+1}/3", file=sys.stderr)
        full[:, m * BLOCK:(m + 1) * BLOCK] = out_slice
    return full


def proj_static(mods, modules_meta, W_slices_packed, mod_id, gemm_name,
                in_int4, ctx):
    """Static-weight projection (q/k/v/o) with M-chunking when o_tile<M_full.

    in_int4: (M_full, K) int8.  Returns int8 (M_full, N_full).
    """
    info = modules_meta[mod_id]
    o_t  = info["o_tile"]
    n_t  = info["n"]
    M_full = in_int4.shape[0]
    n_m_chunks = M_full // o_t
    if n_m_chunks == 1:
        A_tiled = in_int4.reshape(o_t, n_t, 1, BLOCK).astype(np.int8)
        A_packed = pack_int4_for_vta(A_tiled)
        return vta_chunked(mods[mod_id], A_packed,
                            W_slices_packed[gemm_name], None, ctx, o_t,
                            mod_id=mod_id)
    chunks = []
    for m_chunk in range(n_m_chunks):
        chunk_in = in_int4[m_chunk*o_t:(m_chunk+1)*o_t]
        A_tiled  = chunk_in.reshape(o_t, n_t, 1, BLOCK).astype(np.int8)
        A_packed = pack_int4_for_vta(np.ascontiguousarray(A_tiled))
        chunks.append(vta_chunked(mods[mod_id], A_packed,
                                    W_slices_packed[gemm_name], None, ctx, o_t,
                                    mod_id=mod_id))
    return np.concatenate(chunks, axis=0)


def mlp_call(mods, modules_meta, W_slices_packed, bias_full, mod_id, gemm_name,
             in_int4, ctx):
    """4-arg MLP call (fc1 / fc2) with M-chunking when o_tile < M_full.

    in_int4 shape (M_full, K). With o_tile<M_full (fc1/fc2 use o_tile=8 to
    stay below VTA's hardware o×n threshold), we issue n_m_chunks =
    M_full / o_tile invocations of the m=1 chunked path and concat on M.
    """
    info = modules_meta[mod_id]
    o_t  = info["o_tile"]
    n_t  = info["n"]
    M_full = in_int4.shape[0]
    n_m_chunks = M_full // o_t

    bias_tiled = bias_full[gemm_name]                        # (m_full, BLOCK) int32
    n_calls = len(W_slices_packed[gemm_name])
    D_per_call = []
    for m in range(n_calls):
        bias_slice = bias_tiled[m:m + 1]                     # (1, BLOCK)
        bias_bc = np.ascontiguousarray(
            np.broadcast_to(bias_slice.reshape(1, 1, 1, BLOCK), (o_t, 1, 1, BLOCK)),
            dtype=np.int32)
        D_per_call.append(bias_bc)

    if n_m_chunks == 1:
        A_tiled = in_int4.reshape(o_t, n_t, 1, BLOCK).astype(np.int8)
        A_packed = pack_int4_for_vta(A_tiled)
        return vta_chunked(mods[mod_id], A_packed,
                            W_slices_packed[gemm_name], D_per_call, ctx, o_t,
                            mod_id=mod_id)
    chunks = []
    for m_chunk in range(n_m_chunks):
        chunk_in = in_int4[m_chunk*o_t:(m_chunk+1)*o_t]
        A_tiled  = chunk_in.reshape(o_t, n_t, 1, BLOCK).astype(np.int8)
        A_packed = pack_int4_for_vta(np.ascontiguousarray(A_tiled))
        chunks.append(vta_chunked(mods[mod_id], A_packed,
                                    W_slices_packed[gemm_name], D_per_call, ctx, o_t,
                                    mod_id=mod_id))
    return np.concatenate(chunks, axis=0)


def runtime_proj(mods, modules_meta, mod_id, A_unpacked, W_slices, ctx):
    """Runtime-weight call (qk / av) with M-chunking when o_tile<M_full.

    A_unpacked: (M_full, K) int8 activation. W_slices: pre-packed list.
    Caller no longer pre-packs A — this helper tiles + packs per chunk.
    """
    info = modules_meta[mod_id]
    o_t  = info["o_tile"]
    n_t  = info["n"]
    M_full = A_unpacked.shape[0]
    n_m_chunks = M_full // o_t
    if n_m_chunks == 1:
        A_tiled = A_unpacked.reshape(o_t, n_t, 1, BLOCK).astype(np.int8)
        A_packed = pack_int4_for_vta(A_tiled)
        return vta_chunked(mods[mod_id], A_packed, W_slices, None, ctx, o_t,
                            mod_id=mod_id)
    chunks = []
    for m_chunk in range(n_m_chunks):
        chunk_in = A_unpacked[m_chunk*o_t:(m_chunk+1)*o_t]
        A_tiled  = chunk_in.reshape(o_t, n_t, 1, BLOCK).astype(np.int8)
        A_packed = pack_int4_for_vta(np.ascontiguousarray(A_tiled))
        chunks.append(vta_chunked(mods[mod_id], A_packed, W_slices, None, ctx, o_t,
                                    mod_id=mod_id))
    return np.concatenate(chunks, axis=0)


# ============================================================
# Per-stage comparison
# ============================================================
def stage(name, computed, ref_npz, fail_log):
    """Compare `computed` to ref[name]. Prints PASS/FAIL detail.
    On FAIL appends to fail_log and returns the REFERENCE value so downstream
    stages start from a known-good input. On PASS returns the computed value.
    """
    expect = np.asarray(ref_npz[name])
    got    = np.asarray(computed)
    rng_got = (int(got.min()), int(got.max())) if got.size else (0, 0)
    head = got.flatten()[:8].tolist() if got.size else []
    if got.shape != expect.shape:
        print(f"  [{name:18s}] FAIL  shape mismatch  got={got.shape} expect={expect.shape}")
        fail_log.append((name, "shape mismatch", got.shape, expect.shape))
        return expect
    diff = got.astype(np.int64) - expect.astype(np.int64) if np.issubdtype(got.dtype, np.integer) \
           else got.astype(np.float64) - expect.astype(np.float64)
    max_abs = float(np.abs(diff).max()) if diff.size else 0.0
    miscount = int(np.count_nonzero(diff))
    if max_abs == 0:
        print(f"  [{name:18s}] PASS  shape={got.shape}  range={rng_got}  first8={head}")
        return got
    print(f"  [{name:18s}] FAIL  shape={got.shape}  range={rng_got}")
    print(f"        first8 got    = {head}")
    print(f"        first8 expect = {expect.flatten()[:8].tolist()}")
    print(f"        max|diff|={max_abs}  mismatch={miscount}/{got.size}")
    fail_log.append((name, "diff", max_abs, miscount))
    return expect


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True,
                    help="transformer_export/ on the board")
    ap.add_argument("--reference",  required=True,
                    help="debug_full_reference_sample0.npz")
    ap.add_argument("--data", default="/home/xilinx/radioml2018_eval_snr_filtered.npz")
    args = ap.parse_args()

    print("=" * 72)
    print("VTA Transformer Full-Pipeline Diagnostic — Sample 0")
    print("=" * 72)
    print(f"  export-dir : {args.export_dir}")
    print(f"  reference  : {args.reference}")
    print(f"  data       : {args.data}")

    setup_board()
    ctx = tvm.device("ext_dev", 0)

    # ----- Reference -----
    ref = np.load(args.reference)
    print(f"\n[ref] {len(ref.files)} keys loaded")

    # ----- Config / scales / shifts -----
    with open(os.path.join(args.export_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.export_dir, "scales.json")) as f:
        S = json.load(f)
    TUNED = cfg["tuned_shifts"]
    modules_meta = cfg["modules"]
    print(f"[shifts] tuned = {TUNED}")

    # ----- Compiled modules (.so) -----
    mods = {}
    for mid in modules_meta:
        o_path = os.path.join(args.export_dir, modules_meta[mid]["file"])
        so_path = o_path.replace(".o", ".so")
        if not os.path.exists(so_path):
            print(f"  link {os.path.basename(o_path)} -> .so")
            subprocess.check_call([
                "gcc", "-shared", "-o", so_path, o_path,
                "-L/home/xilinx/tvm-src/build", "-ltvm_runtime"])
        mods[mid] = tvm.runtime.load_module(so_path)
        print(f"[load] {mid} <- {os.path.basename(so_path)}")

    # ----- Static weights (pre-packed m=1 slices) and biases -----
    W_slices_packed = {}
    for gemm in ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]:
        W_full = np.load(os.path.join(args.export_dir,
                                       f"weights/{gemm}_W_tiled.npy")).astype(np.int8)
        W_slices_packed[gemm] = [
            pack_int4_for_vta(np.ascontiguousarray(W_full[m:m + 1]))
            for m in range(W_full.shape[0])
        ]
    bias_full = {
        "fc1": np.load(os.path.join(args.export_dir, "weights/fc1_bias_int32.npy"))
                 .astype(np.int32),
        "fc2": np.load(os.path.join(args.export_dir, "weights/fc2_bias_int32_corrected.npy"))
                 .astype(np.int32),
    }

    # ----- CPU params -----
    cpu = lambda name: np.load(os.path.join(args.export_dir, "cpu_params", name))
    W_emb        = cpu("emb_conv_W_int.npy").astype(np.int8)
    b_emb        = cpu("emb_conv_bias.npy").astype(np.float32)
    bn_emb_mean  = cpu("bn_emb_mean.npy").astype(np.float32)
    bn_emb_var   = cpu("bn_emb_var.npy").astype(np.float32)
    pos_enc      = cpu("pos_enc.npy").astype(np.float32)
    bn_attn_mean = cpu("bn_attn_mean.npy").astype(np.float32)
    bn_attn_var  = cpu("bn_attn_var.npy").astype(np.float32)
    bn_mlp_mean  = cpu("bn_mlp_mean.npy").astype(np.float32)
    bn_mlp_var   = cpu("bn_mlp_var.npy").astype(np.float32)
    W_cls        = cpu("cls_W_int.npy").astype(np.int8)
    b_cls        = cpu("cls_bias.npy").astype(np.float32)

    W_emb_flat = W_emb.reshape(96, 32).astype(np.int32)
    bias_emb_int32 = np.round(b_emb / (S["emb_w"] * S["emb_in"])).astype(np.int32)
    pos2d = pos_enc[0]                                                    # (64, 96)

    # ----- Sample 0 -----
    d = np.load(args.data)
    sig = d["signals"][0]                                                 # (1, 1024, 2)
    label = int(d["labels"][0])
    print(f"\n[sample] index=0  label={label}")

    fail_log = []

    # ============ 1. Patch embedding ============
    print("\n--- Stage 1: patch embedding (CPU INT8) ---")
    x = sig.transpose(2, 0, 1).astype(np.float32)
    x_int8 = quant_signed(x, S["emb_in"], -128, 127).astype(np.int8)
    x_patches = x_int8.reshape(2, 1, 64, 16).transpose(2, 0, 1, 3).reshape(64, 32)
    acc = x_patches.astype(np.int32) @ W_emb_flat.T + bias_emb_int32[None, :]
    emb_float = acc.astype(np.float64) * (S["emb_w"] * S["emb_in"])
    emb_bn = (emb_float - bn_emb_mean.astype(np.float64)[None, :]) / \
             np.sqrt(bn_emb_var.astype(np.float64)[None, :] + 1e-5)
    emb_relu = np.maximum(emb_bn, 0.0)
    emb_q = quant_unsigned(emb_relu, S["emb_out"], 0, 255)
    emb_q = stage("emb_q", emb_q, ref, fail_log)

    # ============ 2. Positional encoding ============
    print("\n--- Stage 2: positional encoding ---")
    emb_for_pos = emb_q.astype(np.float64) * S["emb_out"]
    emb_q2 = quant_signed(emb_for_pos, S["pos_in"], -128, 127)
    pos_q  = quant_signed(pos2d, S["pos_in"], -128, 127)
    sum_float = (emb_q2.astype(np.float64) + pos_q.astype(np.float64)) * S["pos_in"]
    after_pos = quant_signed(sum_float, S["pos_out"], -128, 127).astype(np.int8)
    after_pos = stage("after_pos", after_pos, ref, fail_log)

    # ============ 3. Pre-attn BN + INT4 quant ============
    print("\n--- Stage 3: BN_attn + INT4 quant ---")
    x_pos_float = after_pos.astype(np.float64) * S["pos_out"]
    bn_attn = (x_pos_float - bn_attn_mean.astype(np.float64)[None, :]) / \
              np.sqrt(bn_attn_var.astype(np.float64)[None, :] + 1e-5)
    pre_int4 = quant_signed(bn_attn, S["attn_pre_out"], -8, 7).astype(np.int8)
    pre_int4 = stage("pre_int4", pre_int4, ref, fail_log)

    # ============ 4. Q / K / V projections (3 VTA calls each) ============
    print("\n--- Stage 4: Q / K / V VTA ---")
    Q_int8 = proj_static(mods, modules_meta, W_slices_packed,
                          "proj_k96_s3_m1", "q_proj", pre_int4, ctx)
    Q_int8 = stage("Q_int8", Q_int8, ref, fail_log)
    K_int8 = proj_static(mods, modules_meta, W_slices_packed,
                          "proj_k96_s3_m1", "k_proj", pre_int4, ctx)
    K_int8 = stage("K_int8", K_int8, ref, fail_log)
    V_int8 = proj_static(mods, modules_meta, W_slices_packed,
                          "proj_k96_s2_m1", "v_proj", pre_int4, ctx)
    V_int8 = stage("V_int8", V_int8, ref, fail_log)

    # ============ 5. CPU requant Q/K/V to INT4 + head split ============
    print("\n--- Stage 5: CPU requant Q/K/V to INT4 + head split ---")
    Q_int4 = cpu_requant_after_vta(Q_int8, TUNED["q"],
                                    S["attn_pre_out"], S["q_w"], S["q_out"], -8, 7)
    Q_int4 = stage("Q_int4", Q_int4, ref, fail_log)
    K_int4 = cpu_requant_after_vta(K_int8, TUNED["k"],
                                    S["attn_pre_out"], S["k_w"], S["k_out"], -8, 7)
    K_int4 = stage("K_int4", K_int4, ref, fail_log)
    V_int4 = cpu_requant_after_vta(V_int8, TUNED["v"],
                                    S["attn_pre_out"], S["v_w"], S["v_out"], -8, 7)
    V_int4 = stage("V_int4", V_int4, ref, fail_log)

    Qh = Q_int4.reshape(64, 3, 32).transpose(1, 0, 2)
    Kh = K_int4.reshape(64, 3, 32).transpose(1, 0, 2)
    Vh = V_int4.reshape(64, 3, 32).transpose(1, 0, 2)
    Qh = stage("Qh", Qh, ref, fail_log)
    Kh = stage("Kh", Kh, ref, fail_log)
    Vh = stage("Vh", Vh, ref, fail_log)

    # ============ 6. Per-head attention ============
    print("\n--- Stage 6: per-head attention (VTA Q@K^T, softmax, VTA attn@V) ---")
    qkt_int8   = np.empty((3, 64, 64), dtype=np.int8)
    presoftmax = np.empty((3, 64, 64), dtype=np.int8)
    attn_int4  = np.empty((3, 64, 64), dtype=np.int8)
    av_int8    = np.empty((3, 64, 32), dtype=np.int8)
    av_int4    = np.empty((3, 64, 32), dtype=np.int8)
    for h in range(3):
        # Q@K^T  (qkt_s3_m1): A = Q_h (64, 32), B = K_h treated as wgt (64, 32).
        # runtime_proj handles M-chunking on Q_h when qkt o_tile < 64.
        K_h = Kh[h]
        K_tiled  = tile_for_wgt(K_h, m_tiles=4, n_tiles=2)
        K_slices = [pack_int4_for_vta(np.ascontiguousarray(K_tiled[m:m + 1].astype(np.int8)))
                    for m in range(4)]
        scores = runtime_proj(mods, modules_meta, "qkt_s3_m1",
                               Qh[h], K_slices, ctx)
        qkt_int8[h] = scores

        ps = cpu_requant_after_vta(scores, TUNED["qk"],
                                    S["q_out"], S["k_out"], S["softmax_in"],
                                    -8, 7, extra_factor=SQRT_96)
        presoftmax[h] = ps

        scores_f = ps.astype(np.float64) * S["softmax_in"]
        scores_f = scores_f - scores_f.max(axis=-1, keepdims=True)
        exps = np.exp(scores_f)
        attn_f = exps / exps.sum(axis=-1, keepdims=True)
        attn_h = quant_signed(attn_f, S["softmax_out"], -8, 7).astype(np.int8)
        attn_int4[h] = attn_h

        # attn@V (av_s3_m1): A = attn_h (64, 64), B = V_h.T (32, 64) as wgt.
        # runtime_proj handles M-chunking on attn_h when av o_tile < 64.
        V_T = Vh[h].T
        V_tiled  = tile_for_wgt(V_T, m_tiles=2, n_tiles=4)
        V_slices = [pack_int4_for_vta(np.ascontiguousarray(V_tiled[m:m + 1].astype(np.int8)))
                    for m in range(2)]
        ctx_int8_h = runtime_proj(mods, modules_meta, "av_s3_m1",
                                   attn_h, V_slices, ctx)
        av_int8[h] = ctx_int8_h

        ctx_int4 = cpu_requant_after_vta(ctx_int8_h, TUNED["av"],
                                          S["softmax_out"], S["v_out"], S["o_in"],
                                          -8, 7)
        av_int4[h] = ctx_int4

    qkt_int8   = stage("qkt_int8",   qkt_int8,   ref, fail_log)
    presoftmax = stage("presoftmax", presoftmax, ref, fail_log)
    attn_int4  = stage("attn_int4",  attn_int4,  ref, fail_log)
    av_int8    = stage("av_int8",    av_int8,    ref, fail_log)
    av_int4    = stage("av_int4",    av_int4,    ref, fail_log)

    # ============ 7. Concat heads, O projection ============
    print("\n--- Stage 7: concat heads, O proj VTA ---")
    ctx_cat = av_int4.transpose(1, 0, 2).reshape(64, 96)
    ctx_cat = stage("ctx_cat", ctx_cat, ref, fail_log)

    o_int8 = proj_static(mods, modules_meta, W_slices_packed,
                          "proj_k96_s3_m1", "o_proj", ctx_cat, ctx)
    o_int8 = stage("o_int8", o_int8, ref, fail_log)
    o_int4 = cpu_requant_after_vta(o_int8, TUNED["o"],
                                    S["o_in"], S["o_w"], S["attn_residual"], -8, 7)
    o_int4 = stage("o_int4", o_int4, ref, fail_log)

    # ============ 8. Attention residual add ============
    print("\n--- Stage 8: attention residual ---")
    skip_float = after_pos.astype(np.float64) * S["pos_out"]
    skip_int4_attn = quant_signed(skip_float, S["attn_residual"], -8, 7).astype(np.int32)
    attn_block_out = o_int4.astype(np.int32) + skip_int4_attn
    attn_block_out = stage("attn_block_out", attn_block_out, ref, fail_log)

    # ============ 9. BN_mlp + INT4 quant ============
    print("\n--- Stage 9: BN_mlp + INT4 quant ---")
    x_mlp_in_float = attn_block_out.astype(np.float64) * S["attn_residual"]
    bn_mlp = (x_mlp_in_float - bn_mlp_mean.astype(np.float64)[None, :]) / \
             np.sqrt(bn_mlp_var.astype(np.float64)[None, :] + 1e-5)
    mlp_pre = quant_signed(bn_mlp, S["mlp_bn_out"], -8, 7).astype(np.int8)
    mlp_pre = stage("mlp_pre", mlp_pre, ref, fail_log)

    # ============ 10. fc1 (4-arg VTA) ============
    print("\n--- Stage 10: fc1 VTA ---")
    fc1_out8 = mlp_call(mods, modules_meta, W_slices_packed, bias_full,
                        "fc1_s3_m1", "fc1", mlp_pre, ctx)
    fc1_out8 = stage("fc1_out8", fc1_out8, ref, fail_log)

    # ReLU + unsigned [0,15] requant + zero-point shift to signed [-8,7]
    cs_fc1 = TUNED["fc1"]
    recovered = fc1_out8.astype(np.float64) * (2.0 ** cs_fc1)
    real = recovered * (S["mlp_bn_out"] * S["fc1_w"])
    fc1_unsigned = np.clip(np.round(real / S["fc1_out"]), 0, 15).astype(np.int8)
    fc1_unsigned = stage("fc1_unsigned", fc1_unsigned, ref, fail_log)
    fc1_signed = (fc1_unsigned.astype(np.int32) - 8).astype(np.int8)
    fc1_signed = stage("fc1_signed", fc1_signed, ref, fail_log)

    # ============ 11. fc2 (4-arg VTA, corrected bias) ============
    print("\n--- Stage 11: fc2 VTA ---")
    fc2_out8 = mlp_call(mods, modules_meta, W_slices_packed, bias_full,
                        "fc2_s4_m1", "fc2", fc1_signed, ctx)
    fc2_out8 = stage("fc2_out8", fc2_out8, ref, fail_log)
    fc2_int4 = cpu_requant_after_vta(fc2_out8, TUNED["fc2"],
                                      S["fc1_out"], S["fc2_w"], S["mlp_residual"],
                                      -8, 7)
    fc2_int4 = stage("fc2_int4", fc2_int4, ref, fail_log)

    # ============ 12. MLP residual add ============
    print("\n--- Stage 12: MLP residual ---")
    skip_float_mlp = attn_block_out.astype(np.float64) * S["attn_residual"]
    skip_int_mlp = quant_signed(skip_float_mlp, S["mlp_residual"], -8, 7).astype(np.int32)
    mlp_block_out = fc2_int4.astype(np.int32) + skip_int_mlp
    mlp_block_out = stage("mlp_block_out", mlp_block_out, ref, fail_log)

    # ============ 13. Classifier ============
    print("\n--- Stage 13: classifier ---")
    mlp_float = mlp_block_out.astype(np.float64) * S["mlp_residual"]
    gap = mlp_float.mean(axis=0)
    W_cls_float = W_cls.astype(np.float64) * S["cls_w"]
    logits = gap @ W_cls_float.T + b_cls.astype(np.float64)
    pred = int(np.argmax(logits))
    print(f"  logits range  : [{float(logits.min()):.3f}, {float(logits.max()):.3f}]")
    print(f"  ref logits    : [{float(ref['logits'].min()):.3f}, {float(ref['logits'].max()):.3f}]")
    print(f"  pred (board)  : {pred}")
    print(f"  pred (ref)    : {int(ref['pred'])}")
    print(f"  label         : {label}")

    # ============ Summary ============
    print("\n" + "=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)
    if not fail_log:
        print("  ALL STAGES PASS — board pipeline is bit-exact with host reference.")
        print(f"  pred={pred}  ref={int(ref['pred'])}  label={label}")
    else:
        print(f"  {len(fail_log)} stage(s) FAILED. First divergence:")
        first = fail_log[0]
        print(f"    >>> {first[0]} <<<   detail={first[1:]}")
        print()
        print("  Full FAIL list:")
        for entry in fail_log:
            print(f"    - {entry[0]}: {entry[1:]}")


if __name__ == "__main__":
    main()
