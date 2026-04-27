#!/usr/bin/env python3
"""Generate stage-by-stage reference values for the VTA INT4-o8 transformer.

Mirrors finn-vs-vitisai/board/benchmark_vta_transformer.py's infer() pipeline
on sample 0 of the SNR-filtered eval set, using numpy GEMMs in place of VTA
calls. The numpy GEMM (right-shift, clip to int8 [-128, 127], cast int8) is
bit-exact with the recompiled VTA modules — verified by export_vta_transformer.py
sim. The saved NPZ is therefore the value the board SHOULD produce at each stage.

Output: finn-vs-vitisai/vta/debug_full_reference_sample0.npz

Run on host:
    python finn-vs-vitisai/vta/generate_full_reference.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent              # finn-vs-vitisai/vta
sys.path.insert(0, str(HERE))
from vta_transformer_sim import TransformerSim, SQRT_96   # weights + scales loader

NPZ_SCALES  = HERE / "transformer_scales.npz"
DATA        = HERE.parent / "data" / "radioml2018_eval_snr_filtered.npz"
COARSE_PATH = HERE / "phase3_coarse_shifts.json"
OUT_NPZ     = HERE / "debug_full_reference_sample0.npz"


# ---------- CPU ops (must match benchmark_vta_transformer.py byte-for-byte) ----
def quant_signed(x_float, scale, lo, hi):
    return np.clip(np.round(x_float / scale), lo, hi).astype(np.int32)


def quant_unsigned(x_float, scale, lo, hi):
    return np.clip(np.round(x_float / scale), lo, hi).astype(np.int32)


def cpu_requant_after_vta(x_int8, coarse_shift, in_scale, w_scale, out_scale,
                          clip_lo, clip_hi, extra_factor=1.0):
    recovered = x_int8.astype(np.float64) * (2.0 ** coarse_shift)
    real = recovered * (w_scale * in_scale * extra_factor)
    return np.clip(np.round(real / out_scale), clip_lo, clip_hi).astype(np.int8)


# ---------- numpy bit-exact equivalents of the recompiled VTA modules ----------
def vta_3arg(A_int4, W_int4, shift):
    """3-arg VTA: GEMM(int4 x int4 -> int32) -> SHR -> CLIP[-128,127] -> int8.
    A: (M, K), W: (N, K). Returns int8 (M, N).
    """
    acc = A_int4.astype(np.int32) @ W_int4.astype(np.int32).T
    out = np.right_shift(acc, shift)
    return np.clip(out, -128, 127).astype(np.int8)


def vta_4arg(A_int4, W_int4, bias_int32, shift):
    """4-arg VTA: GEMM + per-output bias -> SHR -> CLIP[-128,127] -> int8."""
    acc = A_int4.astype(np.int32) @ W_int4.astype(np.int32).T + bias_int32[None, :]
    out = np.right_shift(acc, shift)
    return np.clip(out, -128, 127).astype(np.int8)


def main():
    print(f"[load] sim from {NPZ_SCALES.name}")
    sim = TransformerSim(str(NPZ_SCALES))
    S = sim.S

    # Phase-3 tuned coarse shifts (baseline minus 1 on v, o, fc2). These also
    # match the modules' baked-in shifts, so the numpy GEMM uses these values
    # directly as the right-shift amount before int8 clip.
    with open(COARSE_PATH) as f:
        TUNED = json.load(f)
    TUNED["v"]   = 2
    TUNED["o"]   = 3
    TUNED["fc2"] = 4
    print(f"[shifts] tuned = {TUNED}")

    print(f"[data] sample 0 from {DATA.name}")
    d = np.load(DATA)
    sig = d["signals"][0]                             # (1, 1024, 2)
    label = int(d["labels"][0])

    out: dict[str, np.ndarray] = {}

    # ============ 1. Patch embedding (CPU INT8) ============
    x = sig.transpose(2, 0, 1).astype(np.float32)     # (2, 1, 1024)
    x_int8 = quant_signed(x, S["emb_in"], -128, 127).astype(np.int8)
    x_patches = x_int8.reshape(2, 1, 64, 16).transpose(2, 0, 1, 3).reshape(64, 32)
    W_emb_flat = sim.W_emb.reshape(96, 32).astype(np.int32)
    acc = x_patches.astype(np.int32) @ W_emb_flat.T + sim.bias_emb_int32[None, :]
    emb_float = acc.astype(np.float64) * (S["emb_w"] * S["emb_in"])
    emb_bn = (emb_float - sim.bn_emb_mean.astype(np.float64)[None, :]) / \
             np.sqrt(sim.bn_emb_var.astype(np.float64)[None, :] + 1e-5)
    emb_relu = np.maximum(emb_bn, 0.0)
    emb_q = quant_unsigned(emb_relu, S["emb_out"], 0, 255)         # (64, 96), [0, 255]
    out["emb_q"] = emb_q.astype(np.int32)

    # ============ 2. Positional encoding (CPU INT8) ============
    emb_for_pos = emb_q.astype(np.float64) * S["emb_out"]
    emb_q2 = quant_signed(emb_for_pos, S["pos_in"], -128, 127)
    pos2d = sim.pos[0]                                              # (64, 96)
    pos_q = quant_signed(pos2d, S["pos_in"], -128, 127)
    sum_float = (emb_q2.astype(np.float64) + pos_q.astype(np.float64)) * S["pos_in"]
    after_pos = quant_signed(sum_float, S["pos_out"], -128, 127).astype(np.int8)
    out["after_pos"] = after_pos

    # ============ 3. Pre-attn BN + INT4 quant ============
    x_pos_float = after_pos.astype(np.float64) * S["pos_out"]
    bn_attn = (x_pos_float - sim.bn_attn_mean.astype(np.float64)[None, :]) / \
              np.sqrt(sim.bn_attn_var.astype(np.float64)[None, :] + 1e-5)
    pre_int4 = quant_signed(bn_attn, S["attn_pre_out"], -8, 7).astype(np.int8)
    out["pre_int4"] = pre_int4

    # ============ 4. Q / K / V projection (VTA-equivalent numpy) ============
    # Module baked shifts: proj_k96_s3=3 (q,k,o), proj_k96_s2=2 (v).
    Q_int8 = vta_3arg(pre_int4, sim.W_q, 3)
    K_int8 = vta_3arg(pre_int4, sim.W_k, 3)
    V_int8 = vta_3arg(pre_int4, sim.W_v, 2)
    out.update(Q_int8=Q_int8, K_int8=K_int8, V_int8=V_int8)

    # ============ 5. CPU requant Q/K/V to INT4 + head split ============
    Q_int4 = cpu_requant_after_vta(Q_int8, TUNED["q"],
                                    S["attn_pre_out"], S["q_w"], S["q_out"], -8, 7)
    K_int4 = cpu_requant_after_vta(K_int8, TUNED["k"],
                                    S["attn_pre_out"], S["k_w"], S["k_out"], -8, 7)
    V_int4 = cpu_requant_after_vta(V_int8, TUNED["v"],
                                    S["attn_pre_out"], S["v_w"], S["v_out"], -8, 7)
    out.update(Q_int4=Q_int4, K_int4=K_int4, V_int4=V_int4)

    Qh = Q_int4.reshape(64, 3, 32).transpose(1, 0, 2)   # (3, 64, 32)
    Kh = K_int4.reshape(64, 3, 32).transpose(1, 0, 2)
    Vh = V_int4.reshape(64, 3, 32).transpose(1, 0, 2)
    out.update(Qh=Qh, Kh=Kh, Vh=Vh)

    # ============ 6. Per-head attention ============
    qkt_int8   = np.empty((3, 64, 64), dtype=np.int8)
    presoftmax = np.empty((3, 64, 64), dtype=np.int8)
    attn_int4  = np.empty((3, 64, 64), dtype=np.int8)
    av_int8    = np.empty((3, 64, 32), dtype=np.int8)
    av_int4    = np.empty((3, 64, 32), dtype=np.int8)
    for h in range(3):
        # Q@K^T: VTA computes A @ B.T where A=Q_h (64, 32), B=K_h (64, 32).
        # K_h is already (N=64, K=32) so no transpose; result is (64, 64).
        scores = vta_3arg(Qh[h], Kh[h], 3)
        qkt_int8[h] = scores

        # CPU requant with attn_scale 1/sqrt(96) folded in
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

        # attn @ V: benchmark transposes V_h to V_T (32, 64), passes as B operand.
        # VTA result C[i,j] = sum_k attn_h[i,k] * V_T[j,k] = (attn_h @ V_h)[i,j].
        V_T = Vh[h].T                                           # (32, 64)
        ctx = vta_3arg(attn_h, V_T, 3)                          # (64, 32)
        av_int8[h] = ctx

        ctx_int4 = cpu_requant_after_vta(ctx, TUNED["av"],
                                          S["softmax_out"], S["v_out"], S["o_in"],
                                          -8, 7)
        av_int4[h] = ctx_int4
    out.update(qkt_int8=qkt_int8, presoftmax=presoftmax,
               attn_int4=attn_int4, av_int8=av_int8, av_int4=av_int4)

    # ============ 7. Concat heads, O projection ============
    ctx_cat = av_int4.transpose(1, 0, 2).reshape(64, 96)        # (64, 96) int8 [-8,7]
    out["ctx_cat"] = ctx_cat

    o_int8 = vta_3arg(ctx_cat, sim.W_o, 3)
    out["o_int8"] = o_int8
    o_int4 = cpu_requant_after_vta(o_int8, TUNED["o"],
                                    S["o_in"], S["o_w"], S["attn_residual"], -8, 7)
    out["o_int4"] = o_int4

    # ============ 8. Attention residual add ============
    skip_float = after_pos.astype(np.float64) * S["pos_out"]
    skip_int4_attn = quant_signed(skip_float, S["attn_residual"], -8, 7).astype(np.int32)
    attn_block_out = o_int4.astype(np.int32) + skip_int4_attn   # (64, 96) int32
    out["attn_block_out"] = attn_block_out

    # ============ 9. Pre-MLP BN + INT4 quant ============
    x_mlp_in_float = attn_block_out.astype(np.float64) * S["attn_residual"]
    bn_mlp = (x_mlp_in_float - sim.bn_mlp_mean.astype(np.float64)[None, :]) / \
             np.sqrt(sim.bn_mlp_var.astype(np.float64)[None, :] + 1e-5)
    mlp_pre = quant_signed(bn_mlp, S["mlp_bn_out"], -8, 7).astype(np.int8)
    out["mlp_pre"] = mlp_pre

    # ============ 10. fc1 (4-arg, baked shift = 3) ============
    fc1_out8 = vta_4arg(mlp_pre, sim.W_fc1, sim.bias_fc1_int32, 3)   # (64, 384) int8
    out["fc1_out8"] = fc1_out8

    cs_fc1 = TUNED["fc1"]
    recovered = fc1_out8.astype(np.float64) * (2.0 ** cs_fc1)
    real = recovered * (S["mlp_bn_out"] * S["fc1_w"])
    fc1_unsigned = np.clip(np.round(real / S["fc1_out"]), 0, 15).astype(np.int8)
    fc1_signed = (fc1_unsigned.astype(np.int32) - 8).astype(np.int8)
    out.update(fc1_unsigned=fc1_unsigned, fc1_signed=fc1_signed)

    # ============ 11. fc2 (4-arg with corrected bias, baked shift = 4) ============
    fc2_out8 = vta_4arg(fc1_signed, sim.W_fc2, sim.bias_fc2_int32_signed, 4)
    out["fc2_out8"] = fc2_out8

    fc2_int4 = cpu_requant_after_vta(fc2_out8, TUNED["fc2"],
                                      S["fc1_out"], S["fc2_w"], S["mlp_residual"],
                                      -8, 7)
    out["fc2_int4"] = fc2_int4

    # ============ 12. MLP residual add ============
    skip_float_mlp = attn_block_out.astype(np.float64) * S["attn_residual"]
    skip_int_mlp = quant_signed(skip_float_mlp, S["mlp_residual"], -8, 7).astype(np.int32)
    mlp_block_out = fc2_int4.astype(np.int32) + skip_int_mlp
    out["mlp_block_out"] = mlp_block_out

    # ============ 13. Classifier (CPU GAP + float matmul + argmax) ============
    mlp_float = mlp_block_out.astype(np.float64) * S["mlp_residual"]
    gap = mlp_float.mean(axis=0)                                # (96,)
    W_cls_float = sim.W_cls.astype(np.float64) * S["cls_w"]
    logits = gap @ W_cls_float.T + sim.b_cls.astype(np.float64) # (24,)
    pred = int(np.argmax(logits))
    out["logits"] = logits
    out["pred"]   = np.array(pred, dtype=np.int32)
    out["label"]  = np.array(label, dtype=np.int32)

    print(f"\n[result] label={label}  pred={pred}  match={pred == label}")
    print(f"[stages] {len(out)} keys saved")
    np.savez(OUT_NPZ, **out)
    print(f"[save] {OUT_NPZ}")


if __name__ == "__main__":
    main()
