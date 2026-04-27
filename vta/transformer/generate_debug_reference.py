#!/usr/bin/env python3
"""Host-side: produce sample-0 reference intermediates the board diagnostic compares against.

Runs the Phase-3 numpy sim (`TransformerSimO8` with tuned coarse shifts) on
sample 0 of `radioml2018_eval_snr_filtered.npz` and saves a single .npz with:

  - signal:           the raw float32 (1, 1024, 2) sample
  - label:            ground-truth class (0..23)
  - mode_e_pred:      argmax from sim (= what the board should produce too)
  - emb_q:            (64, 96) UNSIGNED INT8, after patch embed + BN + ReLU + unsigned-quant
  - after_pos:        (64, 96) signed INT8, after positional add
  - pre_norm:         (64, 96) signed INT4 (in int8 storage), after BN_attn + INT4 quant
  - Q_int8_expected:  (64, 96) signed INT8 = clip( (pre_norm @ W_q.T) >> shift_q, -128, 127 ).
                      This is exactly what the VTA proj_s3 module should DMA out for sample 0.
  - q_shift:          the shift baked into proj_s3 (=3, the Phase-3 tuned value).

Run on the host (finn-t-env or any env with the sim deps):
    python finn-vs-vitisai/vta/generate_debug_reference.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent           # finn-vs-vitisai/vta
sys.path.insert(0, str(HERE))
from vta_transformer_sim_o8 import TransformerSimO8

NPZ_SCALES = HERE / "transformer_scales.npz"
DATA       = HERE.parent / "data" / "radioml2018_eval_snr_filtered.npz"
WEIGHTS_SRC = HERE / "transformer_weights"
COARSE_PATH = HERE / "phase3_coarse_shifts.json"
OUT_NPZ     = HERE / "debug_reference_sample0.npz"

# Tuned shifts from Phase-3 (baseline-1 on v, o, fc2)
with open(COARSE_PATH) as f:
    COARSE = json.load(f)
COARSE["v"]   = 2
COARSE["o"]   = 3
COARSE["fc2"] = 4

print(f"[init] sim from {NPZ_SCALES.name}")
sim = TransformerSimO8(str(NPZ_SCALES))

print(f"[data] loading sample 0 from {DATA.name}")
data = np.load(DATA)
sig = data["signals"][0:1]                 # (1, 1, 1024, 2)
label = int(data["labels"][0])

print(f"[run]  Mode E with coarse shifts {COARSE}")
logits, _ = sim.forward(sig, mode="E", coarse_shifts=COARSE)
mode_e_pred = int(np.argmax(logits[0]))
print(f"  label={label}  mode_e_pred={mode_e_pred}")

# Recompute the pre-VTA intermediates manually (the sim's inter dict in Mode E
# doesn't expose them; the math is short enough to mirror inline). Same code
# path as TransformerSimO8.forward up through pre_norm.
S = sim.S
sig0 = sig[0]                                                 # (1, 1024, 2)
x = np.transpose(sig0[None, :, :, :], (0, 3, 1, 2)).astype(np.float32)
x_int8 = sim._quant_signed(x, S["emb_in"], -128, 127)
x_im2col = x_int8.reshape(1, 2, 1, 64, 16).transpose(0, 3, 1, 2, 4).reshape(1, 64, 32)
acc_emb = (x_im2col.astype(np.int32) @ sim.W_emb.reshape(96, 32).astype(np.int32).T
            + sim.bias_emb_int32[None, None, :])
emb_float = acc_emb.astype(np.float64) * (S["emb_w"] * S["emb_in"])
emb_bn = (emb_float - sim.bn_emb_mean.astype(np.float64)[None, None, :]) / \
         np.sqrt(sim.bn_emb_var.astype(np.float64)[None, None, :] + 1e-5)
emb_relu = np.maximum(emb_bn, 0.0)
emb_q_full = sim._quant_unsigned(emb_relu, S["emb_out"], 0, 255)               # (1, 64, 96)

emb_for_pos = emb_q_full.astype(np.float64) * S["emb_out"]
emb_q_int = sim._quant_signed(emb_for_pos, S["pos_in"], -128, 127)
pos_q_int = sim._quant_signed(sim.pos.astype(np.float64), S["pos_in"], -128, 127)
sum_float = (emb_q_int.astype(np.float64) + pos_q_int.astype(np.float64)) * S["pos_in"]
after_pos_full = sim._quant_signed(sum_float, S["pos_out"], -128, 127)         # (1, 64, 96)

x_pos_float = after_pos_full.astype(np.float64) * S["pos_out"]
bn_attn = (x_pos_float - sim.bn_attn_mean.astype(np.float64)[None, None, :]) / \
          np.sqrt(sim.bn_attn_var.astype(np.float64)[None, None, :] + 1e-5)
pre_norm_full = sim._quant_signed(bn_attn, S["attn_pre_out"], -8, 7)           # (1, 64, 96)

emb_q     = emb_q_full[0].astype(np.int16)        # uint8 values [0, 255]
after_pos = after_pos_full[0].astype(np.int8)
pre_norm  = pre_norm_full[0].astype(np.int8)
print(f"  emb_q     shape={emb_q.shape}     range=[{emb_q.min()}, {emb_q.max()}]")
print(f"  after_pos shape={after_pos.shape} range=[{after_pos.min()}, {after_pos.max()}]")
print(f"  pre_norm  shape={pre_norm.shape}  range=[{pre_norm.min()}, {pre_norm.max()}]")

# Compute the int8 output that proj_s3 should DMA out: (pre_norm @ W_q.T) >> shift_q, clipped.
# This is the bit-exact integer math VTA performs. Comparing the board's VTA output
# against this isolates whether the GEMM + shift + clip + DMA path is correct.
W_q = np.load(WEIGHTS_SRC / "q_proj_W_int.npy").astype(np.int8)   # (96, 96)
shift_q = 3
acc = pre_norm.astype(np.int32) @ W_q.astype(np.int32).T          # (64, 96)
Q_int8_expected = np.clip(np.right_shift(acc, shift_q), -128, 127).astype(np.int8)
print(f"  Q_int8_expected shape={Q_int8_expected.shape} range=[{Q_int8_expected.min()}, "
      f"{Q_int8_expected.max()}]")

np.savez(OUT_NPZ,
         signal=sig[0].astype(np.float32),
         label=np.int64(label),
         mode_e_pred=np.int64(mode_e_pred),
         emb_q=emb_q,
         after_pos=after_pos,
         pre_norm=pre_norm,
         Q_int8_expected=Q_int8_expected,
         q_shift=np.int64(shift_q),
         coarse_shifts=np.array(json.dumps(COARSE)))
print(f"[save] {OUT_NPZ}")
