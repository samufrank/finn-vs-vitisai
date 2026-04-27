"""VTA INT4 RadioML Transformer host simulator.

Self-contained numpy-only implementation of the forward pass for
model_int4_norm_none_70.97pct.pt. Intended for:

  (a) Validating the VTA deployment math (integer GEMM + shift-requant) on CPU
      before moving to the board.
  (b) Comparing Mode A (float combined_scale requant, reference) against Mode D
      (power-of-two shift+clip requant, hardware-exact) to measure accuracy loss
      attributable to the power-of-two approximation.

All weights, scales, BN stats, and the learned positional encoding are loaded
from `transformer_scales.npz` produced in Phase 1. No Brevitas import here.

Integer math conventions match VTA:
  - GEMM: int32 accumulator  acc = X_int8 @ W_int8.T
  - 4-arg GEMM: bias added in the int32 accumulator domain
  - Requant Mode A: out = round(acc * combined_float).clip(lo, hi).astype(int8)
  - Requant Mode D: out = (acc >> shift).clip(lo, hi).astype(int8)
  - Unsigned acts (fc1 post-ReLU, emb post-ReLU) stored as uint; for fc2 a signed
    conversion + bias correction keeps the GEMM signed-only (VTA datapath).
"""
from __future__ import annotations
import math
import numpy as np
from pathlib import Path

SQRT_96 = 96 ** -0.5

# Shift values for Mode D, from Phase 1 transformer_deployment_table.md
SHIFTS = {
    "q":    4,   # Q projection
    "k":    4,   # K projection
    "v":    2,   # V projection       (12.99% / 13.41% / 40.11% err)
    "qk":   6,   # Q@K^T per head
    "av":   7,   # attn@V per head
    "o":    4,   # O projection
    "fc1":  4,   # MLP fc1 (unsigned output)
    "fc2":  7,   # MLP fc2
}


class TransformerSim:
    """Numpy implementation of the Brevitas INT4 transformer forward pass.

    Parameters
    ----------
    npz_path : str or Path
        Path to `transformer_scales.npz` (produced by Phase 1).
    """

    def __init__(self, npz_path: str | Path) -> None:
        d = np.load(str(npz_path))

        # Scales (per-int-step)
        self.S = {k[len("scale_"):]: float(d[k]) for k in d.files if k.startswith("scale_")}

        # Weights as int8
        self.W_emb = d["w_emb_conv"].astype(np.int8)    # (96, 2, 1, 16)  INT8
        self.W_q   = d["w_q_proj"].astype(np.int8)      # (96, 96)        INT4
        self.W_k   = d["w_k_proj"].astype(np.int8)
        self.W_v   = d["w_v_proj"].astype(np.int8)
        self.W_o   = d["w_o_proj"].astype(np.int8)
        self.W_fc1 = d["w_fc1"].astype(np.int8)         # (384, 96)       INT4
        self.W_fc2 = d["w_fc2"].astype(np.int8)         # (96, 384)       INT4
        self.W_cls = d["w_cls"].astype(np.int8)         # (24, 96)        INT8

        # Biases (float) — pre-rounded to int32 in the acc domain at init
        self.b_emb = d["b_emb_conv"].astype(np.float32)
        self.b_fc1 = d["b_fc1"].astype(np.float32)
        self.b_fc2 = d["b_fc2"].astype(np.float32)
        self.b_cls = d["b_cls"].astype(np.float32)

        self.bn_emb_mean  = d["bn_emb_mean"].astype(np.float32)
        self.bn_emb_var   = d["bn_emb_var"].astype(np.float32)
        self.bn_attn_mean = d["bn_attn_mean"].astype(np.float32)
        self.bn_attn_var  = d["bn_attn_var"].astype(np.float32)
        self.bn_mlp_mean  = d["bn_mlp_mean"].astype(np.float32)
        self.bn_mlp_var   = d["bn_mlp_var"].astype(np.float32)

        self.pos = d["pos"].astype(np.float32)           # (1, 64, 96)

        # Pre-computed integer biases for 4-arg GEMMs
        # acc_int32 = x_int @ W.T + bias_int32, where bias_int32 ≈ bias / (in*w)
        S = self.S
        self.bias_emb_int32 = np.round(self.b_emb / (S["emb_w"] * S["emb_in"])).astype(np.int32)
        self.bias_fc1_int32 = np.round(self.b_fc1 / (S["fc1_w"] * S["mlp_bn_out"])).astype(np.int32)
        self.bias_fc2_int32 = np.round(self.b_fc2 / (S["fc2_w"] * S["fc1_out"])).astype(np.int32)
        # fc2 input is unsigned [0,15]; convert to signed with zero-point 8 and fold into bias
        # corrected_bias_int32 = bias_int32 + 8 * sum(W, axis=1)   (signed shift correction)
        self.bias_fc2_int32_signed = (
            self.bias_fc2_int32 + 8 * self.W_fc2.astype(np.int32).sum(axis=1)
        )

        self._presoftmax_combined = S["q_out"] * S["k_out"] * SQRT_96 / S["softmax_in"]
        self._av_combined         = S["softmax_out"] * S["v_out"] / S["o_in"]

        self._gemm_diag = None    # populated when diag=True

    # ----- internal helpers -----
    @staticmethod
    def _quant_signed(x_float: np.ndarray, scale: float, lo: int, hi: int) -> np.ndarray:
        return np.clip(np.round(x_float / scale), lo, hi).astype(np.int32)

    @staticmethod
    def _quant_unsigned(x_float: np.ndarray, scale: float, lo: int, hi: int) -> np.ndarray:
        return np.clip(np.round(x_float / scale), lo, hi).astype(np.int32)

    @staticmethod
    def _requant(acc: np.ndarray, mode: str, combined: float, shift: int, lo: int, hi: int) -> np.ndarray:
        """Apply Mode A or Mode D requantization. acc must be int-valued int32 or int64."""
        if mode == "A":
            out = np.round(acc.astype(np.float64) * combined)
        elif mode == "D":
            # numpy right_shift on signed int does arithmetic shift (matches VTA SHR)
            out = np.right_shift(acc.astype(np.int32), shift)
        else:
            raise ValueError(f"bad mode {mode!r}")
        return np.clip(out, lo, hi).astype(np.int32)

    def _record_diag(self, tag: str, arr: np.ndarray) -> None:
        if self._gemm_diag is not None:
            # Only store aggregated stats to bound memory
            self._gemm_diag[tag] = {
                "shape": tuple(arr.shape),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
            }

    # ----- main forward -----
    def forward(self, x_float: np.ndarray, mode: str = "A",
                diag: bool = False, return_intermediates: bool = False) -> np.ndarray:
        """Run one batch through the VTA-sim forward pass.

        x_float : (B, 1, 1024, 2) float32  [same as Brevitas model input]
        mode    : "A" (float requant) or "D" (shift+clip)
        diag    : if True, record per-GEMM int range/mean for post-hoc analysis
        return_intermediates : return dict of all intermediates at each boundary
        returns : logits  int32 [B, 24]   (values ∈ [-128, 127] after cls output-quant)
        """
        S = self.S
        B = x_float.shape[0]
        if diag:
            self._gemm_diag = {}
        inter = {} if return_intermediates else None

        # ============ Patch embedding (INT8) ============
        # Rearrange (B, 1, 1024, 2) -> (B, 2, 1, 1024)
        x = np.transpose(x_float, (0, 3, 1, 2)).astype(np.float32)
        # INT8 signed input quant
        x_int8 = self._quant_signed(x, S["emb_in"], -128, 127)
        # im2col for stride-(1,16) kernel-(1,16) conv: 64 non-overlapping [2,16] patches
        # (B, 2, 1, 1024) -> (B, 2, 1, 64, 16) -> (B, 64, 2, 1, 16) -> (B, 64, 32)
        x_im2col = x_int8.reshape(B, 2, 1, 64, 16).transpose(0, 3, 1, 2, 4).reshape(B, 64, 32)
        W_emb_flat = self.W_emb.reshape(96, 32).astype(np.int32)                # (96, 32)
        acc_emb = x_im2col.astype(np.int32) @ W_emb_flat.T                       # (B, 64, 96)
        acc_emb = acc_emb + self.bias_emb_int32[None, None, :]
        self._record_diag("emb_conv_acc", acc_emb)
        if inter is not None: inter["emb_conv_acc"] = acc_emb.copy()
        # Dequant to float for BN (still a float op before unsigned quantizer)
        emb_float = acc_emb.astype(np.float64) * (S["emb_w"] * S["emb_in"])     # (B, 64, 96)
        # BN2d(affine=False) on channel dim (last axis here)
        emb_bn = (emb_float - self.bn_emb_mean.astype(np.float64)[None, None, :]) / \
                 np.sqrt(self.bn_emb_var.astype(np.float64)[None, None, :] + 1e-5)
        # ReLU
        emb_relu = np.maximum(emb_bn, 0.0)
        # INT8 UNSIGNED quantizer [0, 255]
        emb_quant = self._quant_unsigned(emb_relu, S["emb_out"], 0, 255)         # (B, 64, 96)
        # AdaptiveAvgPool2d((1, 64)): no-op, already 1×64
        if inter is not None: inter["after_emb"] = emb_quant.copy()

        # ============ Positional encoding (INT8) ============
        # QuantEltwiseAdd: input_quant on BOTH sides, then add, then output_quant
        emb_float_for_pos = emb_quant.astype(np.float64) * S["emb_out"]
        emb_q_int = self._quant_signed(emb_float_for_pos, S["pos_in"], -128, 127)
        pos_q_int = self._quant_signed(self.pos.astype(np.float64), S["pos_in"], -128, 127)
        sum_float = (emb_q_int.astype(np.float64) + pos_q_int.astype(np.float64)) * S["pos_in"]
        after_pos = self._quant_signed(sum_float, S["pos_out"], -128, 127)       # (B, 64, 96)
        if inter is not None: inter["after_pos"] = after_pos.copy()

        # ============ Attention block ============
        # Skip-branch saves: value at pos_out_scale  -> will be re-quantized to attn_residual_scale later
        skip_attn_int = after_pos    # INT8 signed at pos_out_scale

        # pre-norm (BN + INT4 quantizer)
        x_after_pos_float = after_pos.astype(np.float64) * S["pos_out"]
        bn_attn = (x_after_pos_float - self.bn_attn_mean.astype(np.float64)[None, None, :]) / \
                  np.sqrt(self.bn_attn_var.astype(np.float64)[None, None, :] + 1e-5)
        pre_int = self._quant_signed(bn_attn, S["attn_pre_out"], -8, 7)          # INT4 signed (B, 64, 96)
        if inter is not None: inter["pre_norm"] = pre_int.copy()

        # Q / K / V projections (3-arg, no bias)
        def _proj(x_int: np.ndarray, W: np.ndarray, w_scale: float, in_scale: float,
                  out_scale: float, shift: int) -> np.ndarray:
            acc = x_int.astype(np.int32) @ W.astype(np.int32).T
            combined = in_scale * w_scale / out_scale
            return self._requant(acc, mode, combined, shift, -8, 7)

        Q = _proj(pre_int, self.W_q, S["q_w"], S["attn_pre_out"], S["q_out"], SHIFTS["q"])
        K = _proj(pre_int, self.W_k, S["k_w"], S["attn_pre_out"], S["k_out"], SHIFTS["k"])
        V = _proj(pre_int, self.W_v, S["v_w"], S["attn_pre_out"], S["v_out"], SHIFTS["v"])
        self._record_diag("q_out", Q); self._record_diag("k_out", K); self._record_diag("v_out", V)
        if inter is not None:
            inter.update(Q=Q.copy(), K=K.copy(), V=V.copy())

        # Reshape to heads: (B, 64, 96) -> (B, 3, 64, 32)
        Qh = Q.reshape(B, 64, 3, 32).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, 64, 3, 32).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, 64, 3, 32).transpose(0, 2, 1, 3)

        # Q @ K^T per head (batched): (B, 3, 64, 32) @ (B, 3, 32, 64) -> (B, 3, 64, 64)
        acc_qk = Qh.astype(np.int32) @ Kh.transpose(0, 1, 3, 2).astype(np.int32)
        presoftmax_int = self._requant(
            acc_qk, mode, self._presoftmax_combined, SHIFTS["qk"], -8, 7
        )
        self._record_diag("presoftmax", presoftmax_int)

        # Softmax on CPU (float)
        scores = presoftmax_int.astype(np.float64) * S["softmax_in"]
        scores = scores - scores.max(axis=-1, keepdims=True)   # numerical stability
        exps = np.exp(scores)
        attn_float = exps / exps.sum(axis=-1, keepdims=True)
        # Post-softmax quant INT4 signed, scale softmax_out
        attn_int = self._quant_signed(attn_float, S["softmax_out"], -8, 7)        # (B, 3, 64, 64)
        self._record_diag("attn_weights", attn_int)

        # attn @ V per head: (B, 3, 64, 64) @ (B, 3, 64, 32) -> (B, 3, 64, 32)
        acc_av = attn_int.astype(np.int32) @ Vh.astype(np.int32)
        ctx_int = self._requant(acc_av, mode, self._av_combined, SHIFTS["av"], -8, 7)
        self._record_diag("ctx", ctx_int)
        # Concat heads: (B, 3, 64, 32) -> (B, 64, 96)
        ctx_cat = ctx_int.transpose(0, 2, 1, 3).reshape(B, 64, 96)

        # O projection (3-arg, no bias). Input already quantized to o_in scale.
        o_out = _proj(ctx_cat, self.W_o, S["o_w"], S["o_in"], S["attn_residual"], SHIFTS["o"])
        self._record_diag("o_out", o_out)
        if inter is not None: inter["o_out"] = o_out.copy()

        # Residual add (shared INT4 quantizer on BOTH branches)
        skip_float_attn = skip_attn_int.astype(np.float64) * S["pos_out"]
        skip_int_attn = self._quant_signed(skip_float_attn, S["attn_residual"], -8, 7)
        attn_block_out = (o_out.astype(np.int32) + skip_int_attn.astype(np.int32))   # range [-16, 14]
        if inter is not None: inter["attn_block_out"] = attn_block_out.copy()

        # ============ MLP block ============
        skip_mlp_int = attn_block_out   # at attn_residual_scale (range [-16, 14])

        # pre-norm (BN + INT4 quantizer)
        x_mlp_in_float = attn_block_out.astype(np.float64) * S["attn_residual"]
        bn_mlp = (x_mlp_in_float - self.bn_mlp_mean.astype(np.float64)[None, None, :]) / \
                 np.sqrt(self.bn_mlp_var.astype(np.float64)[None, None, :] + 1e-5)
        mlp_pre = self._quant_signed(bn_mlp, S["mlp_bn_out"], -8, 7)             # INT4 signed

        # fc1: 96 -> 384, INT4, 4-arg (bias), unsigned post-ReLU output
        acc_fc1 = mlp_pre.astype(np.int32) @ self.W_fc1.astype(np.int32).T       # (B, 64, 384)
        acc_fc1 = acc_fc1 + self.bias_fc1_int32[None, None, :]
        if mode == "A":
            combined = S["mlp_bn_out"] * S["fc1_w"] / S["fc1_out"]
            # Clip to [0, 15] also implements the ReLU (negatives clipped to 0 by min-clip)
            fc1_out = np.clip(np.round(acc_fc1.astype(np.float64) * combined), 0, 15).astype(np.int32)
        else:  # D
            shifted = np.right_shift(acc_fc1, SHIFTS["fc1"])
            fc1_out = np.clip(shifted, 0, 15).astype(np.int32)
        self._record_diag("fc1_out", fc1_out)

        # fc2: 384 -> 96, INT4, 4-arg (bias), input unsigned [0,15] -> signed [-8,7] + bias correction
        fc1_signed = fc1_out.astype(np.int32) - 8                                 # (B, 64, 384)
        acc_fc2 = fc1_signed @ self.W_fc2.astype(np.int32).T                      # (B, 64, 96)
        acc_fc2 = acc_fc2 + self.bias_fc2_int32_signed[None, None, :]
        if mode == "A":
            combined = S["fc1_out"] * S["fc2_w"] / S["mlp_residual"]
            fc2_out = np.clip(np.round(acc_fc2.astype(np.float64) * combined), -8, 7).astype(np.int32)
        else:
            fc2_out = np.clip(np.right_shift(acc_fc2, SHIFTS["fc2"]), -8, 7).astype(np.int32)
        self._record_diag("fc2_out", fc2_out)
        if inter is not None: inter["fc2_out"] = fc2_out.copy()

        # Residual add (shared INT4 quantizer on BOTH branches)
        skip_float_mlp = skip_mlp_int.astype(np.float64) * S["attn_residual"]
        skip_int_mlp = self._quant_signed(skip_float_mlp, S["mlp_residual"], -8, 7)
        mlp_block_out = fc2_out.astype(np.int32) + skip_int_mlp.astype(np.int32)
        if inter is not None: inter["mlp_block_out"] = mlp_block_out.copy()

        # ============ Classifier ============
        # GAP over 64 temporal positions
        mlp_float = mlp_block_out.astype(np.float64) * S["mlp_residual"]
        gap = mlp_float.mean(axis=1)                                              # (B, 96)
        # cls.0 is INT8 weight, no input_quant in Brevitas -> float matmul with dequantized W
        W_cls_float = self.W_cls.astype(np.float64) * S["cls_w"]
        logits = gap @ W_cls_float.T + self.b_cls.astype(np.float64)              # (B, 24)
        # cls.1 INT8 signed output quant
        logits_int = np.clip(np.round(logits / S["cls_out"]), -128, 127).astype(np.int32)
        if inter is not None: inter["logits"] = logits_int.copy()

        return logits_int, inter


# --------------------------------------------------------------------------
# CLI: runs full eval (both modes) and optionally a Brevitas reference.
# Only the __main__ block imports PyTorch/Brevitas.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, sys, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",     default="finn-vs-vitisai/vta/transformer_scales.npz")
    ap.add_argument("--data",    default="finn-vs-vitisai/data/radioml2018_eval_snr_filtered.npz")
    ap.add_argument("--limit",   type=int, default=None, help="limit number of samples")
    ap.add_argument("--batch",   type=int, default=512)
    ap.add_argument("--brevitas", action="store_true", help="also run Brevitas reference")
    ap.add_argument("--out-json", default="finn-vs-vitisai/vta/transformer_sim_results.json")
    ap.add_argument("--out-log",  default="finn-vs-vitisai/vta/transformer_sim_mode_d.log")
    ap.add_argument("--modes",   default="A,D")
    args = ap.parse_args()

    sim = TransformerSim(args.npz)
    print(f"[init] TransformerSim ready. scales={list(sim.S.keys())[:4]}... weights loaded.")

    data = np.load(args.data)
    sigs_all = data["signals"]
    labels_all = data["labels"]
    if args.limit is not None:
        sigs_all = sigs_all[: args.limit]
        labels_all = labels_all[: args.limit]
    N = sigs_all.shape[0]
    print(f"[data] {N} samples, signal shape {sigs_all.shape[1:]}, label range {labels_all.min()}..{labels_all.max()}")

    # ============ Sim modes ============
    results = {"n_samples": N, "modes": {}}
    preds_per_mode = {}
    modes = [m.strip().upper() for m in args.modes.split(",")]
    for mode in modes:
        t0 = time.time()
        preds = np.empty(N, dtype=np.int32)
        for i in range(0, N, args.batch):
            j = min(i + args.batch, N)
            logits, _ = sim.forward(sigs_all[i:j], mode=mode)
            preds[i:j] = np.argmax(logits, axis=-1)
            if i % (args.batch * 20) == 0 and i > 0:
                rate = i / (time.time() - t0)
                print(f"  [mode {mode}] {i}/{N}  ({rate:.1f} samp/s)")
        dt = time.time() - t0
        acc = float((preds == labels_all).mean())
        results["modes"][mode] = {"accuracy": acc, "time_sec": dt}
        preds_per_mode[mode] = preds
        print(f"[mode {mode}] done in {dt:.1f}s  acc={acc*100:.2f}%")

    # Mode A <-> Mode D agreement
    if "A" in preds_per_mode and "D" in preds_per_mode:
        agree = int(np.sum(preds_per_mode["A"] == preds_per_mode["D"]))
        results["modeA_modeD_agreement"] = agree / N
        print(f"[A vs D] argmax agreement: {agree}/{N} = {agree/N*100:.2f}%")

    # ============ Brevitas reference ============
    if args.brevitas:
        import sys, warnings
        warnings.filterwarnings("ignore")
        sys.path.insert(0, "finn-transformers")
        import torch
        from radioml.model import Model

        print("[brev] loading Brevitas model...")
        model = Model(
            num_classes=24,
            embedding={"patches":[1,64], "kernel_size":[1,16], "stride":[1,16],
                       "padding":[0,0], "activation":"relu", "bits":8},
            positional={"encoding":"learned","bits":8},
            configuration="original", num_layers=1, num_heads=3,
            emb_dim=96, expansion_dim=384, bits=4, cls_bits=8,
            activation="relu", norm="none", norm_placement="pre-norm", dropout=0.0,
        )
        with torch.no_grad():
            model(torch.zeros(1, 1, 1024, 2))
        model.load_state_dict(torch.load(
            "finn-transformers/outputs/radioml/model_int4_norm_none_70.97pct.pt",
            map_location="cpu", weights_only=False))
        model.eval()

        t0 = time.time()
        brev_preds = np.empty(N, dtype=np.int32)
        with torch.no_grad():
            for i in range(0, N, args.batch):
                j = min(i + args.batch, N)
                x = torch.from_numpy(sigs_all[i:j]).float()
                logits = model(x)
                brev_preds[i:j] = logits.argmax(dim=-1).numpy()
                if i % (args.batch * 10) == 0 and i > 0:
                    rate = i / (time.time() - t0)
                    print(f"  [brev] {i}/{N}  ({rate:.1f} samp/s)")
        dt = time.time() - t0
        brev_acc = float((brev_preds == labels_all).mean())
        results["brevitas"] = {"accuracy": brev_acc, "time_sec": dt}
        print(f"[brev] done in {dt:.1f}s  acc={brev_acc*100:.2f}%")

        for mode in modes:
            dis = int(np.sum(brev_preds != preds_per_mode[mode]))
            results["modes"][mode][f"disagreements_vs_brevitas"] = dis
            print(f"[mode {mode}] disagreements vs Brevitas: {dis}/{N}  acc_delta={results['modes'][mode]['accuracy']*100 - brev_acc*100:+.2f}pp")

    results["shifts"] = SHIFTS

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.out_json}")

    if "D" in preds_per_mode:
        with open(args.out_log, "w") as f:
            f.write("# Mode D argmax predictions, one per line\n")
            f.write(f"# N={N}\n")
            for p in preds_per_mode["D"]:
                f.write(f"{int(p)}\n")
        print(f"[save] {args.out_log}")
