"""VTA INT4-o8 host simulator.

Extends the Phase-2 `vta_transformer_sim.TransformerSim` with two additional
modes and optional accumulator capture:

  Mode E  ("int4-o8" + CPU float requant) — VTA executes the integer GEMM and
          shifts the int32 accumulator right by a *coarse* shift just large
          enough to fit the result in int8. VTA emits int8 via DMA. The host
          CPU undoes the shift in float64, applies the true w*in/out scale
          ratio (and the 1/sqrt(96) attention-scale for Q@K^T), and requantizes
          to the activation precision used downstream.

  Mode C  Board-realistic variant of Mode E: always clips to int4 SIGNED
          [-8, 7]. Brevitas-trained unsigned-int4 activations (post-ReLU)
          are stored as signed int4 with a zero-point offset of 8; the
          downstream GEMM's bias is corrected with +8 * sum(W, axis=1) so
          the arithmetic result is identical.  (Parent sim already applies
          that correction to fc2 input, so Mode C ≡ Mode E for this model.)

The Phase-2 Mode A (float requant) and Mode D (single shift+clip) stay
unchanged and are inherited.  The patch-embedding conv and classifier GEMM
run on CPU in all modes (they are outside the 12 VTA GEMMs).

This file imports from vta_transformer_sim; it does NOT modify it.
"""
from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np

from sim import TransformerSim, SHIFTS, SQRT_96   # post-reorg name (was vta_transformer_sim)


# ------------------------------------------------------------------
# Coarse shifts (int32 -> int8) for the 12 VTA INT4 GEMMs.
# Populated from Phase-3 Task-1 measurements. When empty, the sim falls
# back to 0 (no coarse shift) — which may clip large accumulators.
# ------------------------------------------------------------------
COARSE_SHIFTS: dict[str, int] = {
    # "emb_conv" is INT8 path on CPU, not a VTA GEMM.
    # Will be populated after Task-1 measurement.
}


class TransformerSimO8(TransformerSim):
    """Transformer sim extended with Mode E / Mode C and acc capture."""

    @staticmethod
    def _requant_E(acc: np.ndarray, coarse_shift: int, w_scale: float,
                   in_scale: float, out_scale: float,
                   clip_lo: int, clip_hi: int,
                   extra_factor: float = 1.0) -> np.ndarray:
        """Mode E requant for a GEMM.

        VTA: arithmetic right-shift the int32 accumulator by `coarse_shift`
             and clip to int8 [-128, 127].
        CPU: float64: recover acc by <<coarse_shift, apply scale ratio
             w_scale * in_scale * extra_factor / out_scale, round, clip.
        """
        acc_shifted = np.right_shift(acc.astype(np.int32), coarse_shift)
        acc_int8 = np.clip(acc_shifted, -128, 127).astype(np.int32)    # board emits int8
        recovered = acc_int8.astype(np.float64) * (2.0 ** coarse_shift)
        real = recovered * (w_scale * in_scale * extra_factor)
        return np.clip(np.round(real / out_scale), clip_lo, clip_hi).astype(np.int32)

    @staticmethod
    def _requant_any(acc: np.ndarray, mode: str, combined: float, shift: int,
                     clip_lo: int, clip_hi: int,
                     # Mode-E/C extras:
                     coarse_shift: int = 0,
                     w_scale: float | None = None,
                     in_scale: float | None = None,
                     out_scale: float | None = None,
                     extra_factor: float = 1.0) -> np.ndarray:
        if mode == "A":
            out = np.round(acc.astype(np.float64) * combined)
        elif mode == "D":
            out = np.right_shift(acc.astype(np.int32), shift)
        elif mode in ("E", "C"):
            return TransformerSimO8._requant_E(
                acc, coarse_shift, w_scale, in_scale, out_scale,
                clip_lo, clip_hi, extra_factor
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return np.clip(out, clip_lo, clip_hi).astype(np.int32)

    # ------------------------------------------------------------------
    # Override forward() to support all four modes and accumulator capture.
    # Copy of the parent implementation with added branches.
    # ------------------------------------------------------------------
    def forward(self, x_float: np.ndarray, mode: str = "A",
                coarse_shifts: dict | None = None,
                capture_acc: bool = False,
                diag: bool = False,
                return_intermediates: bool = False,
                modeC_clip: bool = False) -> tuple[np.ndarray, dict | None]:
        S = self.S
        B = x_float.shape[0]
        if coarse_shifts is None:
            coarse_shifts = COARSE_SHIFTS
        accs: dict[str, np.ndarray] | None = {} if capture_acc else None
        inter: dict | None = {} if return_intermediates else None
        if diag:
            self._gemm_diag = {}

        # ============ Patch embedding (INT8 CPU path, unchanged) ============
        x = np.transpose(x_float, (0, 3, 1, 2)).astype(np.float32)
        x_int8 = self._quant_signed(x, S["emb_in"], -128, 127)
        x_im2col = x_int8.reshape(B, 2, 1, 64, 16).transpose(0, 3, 1, 2, 4).reshape(B, 64, 32)
        W_emb_flat = self.W_emb.reshape(96, 32).astype(np.int32)
        acc_emb = x_im2col.astype(np.int32) @ W_emb_flat.T + self.bias_emb_int32[None, None, :]
        if accs is not None: accs["emb_conv"] = acc_emb.copy()
        emb_float = acc_emb.astype(np.float64) * (S["emb_w"] * S["emb_in"])
        emb_bn = (emb_float - self.bn_emb_mean.astype(np.float64)[None, None, :]) / \
                 np.sqrt(self.bn_emb_var.astype(np.float64)[None, None, :] + 1e-5)
        emb_relu = np.maximum(emb_bn, 0.0)
        emb_quant = self._quant_unsigned(emb_relu, S["emb_out"], 0, 255)

        # ============ Positional encoding (unchanged) ============
        emb_float_for_pos = emb_quant.astype(np.float64) * S["emb_out"]
        emb_q_int = self._quant_signed(emb_float_for_pos, S["pos_in"], -128, 127)
        pos_q_int = self._quant_signed(self.pos.astype(np.float64), S["pos_in"], -128, 127)
        sum_float = (emb_q_int.astype(np.float64) + pos_q_int.astype(np.float64)) * S["pos_in"]
        after_pos = self._quant_signed(sum_float, S["pos_out"], -128, 127)

        # ============ Attention ============
        skip_attn_int = after_pos
        x_after_pos_float = after_pos.astype(np.float64) * S["pos_out"]
        bn_attn = (x_after_pos_float - self.bn_attn_mean.astype(np.float64)[None, None, :]) / \
                  np.sqrt(self.bn_attn_var.astype(np.float64)[None, None, :] + 1e-5)
        pre_int = self._quant_signed(bn_attn, S["attn_pre_out"], -8, 7)

        # 3-arg GEMM helper (Q/K/V/o_proj)
        def _proj3(x_int: np.ndarray, W: np.ndarray, w_scale: float,
                   in_scale: float, out_scale: float,
                   shift: int, coarse_key: str, lo: int, hi: int,
                   extra: float = 1.0, name_for_capture: str | None = None) -> np.ndarray:
            acc = x_int.astype(np.int32) @ W.astype(np.int32).T
            if accs is not None and name_for_capture is not None:
                accs[name_for_capture] = acc.copy()
            combined = in_scale * w_scale / out_scale * extra
            return self._requant_any(
                acc, mode, combined, shift, lo, hi,
                coarse_shift=coarse_shifts.get(coarse_key, 0),
                w_scale=w_scale, in_scale=in_scale, out_scale=out_scale,
                extra_factor=extra,
            )

        Q = _proj3(pre_int, self.W_q, S["q_w"], S["attn_pre_out"], S["q_out"],
                   SHIFTS["q"], "q", -8, 7, name_for_capture="q")
        K = _proj3(pre_int, self.W_k, S["k_w"], S["attn_pre_out"], S["k_out"],
                   SHIFTS["k"], "k", -8, 7, name_for_capture="k")
        V = _proj3(pre_int, self.W_v, S["v_w"], S["attn_pre_out"], S["v_out"],
                   SHIFTS["v"], "v", -8, 7, name_for_capture="v")
        if inter is not None:
            inter.update(Q=Q.copy(), K=K.copy(), V=V.copy())

        # Heads
        Qh = Q.reshape(B, 64, 3, 32).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, 64, 3, 32).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, 64, 3, 32).transpose(0, 2, 1, 3)

        # Q@K^T  (activation × activation, attention-scale via extra_factor)
        acc_qk = Qh.astype(np.int32) @ Kh.transpose(0, 1, 3, 2).astype(np.int32)   # (B,3,64,64)
        if accs is not None: accs["qk"] = acc_qk.copy()
        # For Mode E/C we pass: in scales are q_out and k_out (both activation), out is softmax_in,
        # with extra_factor = 1/sqrt(96).
        combined_qk = S["q_out"] * S["k_out"] * SQRT_96 / S["softmax_in"]
        presoftmax_int = self._requant_any(
            acc_qk, mode, combined_qk, SHIFTS["qk"], -8, 7,
            coarse_shift=coarse_shifts.get("qk", 0),
            w_scale=S["q_out"], in_scale=S["k_out"], out_scale=S["softmax_in"],
            extra_factor=SQRT_96,
        )

        # Softmax (float, CPU)
        scores = presoftmax_int.astype(np.float64) * S["softmax_in"]
        scores = scores - scores.max(axis=-1, keepdims=True)
        exps = np.exp(scores)
        attn_float = exps / exps.sum(axis=-1, keepdims=True)
        attn_int = self._quant_signed(attn_float, S["softmax_out"], -8, 7)

        # attn@V
        acc_av = attn_int.astype(np.int32) @ Vh.astype(np.int32)                   # (B,3,64,32)
        if accs is not None: accs["av"] = acc_av.copy()
        combined_av = S["softmax_out"] * S["v_out"] / S["o_in"]
        ctx_int = self._requant_any(
            acc_av, mode, combined_av, SHIFTS["av"], -8, 7,
            coarse_shift=coarse_shifts.get("av", 0),
            w_scale=S["softmax_out"], in_scale=S["v_out"], out_scale=S["o_in"],
        )
        ctx_cat = ctx_int.transpose(0, 2, 1, 3).reshape(B, 64, 96)

        # O projection
        o_out = _proj3(ctx_cat, self.W_o, S["o_w"], S["o_in"], S["attn_residual"],
                       SHIFTS["o"], "o", -8, 7, name_for_capture="o")
        if inter is not None: inter["o_out"] = o_out.copy()

        # Residual add (attention)
        skip_float_attn = skip_attn_int.astype(np.float64) * S["pos_out"]
        skip_int_attn = self._quant_signed(skip_float_attn, S["attn_residual"], -8, 7)
        attn_block_out = o_out.astype(np.int32) + skip_int_attn.astype(np.int32)

        # ============ MLP ============
        skip_mlp_int = attn_block_out
        x_mlp_in_float = attn_block_out.astype(np.float64) * S["attn_residual"]
        bn_mlp = (x_mlp_in_float - self.bn_mlp_mean.astype(np.float64)[None, None, :]) / \
                 np.sqrt(self.bn_mlp_var.astype(np.float64)[None, None, :] + 1e-5)
        mlp_pre = self._quant_signed(bn_mlp, S["mlp_bn_out"], -8, 7)

        # fc1 (4-arg, unsigned output)
        acc_fc1 = mlp_pre.astype(np.int32) @ self.W_fc1.astype(np.int32).T + self.bias_fc1_int32[None, None, :]
        if accs is not None: accs["fc1"] = acc_fc1.copy()
        if mode == "A":
            combined = S["mlp_bn_out"] * S["fc1_w"] / S["fc1_out"]
            fc1_out = np.clip(np.round(acc_fc1.astype(np.float64) * combined), 0, 15).astype(np.int32)
        elif mode == "D":
            fc1_out = np.clip(np.right_shift(acc_fc1, SHIFTS["fc1"]), 0, 15).astype(np.int32)
        elif mode in ("E", "C"):
            # Mode C: clip to [-8, 7] signed with zero-point offset 8 applied later.
            # Mode E: clip to [0, 15] unsigned, subtract 8 before fc2 GEMM.
            cs = coarse_shifts.get("fc1", 0)
            acc_shifted = np.right_shift(acc_fc1.astype(np.int32), cs)
            acc_int8 = np.clip(acc_shifted, -128, 127).astype(np.int32)
            recovered = acc_int8.astype(np.float64) * (2.0 ** cs)
            real = recovered * (S["mlp_bn_out"] * S["fc1_w"])
            if mode == "C":
                # Unsigned -> signed with zero-point: out_signed = round(real/fc1_out) - 8, clip [-8,7]
                out_unsigned = np.clip(np.round(real / S["fc1_out"]), 0, 15)
                fc1_out = (out_unsigned - 8).astype(np.int32)  # signed [-8,7]
            else:  # E: unsigned [0, 15]
                fc1_out = np.clip(np.round(real / S["fc1_out"]), 0, 15).astype(np.int32)

        # fc2 (4-arg, signed output); fc1 input unsigned -> signed via zero-point 8
        if mode == "C":
            fc1_signed = fc1_out   # already signed [-8, 7]
        else:
            fc1_signed = fc1_out.astype(np.int32) - 8
        acc_fc2 = fc1_signed @ self.W_fc2.astype(np.int32).T + self.bias_fc2_int32_signed[None, None, :]
        if accs is not None: accs["fc2"] = acc_fc2.copy()
        if mode == "A":
            combined = S["fc1_out"] * S["fc2_w"] / S["mlp_residual"]
            fc2_out = np.clip(np.round(acc_fc2.astype(np.float64) * combined), -8, 7).astype(np.int32)
        elif mode == "D":
            fc2_out = np.clip(np.right_shift(acc_fc2, SHIFTS["fc2"]), -8, 7).astype(np.int32)
        elif mode in ("E", "C"):
            cs = coarse_shifts.get("fc2", 0)
            acc_shifted = np.right_shift(acc_fc2.astype(np.int32), cs)
            acc_int8 = np.clip(acc_shifted, -128, 127).astype(np.int32)
            recovered = acc_int8.astype(np.float64) * (2.0 ** cs)
            real = recovered * (S["fc1_out"] * S["fc2_w"])
            fc2_out = np.clip(np.round(real / S["mlp_residual"]), -8, 7).astype(np.int32)

        # Residual add (MLP)
        skip_float_mlp = skip_mlp_int.astype(np.float64) * S["attn_residual"]
        skip_int_mlp = self._quant_signed(skip_float_mlp, S["mlp_residual"], -8, 7)
        mlp_block_out = fc2_out.astype(np.int32) + skip_int_mlp.astype(np.int32)

        # Classifier (CPU float)
        mlp_float = mlp_block_out.astype(np.float64) * S["mlp_residual"]
        gap = mlp_float.mean(axis=1)
        W_cls_float = self.W_cls.astype(np.float64) * S["cls_w"]
        logits = gap @ W_cls_float.T + self.b_cls.astype(np.float64)
        logits_int = np.clip(np.round(logits / S["cls_out"]), -128, 127).astype(np.int32)
        if inter is not None: inter["logits"] = logits_int.copy()

        # Return: in capture_acc mode we return the accs dict too
        if capture_acc:
            return logits_int, accs
        return logits_int, inter


# ---------------------------------------------------------------------
# CLI driver for Phase-3: runs Task-1 measurement, Task-3/5 full eval,
# Task-4 sensitivity, and writes transformer_sim_results_phase3.json.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, time, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",    default="finn-vs-vitisai/vta/transformer_scales.npz")
    ap.add_argument("--data",   default="finn-vs-vitisai/data/radioml2018_eval_snr_filtered.npz")
    ap.add_argument("--stage",  required=True,
                    choices=["measure", "mode_e_full", "mode_c_full", "sensitivity"])
    ap.add_argument("--out",    default=None)
    ap.add_argument("--batch",  type=int, default=512)
    ap.add_argument("--samples",type=int, default=1000)   # for measure
    ap.add_argument("--coarse-shifts", default=None,
                    help="JSON file with coarse shifts (for mode_e/c/sensitivity)")
    args = ap.parse_args()

    sim = TransformerSimO8(args.npz)
    data = np.load(args.data)
    sigs  = data["signals"]
    labels = data["labels"]
    N = len(sigs)

    if args.coarse_shifts:
        with open(args.coarse_shifts) as f:
            COARSE_SHIFTS.update(json.load(f))
        print(f"[init] loaded coarse shifts: {COARSE_SHIFTS}")

    def run_full(mode: str, batch: int, coarse: dict | None = None) -> tuple[np.ndarray, float, float]:
        preds = np.empty(N, dtype=np.int32)
        t0 = time.time()
        for i in range(0, N, batch):
            j = min(i + batch, N)
            logits, _ = sim.forward(sigs[i:j], mode=mode, coarse_shifts=coarse or COARSE_SHIFTS)
            preds[i:j] = np.argmax(logits, axis=-1)
            if i % (batch * 50) == 0 and i > 0:
                rate = i / (time.time() - t0)
                print(f"  [{mode}] {i}/{N}  ({rate:.0f} samp/s)", flush=True)
        dt = time.time() - t0
        acc = float((preds == labels).mean())
        return preds, acc, dt

    # --------- stage: measure accumulator ranges on 1000 samples ---------
    if args.stage == "measure":
        idx = np.linspace(0, N - 1, args.samples, dtype=np.int64)
        print(f"[measure] sampling {len(idx)} evenly-spaced signals (from 0..{N-1})")
        x = sigs[idx]
        # Run in small batches to keep memory bounded.
        all_accs = {}
        for i in range(0, len(x), 100):
            j = min(i + 100, len(x))
            _, accs_batch = sim.forward(x[i:j], mode="A", capture_acc=True)
            for k, v in accs_batch.items():
                all_accs.setdefault(k, []).append(v.astype(np.int64))
        stats = {}
        for k, chunks in all_accs.items():
            arr = np.concatenate([c.reshape(-1) for c in chunks])
            amax = int(np.abs(arr).max())
            mean_abs = float(np.abs(arr).mean())
            std = float(arr.std())
            shift_coarse = max(0, int(math.ceil(math.log2(amax / 127.0))) if amax > 127 else 0)
            shift_full   = max(0, int(math.ceil(math.log2(amax / 7.0)))   if amax > 7   else 0)
            # Clipping rates for int8 at shift_coarse and at shift_coarse+1
            if shift_coarse > 0:
                shifted = np.right_shift(arr.astype(np.int32), shift_coarse)
            else:
                shifted = arr.astype(np.int32)
            clipped0 = float(np.sum((shifted < -128) | (shifted > 127))) / len(shifted)
            shifted1 = np.right_shift(arr.astype(np.int32), shift_coarse + 1)
            clipped1 = float(np.sum((shifted1 < -128) | (shifted1 > 127))) / len(shifted1)
            shifted_m1 = np.right_shift(arr.astype(np.int32), max(0, shift_coarse - 1))
            clipped_m1 = float(np.sum((shifted_m1 < -128) | (shifted_m1 > 127))) / len(shifted_m1)
            stats[k] = {
                "min": int(arr.min()), "max": int(arr.max()), "amax": amax,
                "mean_abs": mean_abs, "std": std,
                "shift_coarse_int8": shift_coarse,
                "shift_full_int4": shift_full,
                "clip_rate_at_coarse_pct":     clipped0 * 100.0,
                "clip_rate_at_coarse_plus_1":  clipped1 * 100.0,
                "clip_rate_at_coarse_minus_1": clipped_m1 * 100.0,
            }
        # Report
        hdr = f"{'gemm':10s} {'min':>10s} {'max':>10s} {'|a|max':>10s} {'|a|mean':>10s} " \
              f"{'std':>10s} {'sh_coarse':>10s} {'sh_full':>8s} {'clip%':>6s}"
        print(hdr); print("-" * len(hdr))
        for k, s in stats.items():
            print(f"{k:10s} {s['min']:>10d} {s['max']:>10d} {s['amax']:>10d} "
                  f"{s['mean_abs']:>10.1f} {s['std']:>10.1f} "
                  f"{s['shift_coarse_int8']:>10d} {s['shift_full_int4']:>8d} "
                  f"{s['clip_rate_at_coarse_pct']:>5.2f}")
        # Save shifts json (just the shift values) + full stats
        out = args.out or "finn-vs-vitisai/vta/phase3_acc_stats.json"
        coarse_only = {k: int(v["shift_coarse_int8"]) for k, v in stats.items() if k != "emb_conv"}
        with open(out, "w") as f:
            json.dump({"stats": stats, "coarse_shifts": coarse_only}, f, indent=2)
        print(f"[save] {out}")
        # Dump shifts only
        shifts_out = args.out.replace(".json", "_shifts.json") if args.out else \
                     "finn-vs-vitisai/vta/phase3_coarse_shifts.json"
        with open(shifts_out, "w") as f:
            json.dump(coarse_only, f, indent=2)
        print(f"[save] {shifts_out}")

    elif args.stage in ("mode_e_full", "mode_c_full"):
        mode = "E" if args.stage == "mode_e_full" else "C"
        preds, acc, dt = run_full(mode, args.batch)
        print(f"[{mode}] full-eval acc={acc*100:.2f}%  time={dt:.1f}s")
        out = args.out or f"finn-vs-vitisai/vta/phase3_mode_{mode.lower()}_preds.npy"
        np.save(out, preds)
        with open(out.replace(".npy", ".json"), "w") as f:
            json.dump({"accuracy": acc, "time_sec": dt, "mode": mode,
                       "coarse_shifts": COARSE_SHIFTS}, f, indent=2)
        print(f"[save] {out}  and  {out.replace('.npy', '.json')}")

    elif args.stage == "sensitivity":
        # Per-GEMM, try +1 on the coarse shift (one GEMM at a time)
        base_shifts = dict(COARSE_SHIFTS)
        base_preds, base_acc, _ = run_full("E", args.batch, coarse=base_shifts)
        print(f"[sens] baseline (Mode E) acc={base_acc*100:.2f}%")
        results = {"baseline": {"shifts": base_shifts, "accuracy": base_acc}}
        for k in base_shifts.keys():
            over = dict(base_shifts); over[k] = base_shifts[k] + 1
            _, acc, dt = run_full("E", args.batch, coarse=over)
            results[f"{k}+1"] = {"shifts": over, "accuracy": acc, "time_sec": dt}
            print(f"[sens] +1 on {k:5s}: acc={acc*100:.2f}%  ({dt:.0f}s)")
        out = args.out or "finn-vs-vitisai/vta/phase3_sensitivity.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[save] {out}")
