# Steady-state vs all-runs power — DPU and FINN (decision input)

**Question for Sam:** the DPU per-run data shows run 1 dynamic power consistently
below runs 2–3 (warmup). Should the paper report steady-state (runs 2–3) or the
all-3 average — and is dropping run 1 *symmetric* across frameworks? This file
quantifies it on both sides. **No policy is applied here; nothing is edited.**

## Headline (MEASURED)
- **The run-1-low warmup is present on BOTH frameworks, universally:** run 1 is the
  strict minimum dynamic power in **14/14 DPU models** and **31/31 FINN builds**
  scanned (45/45 = 100%). run 2 ≈ run 3 everywhere → steady state is reached by run 2.
- **FINN persists per-run power identically** (`runs[].fnb58_power.dynamic_power_w`,
  3 runs) — 31/31 builds. **No persistence asymmetry.**
- Effect on **dynamic power** (runs-2-3 vs all-3): DPU **+9% to +18%**, FINN
  **+0.3% to +36%** (magnitude tracks run *duration*, not framework — see below).
- Effect on **energy/image** is small on both (DPU **+0.9–1.4%**, FINN **+0.0–3.1%**)
  because energy uses *total* board power (idle ~3.3–4.4 W dominates the ~0.04–0.5 W
  dynamic). **The basis choice mostly moves the dynamic-power number, not energy.**

## Method (MEASURED)
- Per run: `runs[].fnb58_power.dynamic_power_w` (run_power − idle). all-3 = mean of
  runs 1–3 (= `summary.dynamic_power_w`); steady = mean of runs 2–3.
- Energy per run: `runs[].energy_per_image_mj` (= 1000 × that run's *total* power ÷
  that run's throughput); all-3 vs runs-2-3 averages the same per-run values over
  different run subsets (throughput stays per-run-consistent).
- ⚠️ Use `fnb58_power.*`, **not** the top-level `runs[].power_samples` (=0, stale
  INA260 placeholder) or a single blended number.

## DPU — all 14 (MEASURED)
dyn = dynamic_power_w (W); run length ≈ 10000/fps ≈ 3.6–6.4 s.

| model | r1 | r2 | r3 | all-3 | runs2-3 | Δdyn | Δdyn% | E all-3 | E 2-3 | ΔE% |
|-------|---:|---:|---:|------:|--------:|-----:|------:|--------:|------:|----:|
| cnn_tiny | 0.165 | 0.288 | 0.290 | 0.248 | 0.289 | +0.041 | +16.7% | 1.760 | 1.777 | +1.0% |
| cnn_small | 0.183 | 0.317 | 0.319 | 0.273 | 0.318 | +0.045 | +16.5% | 1.814 | 1.837 | +1.3% |
| cnn_medium | 0.249 | 0.393 | 0.395 | 0.346 | 0.394 | +0.048 | +14.0% | 2.079 | 2.102 | +1.1% |
| cnn_large | 0.315 | 0.473 | 0.477 | 0.422 | 0.475 | +0.053 | +12.6% | 2.324 | 2.346 | +0.9% |
| cnn_deep_3 | 0.230 | 0.392 | 0.391 | 0.338 | 0.392 | +0.054 | +16.0% | 1.858 | 1.882 | +1.2% |
| mlp_tiny | 0.167 | 0.317 | 0.314 | 0.266 | 0.315 | +0.049 | +18.5% | 1.667 | 1.684 | +1.0% |
| mlp_tiny_plus | 0.176 | 0.312 | 0.316 | 0.268 | 0.314 | +0.046 | +17.2% | 1.708 | 1.728 | +1.2% |
| mlp_small | 0.206 | 0.371 | 0.369 | 0.316 | 0.370 | +0.055 | +17.3% | 1.749 | 1.772 | +1.3% |
| mlp_small_plus | 0.260 | 0.439 | 0.435 | 0.378 | 0.437 | +0.059 | +15.6% | 1.861 | 1.884 | +1.2% |
| mlp_medium | 0.265 | 0.428 | 0.431 | 0.375 | 0.429 | +0.055 | +14.6% | 2.013 | 2.040 | +1.3% |
| mlp_large | 0.376 | 0.555 | 0.549 | 0.493 | 0.552 | +0.059 | +11.9% | 2.483 | 2.516 | +1.4% |
| mlp_original | 0.302 | 0.465 | 0.461 | 0.409 | 0.463 | +0.054 | +13.1% | 2.221 | 2.250 | +1.3% |
| mlp_tfc | 0.186 | 0.333 | 0.336 | 0.285 | 0.334 | +0.050 | +17.4% | 1.728 | 1.751 | +1.3% |
| resnet8 | 0.418 | 0.555 | 0.552 | 0.508 | 0.553 | +0.045 | +8.8% | 3.127 | 3.156 | +1.0% |

**run 1 = lowest dynamic of 3: 14/14.** DPU Δdyn range +8.8% … +18.5%; ΔE ≤ +1.4%.

## FINN — representative set (MEASURED; 31/31 of all scanned are run-1-low)
Canonical = MLP/CNN size sweeps (pair with the DPU models) + the CNN fps sweeps.
run length ≈ 10000/fps: slow MLPs ~6–55 s, the fps-sweep CNNs ~0.9–1.1 s.

| build (≈ DPU pair) | fps | dyn all-3 | dyn 2-3 | Δdyn% | E all-3 | E 2-3 | ΔE% |
|--------------------|----:|----------:|--------:|------:|--------:|------:|----:|
| cnn-8x16 int8 (cnn_tiny) | 454 | 0.177 | 0.185 | +4.7% | 7.591 | 7.609 | +0.2% |
| cnn-16x32 int8 | 308 | 0.195 | 0.198 | +1.4% | 11.578 | 11.587 | +0.1% |
| cnn-16x32x64 int8 | 291 | 0.207 | 0.208 | +0.6% | 12.644 | 12.647 | +0.0% |
| mlp-64x32 int8 (mlp_tiny) | 1576 | 0.158 | 0.187 | +18.3% | 2.209 | 2.228 | +0.8% |
| mlp-96x48 int8 | 1043 | 0.180 | 0.192 | +6.3% | 3.339 | 3.350 | +0.3% |
| mlp-128x64 int8 | 774 | 0.193 | 0.203 | +4.9% | 4.524 | 4.537 | +0.3% |
| mlp-192x96 int8 | 501 | 0.233 | 0.240 | +2.8% | 7.075 | 7.087 | +0.2% |
| mlp-256x128 int8 | 363 | 0.263 | 0.269 | +2.1% | 9.858 | 9.873 | +0.1% |
| mlp-256x256x128 int8 | 294 | 0.263 | 0.268 | +1.9% | 12.368 | 12.384 | +0.1% |
| mlp-512x256 int8 | 182 | 0.294 | 0.295 | +0.5% | 19.917 | 19.924 | +0.0% |
| tfc int8 (mlp_tfc) | 1434 | 0.172 | 0.174 | +1.4% | 2.465 | 2.467 | +0.1% |
| cnn-8x16 int8 QI fps10k | 10740 | 0.354 | 0.470 | +32.6% | 0.344 | 0.355 | +3.1% |
| cnn-8x16 int4 QI fps10k | 8934 | 0.174 | 0.221 | +27.1% | 0.390 | 0.395 | +1.4% |
| cnn-8x16 int4 QI fps100k | 8930 | 0.154 | 0.210 | +36.3% | 0.386 | 0.392 | +1.6% |

**run 1 = lowest dynamic of 3: 31/31 FINN builds scanned** (size_sweep + size_sweep_qi
+ headline trio). Δdyn range +0.3% … +36.3%; ΔE ≤ +3.1%.

## Side-by-side summary

| | run-1-low present? | per-run power persisted? | Δdyn (2-3 vs all-3) | ΔE/image |
|---|---|---|---|---|
| **DPU** (14) | **yes, 14/14** | yes (`fnb58_power`, 3 runs) | +8.8% … +18.5% | +0.9% … +1.4% |
| **FINN** (31) | **yes, 31/31** | yes (identical schema) | +0.3% … +36.3% | +0.0% … +3.1% |

**Why the magnitude differs (and why it's still symmetric):** the warmup is a
*fixed-duration* board/meter settling transient at the start of run 1 (idle→active);
its **fractional** size therefore scales with how short the run is. Run length, not
framework, sets the magnitude — e.g. DPU's ~3.6–6.4 s runs land at +9–18%; FINN's
slowest MLPs (~55 s) wash it out (+0.5%); FINN's ~1 s fps-sweep CNNs are dominated
by it (+27–36%). A DPU and a FINN build at the same throughput would warm up the
same. (INFERRED mechanism; the run-1-low *direction* is MEASURED, 45/45.)

## Asymmetry verdict
- **Is "discard run 1" fair / symmetric?** **Yes (MEASURED).** Both frameworks warm
  up the same way — run 1 is the strict minimum in 45/45 cases, and run 2 ≈ run 3 on
  both. It is not a one-sided artifact, so dropping run 1 applies identically to both
  and does not bias the FINN-vs-DPU comparison.
- **Is averaging all-3 the unbiased choice instead?** It is *procedurally* symmetric,
  but **not result-neutral**: because the warmup's fractional size is
  throughput-dependent, all-3 averaging pushes dynamic power down *more* for faster
  builds (most for the ~1 s FINN fps sweeps, moderately for the DPU's short runs,
  barely for slow FINN MLPs). So all-3 introduces a throughput-correlated downward
  bias in *dynamic power* specifically.

## Recommendation (INFERRED — Sam decides; not applied)
Report **steady-state (runs 2–3)**, applied identically to both frameworks, for the
**dynamic-power** numbers. Basis: (1) run 1 is a demonstrable cold-start transient
(strict-min 45/45; run 2 ≈ run 3); (2) all-3 averaging's bias is throughput-dependent,
which distorts exactly the cross-build/cross-framework efficiency comparison the paper
makes. **Caveat:** the choice barely changes **energy/image** (≤1.4% DPU, ≤3.1% FINN,
idle-dominated), so if only energy is reported, all-3 vs steady is nearly moot — the
decision matters for the dynamic-power column. Either way, **whatever is chosen must
be the same policy on both sides** (it is symmetric, so that is achievable). To apply
steady-state, re-derive from `runs[1:]` per model — do **not** trust
`summary.dynamic_power_w` (it is the all-3 mean).
