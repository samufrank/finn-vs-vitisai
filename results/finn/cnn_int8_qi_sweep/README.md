# CNN INT8 QI — target_fps resource/fit sweep

Synth/implementation-only sweep of CNN INT8 **QuantIdentity (QI)** MNIST models
across new `target_fps` points, to map resource utilization and the fit→bust
boundary on the Ultra96 (`xczu3eg`). **No board deployment** — these capture the
post-implementation resource + fit verdict only. Run 2026-06-09.

`summary.md` is the machine-generated table (rewritten by the driver/rebuild).
This README is the human narrative and is **not** regenerated — edit by hand.

## What was built (10 builds)

INT8 QI = `cnn_mnist_<size>_qi.onnx` (QI is baked into the export; no flag). Same
FINN path as the existing tiny QI 1K/10K builds — `compile.py --board Ultra96`,
`synth_clk_period_ns=10.0`, full VIVADO_ZYNQ flow — so these extend, and are
directly comparable to, the prior `results/finn/size_sweep_qi/` data. Only
`--model`/`--fps` vary. Sizes: tiny ~1.4K, small ~5.2K, medium ~20K, deep_3 ~24K
params (FINN's exact count is in each build's `report/op_and_param_counts.json`).

## Results — all MEASURED (post-impl Vivado `Util%`, Tier-1)

Every % is read straight from that build's `top_wrapper_utilization_placed.rpt`
(`Util%` column; device totals are the rpt's own `Available` column). BRAM% is on
the **Block RAM Tile** basis (RAMB36-tile, /216); native RAMB36/RAMB18 counts are
in each `report/post_synth_resources.json`.

| size | fps | verdict | LUT% | BRAM% | DSP% | CARRY8% | binding | elapsed |
|---|---:|---|---:|---:|---:|---:|---|---:|
| tiny | 200 | FIT | 29.5 | 5.3 | 0.6 | 13.7 | LUT | 1557s |
| tiny | 500 | FIT | 29.6 | 5.6 | 0.8 | 13.8 | LUT | 1544s |
| tiny | 3000 | FIT | 35.6 | 6.2 | 3.1 | 19.6 | LUT | 1642s |
| tiny | 5000 | FIT | 39.6 | 3.9 | 5.3 | 31.2 | LUT | 1662s |
| small | 200 | FIT | 34.1 | 9.5 | 0.8 | 14.2 | LUT | 1533s |
| small | 500 | FIT | 34.0 | 6.5 | 1.9 | 14.2 | LUT | 1529s |
| small | 3000 | FIT | 64.5 | 5.3 | 11.1 | 54.9 | LUT | 2305s |
| medium | 500 | FIT | 87.9 | 14.8 | 9.7 | 55.5 | LUT | 2709s |
| deep_3 | 500 | FIT | 42.4 | **68.5** | 3.6 | 20.4 | **BRAM** | 1662s |
| deep_3 | 1000 | FIT | 87.6 | 13.2 | 9.7 | 55.1 | LUT | 2293s |

**All 10 fit.** No busts at any requested point.

## Findings

**deep_3 changes which resource binds with target_fps — the headline result.**
It is the only model here that is not LUT-bound at low fps:
- **fps500: BRAM-bound** — 148/216 BRAM tiles (68.5%) vs LUT 29,894/70,560 (42.4%).
  Native: `BRAM_36K=17, BRAM_18K=262`.
- **fps1000: flips to LUT-bound** — LUT 61,831/70,560 (87.6%), BRAM collapses to
  28.5 tiles (13.2%). Native: `BRAM_36K=26, BRAM_18K=5`.

Between 500 and 1000, ~120 BRAM tiles of weight storage move out of block RAM
(`BRAM_18K` 262→5) and ~32k LUTs appear — the FINN signature of higher folding
pushing MVAU weights from `ram_style=block` into distributed LUT-RAM. The two
builds' `report/` folding configs (`estimate_layer_resources.json`,
`auto_folding_config.json` if present) localize which layer flips. Every other
point in the sweep is LUT-bound.

**fit→bust crossings pinned (anchors from prior `size_sweep_qi` builds):**
- **tiny:** all four fit, 29.5→39.6% LUT, slotting between the existing 1K (29.6%)
  and 10K (56%) anchors. LUT-bound throughout.
- **small:** fps3000 fits at 64.5% LUT (was expected to maybe bust) → crossing is
  between **3000 (fit) and 10000 (existing 173% bust)**.
- **medium:** fps500 fits at 87.9% LUT → crossing between **500 (fit) and 1000
  (existing 108.8% bust)**; 500 is the last fit before the wall.

## deep_3 fps1000 — transient failure on first attempt (resolved)

The first attempt returned INCOMPLETE after 136s — **not a resource bust**. A DNS
outage during the FINN container's boot-time `pip install` left `unfoldNd`
unfetchable (`Temporary failure in name resolution`), so `brevitas_examples`
failed to import and `compile.py` crashed before any FINN step ran (`last_step=
None`). Re-run after network recovery (`finn/rerun_deep3_fps1000.py`, with a retry
guard) → FIT, as tabled above. The failed attempt's raw FINN log is preserved at
`cnn_int8_deep_3_qi_fps1000/FAILED_attempt_finn_build.log` (the DNS/import error in
full). `progress.txt` also keeps a stale `deep_3_qi_fps1000 … INCOMPLETE` line above
the final FIT line — the append-only log; `summary.md`/this README are authoritative.

## File layout (per build, under `<label>/`)

- `impl_runme.log` — Vivado place_design log (carries the DRC bust verdict on a bust)
- `top_wrapper_utilization_placed.rpt` — post-place utilization (LUT/FF/BRAM/DSP/CARRY8 actuals + %)
- `top_wrapper_clock_utilization_routed.rpt` — post-route clock utilization
- `report/` — 9 FINN report files incl. `post_synth_resources.json` (Tier-1
  native-unit resources), `estimate_*.json`, `op_and_param_counts.json`,
  `post_route_timing.rpt`
- `finn_build.log` — FINN flow log · `capture_console.log` — capture_build.py stdout
- `verdict.txt` — FIT/BUST, the four %s, binding resource, elapsed, last FINN step

Top level: `summary.md` (auto table) · `resource_summary.csv` (machine-readable,
native units) · this `README.md` · `driver.log` · `progress.txt` · `SWEEP_DONE` ·
`RERUN_DONE`.

## Re-extracting the numbers (don't trust the prose)

`README.md` and `summary.md` are **derived**. Every number is independently
recoverable from the per-build Tier-1 files, and `analysis/extract_cnn_int8_qi_sweep.py`
re-derives the whole table from them — re-run it anytime to regenerate
`resource_summary.csv`:
- native counts: `<label>/report/post_synth_resources.json` `(top)` (LUT, FF,
  BRAM_36K, BRAM_18K, DSP)
- percentages + CARRY8: `<label>/top_wrapper_utilization_placed.rpt` (`Util%`)
- full Vivado place log / bust DRC verdict: `<label>/impl_runme.log`
- FINN flow log, estimates, timing: `<label>/finn_build.log`, `<label>/report/*.json`
The extractor reads only those primary files (it does not parse this README).

## Reproduce

Built by `finn/run_cnn_int8_qi_sweep.py`, which drives `finn/capture_build.py` once
per build. Isolation: this batch used `--finn-build-dir /tmp/finn_qi_int8_sweep`
(an option added to `capture_build.py`, default unchanged) so it ran safely
alongside a concurrent ResNet-8 build in the default `/tmp/finn_dev_samu` — the two
cannot share/clean each other's scratch or cross-capture each other's runme.log.
The driver copies the FINN `report/` JSONs out of the **gitignored**
`finn/size_sweep_runs/<label>/` into this committed path per build, immediately on
completion. To re-run one point: `python3 finn/rerun_deep3_fps1000.py` (pattern).
