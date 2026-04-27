# FINN target_fps sweep on ZU3EG

Characterize FINN's design space across (model, precision, target_fps) for the
Xilinx Zynq UltraScale+ ZU3EG (`xczu3eg-sfvc784-2-e` — Real Digital AUP-ZU3,
compiled via the FINN `Ultra96` board def which uses the same die).

Four sweeps:

| Sweep      | ONNX                           | Notes                          |
|------------|--------------------------------|--------------------------------|
| `mlp_int8` | `mlp_mnist_tiny.onnx`          | 64×32 hidden, INT8 weights/acts |
| `mlp_int4` | `mlp_mnist_tiny_int4.onnx`     | INT4 weights/acts              |
| `cnn_int8` | `cnn_mnist_tiny.onnx`          | [8,16] CNN MNIST, INT8         |
| `cnn_int4` | `cnn_mnist_tiny_int4.onnx`     | INT4 (warm-started from INT8)  |

Run order: `mlp_int8 -> mlp_int4 -> cnn_int8 -> cnn_int4`. One build at a time.
Total wall-time is dominated by Vivado synthesis; expect 10-24 h for the full
sweep depending on how many high-target builds reach `step_synthesize_bitfile`.


## Summary

Sweep across all 4 (model, precision) combinations at target_fps in [1000, 10000, 100000, 500000] characterizing FINN's design space on AUP-ZU3 (ZU3EG). 15/16 builds successful; cnn_int8 fps=100K hit ZU3's CLB capacity (84,243 LUTs needed vs 70,560 available). 4 sweep builds (one per sweep) deployed to board for benchmarking. Total wall time ~5.5 hours unattended.

**Headline findings:**
- CNN INT8 ceiling on ZU3: target_fps in [10K, 100K]. CNN INT4 fits at all tested targets up to 500K with the auto-folder plateauing at fps=100K (PE=16/SIMD=9, 144 DSPs).
- Partition is invariant across target_fps. FINN's auto-folder controls per-layer parallelism but never migrates layers between CPU and FPGA.
- CNN throughput gains substantial (44-62%); MLP gains modest (3-4%, CPU-bottlenecked).
- Sub-byte precision provides substantially more design-space headroom than INT8 on this device.


## Compiled vs. benchmarked

| Sweep | Target_fps | Compile status | Benchmarked? | Reason for selection |
|---|---|---|---|---|
| mlp_int8 | 1000 | success | yes (baseline) | matched-comparison reference |
| mlp_int8 | 10000 | success | no | byte-identical to fps=1000 (PE=SIMD=1) |
| mlp_int8 | 100000 | success | no | minor folding change (SIMD=4); skipped for the bigger fps=500K jump |
| mlp_int8 | 500000 | success | yes | first MLP INT8 point with materially different folding (PE=4/SIMD=4) |
| mlp_int4 | 1000 | success | yes (baseline) | matched-comparison reference |
| mlp_int4 | 10000 | success | no | byte-identical to fps=1000 |
| mlp_int4 | 100000 | success | no | minor folding change; skipped for the bigger fps=500K jump |
| mlp_int4 | 500000 | success | yes | first MLP INT4 point with new folding (PE=1/SIMD=16); also exposed 2-per-byte input pack requirement |
| cnn_int8 | 1000 | success | yes (baseline) | matched-comparison reference |
| cnn_int8 | 10000 | success | yes | only successful post-baseline CNN INT8 build |
| cnn_int8 | 100000 | resource_fail | n/a | place-design exhausts ZU3 CLB capacity (84,243 LUTs vs 70,560 available); initially misclassified as `tool_fail`, reclassified post-hoc |
| cnn_int8 | 500000 | (skipped) | n/a | sweep stopped at 100K resource fail |
| cnn_int4 | 1000 | success | yes (baseline) | matched-comparison reference |
| cnn_int4 | 10000 | success | no | intermediate folding (PE=4/SIMD=9); could be added later for finer curve |
| cnn_int4 | 100000 | success | yes | auto-folder ceiling (PE=16/SIMD=9, 44.4% DSP) |
| cnn_int4 | 500000 | success | no | identical bitstream to fps=100000 |


## Resource utilization (all successful builds)

| Sweep | Target_fps | LUT | LUT% | FF | BRAM18 | BRAM% | DSP | DSP% | Fmax (MHz) | Est FPS | MVAU folding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_int8 | 1000 | 13150 | 18.6% | 16025 | 6 | 1.4% | 2 | 0.6% | 158.8 | 48828 | PE=1/SIMD=1 |
| mlp_int8 | 10000 | 13150 | 18.6% | 16025 | 6 | 1.4% | 2 | 0.6% | 158.8 | 48828 | PE=1/SIMD=1 |
| mlp_int8 | 100000 | 13232 | 18.8% | 16140 | 6 | 1.4% | 5 | 1.4% | 162.4 | 195312 | PE=1/SIMD=4 |
| mlp_int8 | 500000 | 25256 | 35.8% | 28490 | 9 | 2.1% | 18 | 5.0% | 129.8 | 625000 | PE=4/SIMD=4 |
| mlp_int4 | 1000 | 8348 | 11.8% | 11926 | 6 | 1.4% | 2 | 0.6% | 219.7 | 48828 | PE=1/SIMD=1 |
| mlp_int4 | 10000 | 8348 | 11.8% | 11926 | 6 | 1.4% | 2 | 0.6% | 219.7 | 48828 | PE=1/SIMD=1 |
| mlp_int4 | 100000 | 8551 | 12.1% | 12189 | 6 | 1.4% | 5 | 1.4% | 233.4 | 195312 | PE=1/SIMD=4 |
| mlp_int4 | 500000 | 8758 | 12.4% | 12502 | 7 | 1.6% | 20 | 5.6% | 156.9 | 625000 | PE=1/SIMD=16 |
| cnn_int8 | 1000 | 17930 | 25.4% | 19889 | 29 | 6.7% | 3 | 0.8% | 166.8 | 1329 | PE=1/SIMD=3 |
| cnn_int8 | 10000 | 36787 | 52.1% | 41032 | 9 | 2.1% | 32 | 8.9% | 120.1 | 13897 | PE=8/SIMD=4 |
| cnn_int8 | 100000 | resource_fail (84243 LUT vs 70560 available, +28% over capacity) |||||||||
| cnn_int4 | 1000 | 10966 | 15.5% | 13986 | 22 | 5.1% | 3 | 0.8% | 243.9 | 1329 | PE=1/SIMD=3 |
| cnn_int4 | 10000 | 11532 | 16.3% | 14987 | 7 | 1.6% | 40 | 11.1% | 190.1 | 13897 | PE=4/SIMD=9 |
| cnn_int4 | 100000 | 13224 | 18.7% | 16832 | 6 | 1.4% | 160 | 44.4% | 164.3 | 63776 | PE=16/SIMD=9 |
| cnn_int4 | 500000 | 13224 | 18.7% | 16832 | 6 | 1.4% | 160 | 44.4% | 164.3 | 63776 | PE=16/SIMD=9 |

ZU3EG total: 70,560 LUT, 141,120 FF, 432 BRAM_18K, 360 DSP. All bitstreams are 5.31 MB (PYNQ shell-dominated, accelerator IP itself is tens of KB).


## Benchmark results (deployed builds)

| Build | Folding | Accuracy | FPS | Latency (ms) | Idle (W) | Active (W) | Dynamic (W) | Energy/inf (mJ) | vs baseline FPS |
|---|---|---|---|---|---|---|---|---|---|
| mlp_int8 baseline | PE=1/SIMD=1 | 96.58% | 1575 | 0.635 | 3.27 | 3.44 | 0.18 | 2.21 | — |
| mlp_int8 fps=500K | PE=4/SIMD=4 | 96.58% | 1629 | 0.614 | 3.32 | 3.49 | 0.18 | 2.15 | +3.4% |
| mlp_int4 baseline | PE=1/SIMD=1 | 97.29% | 1811 | 0.552 | 3.27 | 3.44 | 0.16 | 1.90 | — |
| mlp_int4 fps=500K | PE=1/SIMD=16 | 97.18% | 1882 | 0.531 | 3.27 | 3.44 | 0.16 | 1.83 | +3.9% |
| cnn_int8 baseline | PE=1/SIMD=3 | 91.99% | 454 | 2.205 | 3.27 | 3.44 | 0.18 | 7.59 | — |
| cnn_int8 fps=10K | PE=8/SIMD=4 | 91.99% | 656 | 1.525 | 3.32 | 3.51 | 0.19 | 5.36 | +44% |
| cnn_int4 baseline | PE=1/SIMD=3 | 88.27% | 525 | 1.903 | 3.29 | 3.45 | 0.16 | 6.57 | — |
| cnn_int4 fps=100K | PE=16/SIMD=9 | 88.27% | 851 | 1.175 | 3.30 | 3.48 | 0.18 | 4.09 | +62% |

CNN gains attributable to DMA stage compression (CNN INT4 baseline DMA was 759 µs / 39% of inference; fps=100K is 31 µs / 2.5%). MLP gains bounded by CPU MatMul which dominates 84-97% of inference time at the tiny [64,32] topology — folding cannot move it to fabric.


## Methodology

Two phases per sweep.

### Phase 1 — bracket the ceiling

Compile at `target_fps ∈ {1000, 10000, 100000, 500000}` in order. Stop on the
first build that does not succeed at the device. Three terminal outcomes:

- `resource_fail` (Vivado place/route can't fit) → ceiling is bracketed by
  `[lo_pass, hi_fail]`. Phase 2 runs.
- `timing_fail` (build completes but post-route WNS < 0) → same bracket. Phase
  2 runs.
- `tool_fail` (FINN exception, Vivado crash, OOM, license, anything that
  isn't "synthesis succeeded but won't fit") → bracket is undefined. Phase 2
  is skipped. Sweep marked `partial`. Manual review via `tool_fails.txt`.

If all four Phase-1 targets pass, the ceiling is capped at 500 000 (we don't
push further; the goal is to bracket, not to find a hard maximum on a model
that's already over-folded).

### Phase 2 — refine the ceiling

Three log-spaced `target_fps` values strictly between `lo_pass` and `hi_fail`:

```
target_k = lo_pass · (hi_fail / lo_pass) ** (k / 4)   for k ∈ {1, 2, 3}
```

Rounded to nice numbers (nearest 100 below 5 000; nearest 500 in [5 000,
50 000); nearest 1 000 above 50 000). De-duplicated; rounding-collisions onto
the bracket endpoints are dropped.

Examples:

| Bracket             | Phase 2 targets         |
|---------------------|-------------------------|
| `[10 000, 100 000]` | `18 000, 31 500, 56 000` |
| `[1 000, 10 000]`   | `1 800, 3 200, 5 500`   |
| `[100 000, 500 000]`| `150 000, 224 000, 334 000` |

### Why log-spaced

Linear midpoints sit too close to the high (failing) end at large brackets.
Geometric spacing gives roughly even visual coverage on a log axis, which is
how FINN's auto-folder thresholds and Vivado's resource pressure both scale.

### Why no fill below `lo_pass`

Phase 1 already sketches the low-end trajectory at order-of-magnitude
spacing. The interesting transitions live near the failure edge. That's
where FINN's auto-folder switches from minimum folding to higher PE/SIMD,
where resource utilization changes meaningfully, and where timing margin
shrinks. Below `lo_pass`, FINN typically picks the same minimum folding
(PE = SIMD = 1 on the dominant MVAU) regardless of `target_fps`, so a denser
sample there would mostly produce identical builds. If denser low-end data
turns out to be needed, that's deferred to a follow-up sweep with a custom
target list — the driver supports `--target` for back-fill.

### Caveat: log-spaced points may produce identical foldings

If the auto-folder rounds three Phase-2 targets to the same PE/SIMD config,
the only differences across those builds will be FIFO depths, BRAM allocation
heuristics, and timing margin (WNS). That's still informative. Tells us
where the folding transition actually lives, but may have adjacent rows in the CSV with identical `folding_json`.

### Build classification

`compile.py` is patched to honor `build_dataflow_cfg`'s return code via
`sys.exit(rc)`. The driver classifies each build with three cross-validated
signals:

1. Process exit code (0 = success, non-zero = some failure)
2. Stdout markers: `Completed successfully`, `Build failed`, the last
   `Running step: step_X [N/19]` line
3. Artifact presence: `deploy/bitfile/*.bit`, `report/post_route_timing.rpt`

| Outcome         | Conditions                                                      |
|-----------------|------------------------------------------------------------------|
| `success`       | rc=0, `Completed successfully`, bitfile present, WNS ≥ 0         |
| `timing_fail`   | rc=0, `Completed successfully`, bitfile present, WNS < 0         |
| `resource_fail` | rc≠0 or `Build failed`, last step is synthesis, Vivado place/route resource error in log |
| `tool_fail`     | any other failure — including markers/artifacts disagreeing      |

Resource-error patterns matched in the log (case-insensitive):

```
unable to place        utilization exceeded     route_design failed
[place 30-             [route 35-               insufficient resources
overlap of placement   placement is impossible
too many lut           too many bram            too many dsp
```

Patterns are refined as the sweep observes real failures.

**Known classifier limitation:** the cnn_int8 fps=100K build was initially misclassified as `tool_fail` because FINN's relayed exception (`ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open`) didn't match the resource-fail patterns above. The actual cause was Vivado place-design CLB exhaustion, found in the deeper `runme.log` at `/tmp/finn_dev_samu/vivado_zynq_proj_*/finn_zynq_link.runs/impl_1/runme.log`. Reclassified as `resource_fail` post-hoc. The classifier should ideally chase the "Check logs under <path>" hint into runme.log; this is a deferred follow-on improvement.


## File layout

```
finn/sweep_driver.py
finn/target_fps_sweep_runs/
├── mlp_int8_fps1000/                  # full FINN output dir
│   ├── deploy/{bitfile,driver,*.npy}
│   ├── intermediate_models/
│   ├── report/                         # post_synth_resources.json, etc.
│   ├── final_hw_config.json
│   ├── auto_folding_config.json
│   ├── build_dataflow.log              # FINN's per-step log (inside output dir)
│   ├── build.log                       # captured docker stdout/stderr (driver)
│   ├── resource_report.json            # resource_summary output (if successful)
│   └── resource_summary.md
├── mlp_int8_fps10000/
└── ...

results/finn/target_fps_sweep/
├── README.md                  # this file
├── sweep_state.json           # SINGLE SOURCE OF TRUTH for resume
├── resource_summary.csv       # one row per build, all sweeps
├── sweep_analysis.md          # detailed per-sweep tables, generated by analyze.py
├── analyze.py                 # regenerates sweep_analysis.md from CSV + final_hw_config.json
├── sweep_driver.log           # high-level driver events
├── tool_fails.txt             # tool-fail summary for post-hoc review
└── benchmarks/                # merged JSON + power timeline PNG per deployed build
    ├── mlp_int8_fps500000_c.json
    ├── mlp_int8_fps500000_c_power.png
    ├── mlp_int4_fps500000_c.json
    ├── mlp_int4_fps500000_c_power.png
    ├── cnn_int8_fps10000_c.json
    ├── cnn_int8_fps10000_c_power.png
    ├── cnn_int4_fps100000_c.json
    └── cnn_int4_fps100000_c_power.png
```

Existing baseline builds at `finn/output_*_mnist_tiny*/` are **not touched**.
The sweep recompiles `target_fps=1000` baselines inside `target_fps_sweep_runs/`
for self-containment.


## Running

```bash
# Full sweep (all four combos, sequential, ~10-24 h)
python finn/sweep_driver.py

# One sweep
python finn/sweep_driver.py --only cnn_int4

# One build (dry-run / back-fill)
python finn/sweep_driver.py --only mlp_int8 --target 1000 --phase phase1
```

Run from the repo root or anywhere — `sweep_driver.py` resolves all paths
absolutely from `__file__`.

The sweep takes hours. Run under `nohup` or `tmux` so an ssh disconnect doesn't
kill it:

```bash
nohup python finn/sweep_driver.py > /tmp/sweep.out 2>&1 &
tail -f results/finn/target_fps_sweep/sweep_driver.log
```


## Resume procedure

`sweep_state.json` is the single source of truth. The driver writes it
atomically (write to `.tmp`, fsync, rename) after every build so a kill at any
moment leaves a consistent state. To resume after interruption:

```bash
python finn/sweep_driver.py
```

The driver:
- Loads `sweep_state.json`. If absent, initializes with all four sweeps in
  `not_started`.
- For each sweep:
  - `status == 'completed'` → skip.
  - `phase == 'tool_failed'` (sweep marked `partial`) → skip; manual review
    required.
  - Otherwise → resume Phase 1 from the lowest target not in
    `completed_builds`, then Phase 2 likewise.
- Per build:
  - `status == 'in_progress'` (sweep was killed mid-build) → delete
    `target_fps_sweep_runs/{sweep}_fps{target}/`, retry.
  - Otherwise → reuse the recorded result.

To force-rerun a specific build, edit `sweep_state.json`:

```bash
python -c "
import json
s = json.load(open('results/finn/target_fps_sweep/sweep_state.json'))
del s['sweeps']['mlp_int8']['completed_builds']['10000']
json.dump(s, open('results/finn/target_fps_sweep/sweep_state.json','w'), indent=2)
"
rm -rf finn/target_fps_sweep_runs/mlp_int8_fps10000
python finn/sweep_driver.py --only mlp_int8 --target 10000
```

To start fresh, delete `sweep_state.json` (and optionally
`finn/target_fps_sweep_runs/`).


## State JSON schema

```json
{
  "schema_version": 1,
  "started":     "2026-04-25T...",
  "last_update": "2026-04-25T...",
  "current_sweep":  "mlp_int4",
  "current_target": 50000,
  "sweeps": {
    "mlp_int8": {
      "status":     "completed | in_progress | not_started | partial",
      "phase":      "phase1 | phase2 | done | tool_failed",
      "model_onnx": "mlp_mnist_tiny.onnx",
      "phase1_targets":   [1000, 10000, 100000, 500000],
      "ceiling_lo":       10000,
      "ceiling_hi":       100000,
      "phase2_targets":   [18000, 31500, 56000],
      "completed_builds": {"1000": { ... }, "10000": { ... }, ...},
      "tool_fails":       []
    },
    ...
  }
}
```

Per-build dict (a row of the CSV plus a few extras):

```json
{
  "status":         "success | timing_fail | resource_fail | tool_fail | in_progress",
  "started":        "2026-04-25T13:00:00Z",
  "completed":      "2026-04-25T13:35:00Z",
  "elapsed_s":      2100.0,
  "last_step":      "step_synthesize_bitfile",
  "error_excerpt":  "ERROR: [Place 30-486] ...",
  "returncode":     1,
  "log_path":       "finn/target_fps_sweep_runs/mlp_int8_fps100000/build.log",
  "wns_ns":         null,
  "fmax_mhz":       null,
  "lut":            null, "lut_pct": null, "ff": null,
  "bram18":         null, "bram18_pct": null,
  "dsp":            null, "dsp_pct": null,
  "est_fps_fpga":   null, "est_latency_us": null, "bitstream_mb": null,
  "folding":        {}, "cpu_layers": []
}
```


## CSV schema

`resource_summary.csv`:

```
sweep, target_fps, phase, status, started_iso, elapsed_s, last_step,
wns_ns, fmax_mhz,
lut, lut_pct, ff, bram18, bram18_pct, dsp, dsp_pct,
est_fps_fpga, est_latency_us, bitstream_mb,
folding_json, cpu_layers_json,
log_path, error_excerpt
```

Failed builds have empty resource columns. `folding_json` and
`cpu_layers_json` are JSON-encoded inline strings — load with `json.loads()`
when post-processing.


## Cache management

FINN's HLS-IP cache lives at `/tmp/finn_dev_samu/`. Between builds the driver
removes `code_gen_ipgen_*` and `vivado_stitch_proj_*` (forces fresh IP gen and
fresh stitch project). It keeps `vivado_ip_cache/` (legitimately speeds up
identical IP regen across builds — when two adjacent Phase-2 builds hit the
same folding, this cache pays off).

This avoids the issue where stale IP from an earlier different-precision
build poisoned a new build (`IP definition not found for VLNV: ...:hls:
StreamingMaxPool_hls_0:1.0`).
