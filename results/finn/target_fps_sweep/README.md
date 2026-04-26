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
├── partitioning_changes.md    # findings (filled in post-sweep)
├── sweep_driver.log           # high-level driver events
└── tool_fails.txt             # tool-fail summary for post-hoc review
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
