# DPU canonical results — 2026-06-21 (the summary substrate)

14-model DPU sweep, B512 @ AUP-ZU3, INT8, 3 runs × 10,000 images, FNB58 power.
Every value MEASURED from `merged/<model>_…_20260621_*.json`. Field path per column:

| column | source field |
|--------|--------------|
| accuracy (%) | `summary.accuracy` |
| throughput_fps_mean ± std | `summary.throughput_fps_mean` ± `summary.throughput_fps_std` |
| ss_allin_ms | `config.single_shot_latency_allin_ms` (layout+engine+argmax, median of K=200) |
| ss_engine_ms | `config.single_shot_latency_engine_ms` (execute_async+wait only, median of K=200) |
| latency_ms (1/fps, legacy) | `summary.latency_ms_mean` (= 1000/throughput; NOT single-shot) |
| dynamic_W | `summary.dynamic_power_w` (mean of `runs[].fnb58_power.dynamic_power_w` over **all 3 runs**) |
| energy_mj | `summary.energy_per_image_mj_mean` |

| model | acc | fps_mean ± std | ss_allin_ms | ss_engine_ms | latency_ms (1/fps) | dynamic_W | energy_mj |
|-------|----:|----------------:|------------:|-------------:|-------------------:|----------:|----------:|
| cnn_tiny | 78.7 | 2659 ± 5.8 | 0.329 | 0.224 | 0.376 | 0.248 | 1.760 |
| cnn_small | 94.4 | 2588 ± 18.4 | 0.342 | 0.239 | 0.386 | 0.273 | 1.814 |
| cnn_medium | 96.5 | 2293 ± 3.2 | 0.389 | 0.285 | 0.436 | 0.346 | 2.079 |
| cnn_large | 99.2 | 2082 ± 4.8 | 0.437 | 0.333 | 0.480 | 0.422 | 2.324 |
| cnn_deep_3 | 98.7 | 2539 ± 9.4 | 0.355 | 0.251 | 0.394 | 0.338 | 1.858 |
| mlp_tiny | 97.4 | 2807 ± 8.8 | 0.309 | 0.222 | 0.356 | 0.266 | 1.667 |
| mlp_tiny_plus | 97.7 | 2739 ± 13.8 | 0.318 | 0.232 | 0.365 | 0.268 | 1.708 |
| mlp_small | 97.6 | 2684 ± 8.9 | 0.333 | 0.245 | 0.373 | 0.316 | 1.749 |
| mlp_small_plus | 97.6 | 2557 ± 2.5 | 0.351 | 0.264 | 0.391 | 0.378 | 1.861 |
| mlp_medium | 98.0 | 2381 ± 6.1 | 0.373 | 0.285 | 0.420 | 0.375 | 2.013 |
| mlp_large | 97.8 | 1978 ± 4.7 | 0.457 | 0.369 | 0.506 | 0.493 | 2.483 |
| mlp_original | 97.8 | 2158 ± 10.3 | 0.420 | 0.333 | 0.463 | 0.409 | 2.221 |
| mlp_tfc | 97.4 | 2702 ± 18.6 | 0.330 | 0.242 | 0.370 | 0.285 | 1.728 |
| resnet8_cifar10 | 86.3 | 1565 ± 1.0 | 0.605 | 0.443 | 0.639 | 0.508 | 3.127 |

## Notes
- **`dynamic_W` here is the all-3-run mean**, which includes a low cold-start run 1.
  Run-1 is the lowest dynamic power in **14/14** models; the steady-state (runs 2–3)
  dynamic power is **+9% to +18% higher**. See `STEADY_STATE_COMPARISON.md` before
  choosing which basis the paper reports. The same warmup is present on FINN.
- **`latency_ms` = 1/throughput** (batch-amortized), a legacy field — NOT a real
  single-shot latency. Use `ss_allin_ms` (image→prediction) for single-shot and
  `ss_engine_ms` for the bare VART call. `ss_allin` is the DPU↔FINN-matched
  single-shot (FINN reports `single_shot_latency_allin_ms` on the same boundary).
- The on-board INA260 path is absent, so `runs[].power_samples = 0` is a **stale
  placeholder**; the real sample count is `runs[].fnb58_power.n_samples`
  (376 for the per-model logger, ~376–2208 depending on run length). All power
  here is the host-side FNB58 merge.
- **resnet8_cifar10 = 86.3%** - this is the corrected run (the old bugged
  number was 25.21%)
- ss_allin > ss_engine on every model by ~0.09–0.16 ms = the Python
  layout/alloc/argmax around the VART call (the per-image overhead targets).
