# Analysis Scripts

Scripts that extract and consolidate results from benchmark JSONs, Vivado reports,
and FINN build outputs. Run from the finn-vs-vitisai repo root.

    python3 analysis/extract_results.py     # primary 3-framework comparison
    python3 analysis/extract_resources.py   # FPGA resource utilization
    python3 analysis/extract_sweeps.py      # FINN design space (target_fps + model size + QI)

Generated outputs are committed for reference but can be regenerated at any time.
Re-run after adding new benchmark results or build outputs.

## Generated Files

### extract_results.py
- `analysis/verified_results.md` — cross-framework comparison tables (FINN, VTA, DPU)
- `analysis/verified_results.csv` — same data in CSV

### extract_resources.py
- `analysis/resource_utilization.md` — FPGA resource usage across all frameworks and bitstreams

### extract_sweeps.py
Writes four files covering FINN design space exploration:
- `analysis/finn_sweep_summary.md` — wide view across all FINN builds: compile times, resources, benchmarks, failures (one row per build)
- `analysis/finn_sweep_summary.csv` — same data in CSV
- `results/finn/size_sweep/sweep_analysis.md` — per-sweep deep dive for model size sweep
- `results/finn/target_fps_sweep/sweep_analysis.md` — per-sweep deep dive for target_fps sweep

Three sweep sources are covered:
- `default` — baseline FINN builds (tiny models, TFC, autoencoder)
- `target_fps_sweep` — target_fps variation at fixed tiny model (1K-500K)
- `size_sweep` — model size variation at fixed target_fps=1000 (classic partition)
- `size_sweep_qi` — QuantIdentity partition builds (CNN only; MLP QI excluded as counterproductive)

The two `sweep_analysis.md` files were originally written by per-sweep `analyze.py`
scripts; that logic is now consolidated in `extract_sweeps.py` so a single invocation
regenerates everything FINN-related.

## Vivado Reports

`vivado_utilization_reports/` contains raw Vivado utilization and timing reports for
all bitstreams (FINN, VTA, DPU, FINN-T). These are the source data for
`extract_resources.py`.

## Coverage

- 30+ C-runner benchmarks across FINN, VTA, Vitis AI (MLP, CNN, Transformer)
- 9 resource utilization entries (VTA INT8 from archived report, rest from disk)
- 58 FINN builds (default + target_fps sweep + model size sweep + QI sweep)
- QI sweep: 24 builds including CNN tiny INT8 fps=10K (10,740 FPS — beats DPU 3.7×)
- DPU size sweep: 14 models (12 MLP/CNN + TFC + ResNet-8)
- VTA size sweep: 3 models (CNN small INT8/INT4, MLP small INT8)
- CIFAR-10 CNN compiled but not benchmarked (C runner does not support 32×32×3 input)
