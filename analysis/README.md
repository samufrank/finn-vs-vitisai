# Analysis Scripts

Scripts that extract and consolidate results from benchmark JSONs, Vivado reports,
and FINN build outputs. Run from the finn-vs-vitisai repo root.

    python3 analysis/extract_results.py     # primary 3-framework comparison
    python3 analysis/extract_resources.py   # FPGA resource utilization
    python3 analysis/extract_sweeps.py      # FINN design space (target_fps + model size)

Generated outputs are committed for reference but can be regenerated at any time.
Re-run after adding new benchmark results or build outputs.

`extract_sweeps.py` writes four files:
  - `analysis/finn_sweep_summary.md`              cross-sweep wide view (compile times,
  - `analysis/finn_sweep_summary.csv`              resources, benchmarks; one row/build)
  - `results/finn/size_sweep/sweep_analysis.md`        per-sweep deep dive: status,
  - `results/finn/target_fps_sweep/sweep_analysis.md`  cross-precision tables, partition
                                                       shifts, Vivado failure forensics

The two `sweep_analysis.md` files used to be written by per-sweep `analyze.py`
scripts; that logic is now consolidated here so a single invocation regenerates
everything FINN-related.

Coverage as of 2026-04-29:
- 12 C-runner benchmarks across FINN, VTA, Vitis AI (MLP, CNN, Transformer)
- 9 resource utilization entries (VTA INT8 from archived report, rest from disk)
- 44 FINN builds (default + target_fps sweep + model size sweep)
- Model size sweep benchmarks in progress
- Non-custom models (ResNet-18, FC autoencoder) not yet included
