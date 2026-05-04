# `board/` — On-board runtime, benchmark orchestration, and deploy artifacts

This directory holds everything that runs **on the AUP-ZU3 (Ultra96-class) board**
or coordinates board-side benchmarks from the host: FINN/VTA/DPU C runners,
Python wrappers, the cross-framework benchmark driver, host-side power
logging, and the playbooks that describe how to stage a session.

**Convention:** the host pushes deploys + sources to `xilinx@192.168.3.1`
(or `petalinux@…` for DPU). The board has matching copies of `.c` runners,
their built `.so` files, and `benchmark.py`. Each session typically scp's
the latest sources, rebuilds the `.so`, then runs `benchmark.py` per deploy.

---

## Documented in this README

The entries below cover what's been touched in tracked work. Other scripts
in this directory exist from parallel investigations (DPU experiments, VTA
debugging) — those are listed in [§ Not yet documented](#not-yet-documented)
for a future session to expand.

### Cross-framework benchmark orchestration

- **`benchmark.py`** — the host-or-board entry point for every benchmark
  in this project (FINN, FINN-T, VTA, DPU). One Python script with
  toolchain-specific code paths gated on `--toolchain`. Outputs a JSON
  per run with throughput, latency, accuracy, idle/active power,
  per-stage breakdowns (when `--finn-runtime c` is used), and the
  config that produced the run.
  - **Key flags:** `--toolchain {finn,finn-t,vta,dpu,vitis_ai}`,
    `--model <deploy_dir>`, `--dataset {mnist,cifar10,radioml2018}`,
    `--runs N`, `--stabilize/--idle <sec>`, `--finn-runtime {python,c}`,
    `--finn-double-buffer`, `--name <run_id>`.
  - **FINN code path:** auto-detects partition (`classic` vs `qi`) from
    the deploy's `cpu_config.json` and wires the runner accordingly.
    Works with QI builds for both CNN and MLP. Mixed-precision QI
    (e.g. CNN INT4 with INT8 input from QuantIdentity) is supported via
    separate `in_precision`/`out_precision` arguments to the C runner.
  - **DPU code path:** uses Vitis AI VART. Per-precision benchmark
    JSONs follow the `<model>_<size>_<dataset>_b1_<timestamp>.json`
    naming convention.
  - **VTA code path:** uses TVM runtime + the VTA overlay at the clock
    encoded by precision (250 MHz INT8 / 200 MHz INT4 / 166 MHz INT4-o8).

### FINN C runners

These produce the `.so` files that `benchmark.py` loads via `ctypes` when
`--finn-runtime c` is set. The C path is the production path; the Python
fallback exists for correctness diffing.

- **`finn_cnn_infer.c`** — CNN runner. Handles classic and QI partitions,
  both INT8 and INT4, with double-buffered DMA. Public API:
  `finn_cnn_runner_init` (now takes `in_precision` + `out_precision` so
  QI INT4 deploys with INT8 input + INT4 output work), `…_set_second_buffers`
  (optional, enables double-buffered batch), `…_infer_batch`,
  `…_infer_one_profiled`, `…_destroy`. State is module-global
  (`g_cnn`); only one model loaded at a time.
- **`finn_mlp_infer.c`** — MLP runner. Mirror of the CNN runner with the
  classic + QI partitions and double-buffered DMA, but only one
  monolithic `precision` argument (the QI partition is treated as a
  CNN-only optimization in this project — see project memory
  `finn_qi_cnn_only.md` — so MLP QI deploys are not in the result set
  and the mixed-precision plumbing isn't wired here).
- **`test_finn_cnn_infer.py`** — host-side correctness harness for
  `libfinn_cnn_infer.so`. Builds the `.so` with `gcc -Werror`, then
  exercises pack/unpack byte-exactness, GAP integer accumulator
  exactness, end-to-end mock inference against MNIST images, and a QI
  partition mock against a real deploy. Run from the repo root:
  ```bash
  python3 board/test_finn_cnn_infer.py --skip-deep3 --pack-trials 5
  ```

### Documentation / playbooks

- **`size_sweep_deploy_playbook.md`** — full step-by-step recipe for the
  size-sweep board sessions: how to bundle deploy tarballs, scp to the
  board, rebuild `.so`s, run benchmarks per (model, size, precision)
  pair, capture power via the FNB58 logger, and pull JSONs back.
- **`benchmark_inventory.md`** — cross-framework inventory of which
  models have been benchmarked at which precisions on which clocks.
  Updated by hand after each session.
- **`recognized_benchmarks_board_playbook.md`** — playbook for the
  "recognized FINN benchmark" set (TFC MLP, etc.).

### Power measurement

- **`fnb58_logger.py`** — host-side USB-attached FNB58 power-meter logger.
  Streams power readings to a CSV with timestamps. Run it on the host
  alongside the board-side benchmark to capture idle/active windows;
  `benchmark.py` then matches its run-window timestamps against the CSV
  to compute `dynamic_power_w` and `energy_per_image_mj`. The udev rule
  `/etc/udev/rules.d/90-fnirsi.rules` (vendor `0x2e3c`) lets it run
  without sudo.

### Local archive (not committed; `**/archive/` is in `.gitignore`)

These directories exist on the original dev machine but a fresh clone
won't have them. Mentioned here for forensic context — the abandoned
approaches and one-off diagnostics behind them are referenced in commit
messages and project memory.

- **`archive/vta_gemm_investigation/`** — one-off diagnostic scripts
  from the VTA GEMM tile-size investigation. Probe scripts (GEMM
  sanity, isolated layers, tile-size sweeps) and a recompile test that
  bracketed the failure. Conclusion is captured in user memory
  (`vta_gemm_tile_limit.md`): with the standard schedule, `n_tiles > ~12`
  or `m_tiles > 4` produces zeros/garbage, which is why the FINN-vs-VTA
  comparison caps at the small CNNs.
- **`archive/exploratory/`** — superseded approaches:
  - `validate_qi.py` — predecessor to wiring QI support directly into
    `benchmark.py` and the C runners. Replaced once the partition
    auto-detect (`config.partition == 'qi'`) was added to the production
    runner path.
  - `test_relay_vta_compile.py` — direct TVM-Relay-on-VTA compile attempt
    for the CIFAR-10 ONNX path. Abandoned because TVM v0.12 needs an
    `onnx.mapping` shim (modern onnx removed it) and Brevitas's
    `Quant` op isn't supported by the TVM frontend.

---

## Not yet documented

The scripts below exist but haven't been described in this README yet —
add details here as future sessions touch them. Suggested categorization:

### VTA-side helpers (host)
- `export_vta_cnn.py`, `export_vta_cnn_int4_o8.py` — TVM/VTA model export
- `calibrate_int4_shifts.py` — INT4 shift calibration
- `vta_compile_all_sizes.py` — bulk VTA compile across model sizes
- `load_vta_bitstream.py` — bitstream loader
- `prepare_cifar10_for_board.py` — CIFAR-10 preprocess for VTA path
- `verify_cifar10_data_path.py`, `verify_m_loop_math.py` — VTA diagnostics
- `vta_infer.c` — VTA C runner (analog of `finn_cnn_infer.c`)
- `rebuild_libvta.sh` — rebuild the VTA `.so`

### DPU / Vitis AI (board side runs as `petalinux@…`)
- `run_dpu_transformer.py` — DPU transformer runner
- `probe_dpu_transformer.py`, `dpu_layout_probe.py`, `profile_dpu_subgraphs.py`
  — DPU subgraph and layout investigations

### INT8/precision probes (toolchain-agnostic)
- `probe_dump_int8.py`, `probe_full_int8.py`, `probe_l0_int8.py`
- `probe_subgraph_1.py`, `probe_subgraph_2.py`
- `probe_gemm_isolated.py`

### Brevitas / weight extraction
- `extract_brevitas_weights.py`

### Network setup (one-time host config)
- `board_net_setup.sh` — board-side static IP + USB-Ethernet bringup
- `host_nat_setup.sh` — host-side NAT to give the board internet
