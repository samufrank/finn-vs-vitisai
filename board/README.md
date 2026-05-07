# `board/` — on-board runtime, benchmark orchestration, and host-side tooling

Everything that runs on the AUP-ZU3 (PYNQ or PetaLinux SD card) or coordinates
board-side benchmarks from the host: cross-framework benchmark driver, FINN/VTA
C inference runners, host-side power logging, model-export utilities, and
debug/probe scripts used during bring-up.

The host pushes deploys + sources to `xilinx@192.168.3.1` (PYNQ) or
`petalinux@…` (PetaLinux DPU). The board has matching copies of `.c` runners,
their built `.so` files, and `benchmark.py`. Each session typically scp's the
latest sources, rebuilds the `.so`, then runs `benchmark.py` per deploy. Usage
recipes are in the top-level [README](../README.md) §Reproducing results.

---

## Cross-framework benchmark driver

| File | Description |
|------|-------------|
| `benchmark.py` | Single entry point for every benchmark. `--toolchain {finn,finn-t,vta,dpu,vitis_ai}` selects the code path. Auto-detects FINN partition (classic vs QI) from `cpu_config.json` and dispatches accordingly. Writes a JSON with throughput, latency, accuracy, idle/active power, and the run config. |
| `benchmark_vta_transformer.py` | Python VTA-transformer driver (M-chunking, retry-on-zero, per-stage timing). Used for accuracy validation; `vta_transformer_infer.c` covers throughput/energy. |

## C inference runners

Loaded by `benchmark.py` via ctypes (FINN) or built as standalone executables
(VTA, VTA transformer). Build with `gcc -O2 -shared -fPIC -Wall` on the board
(aarch64, gcc 11.4) for the FINN `.so` files, or per-runner instructions for VTA.

| File | Description |
|------|-------------|
| `finn_mlp_infer.c` | FINN MLP runner. INT8 + INT4 (1-per-byte and 2-per-byte input layouts), classic + QI partitions, single + double-buffer DMA. Public API: `finn_mlp_runner_init`, `…_infer_batch`, `…_infer_one_profiled`, `…_destroy`. |
| `finn_cnn_infer.c` | FINN CNN runner. INT8 + INT4 (mixed in/out precision for QI INT4), classic + QI partitions, double-buffer DMA, binary-search MultiThreshold. |
| `finn_t_infer.c` | FINN-T transformer runner. Single streaming DMA per inference, double-buffered with CPU classifier tail. `FINN_T_OPT` and `FINN_T_TIMING` env-var gates. |
| `vta_infer.c` | VTA C runner. MLP (INT8 + INT4), CNN (INT8 + INT4-o8 with per-channel dequant + zero-point offset), CIFAR-10 input via `cnn_input_c==3` auto-dispatch. Outputs merge-power-compatible JSON. |
| `vta_transformer_infer.c` | VTA transformer C runner. 12 GEMMs + CPU softmax/BN/residual orchestration, `--timing` flag for per-stage breakdown, retry-on-zero logic for intermittent DMA failures. |
| `pynq_driver_xrt.cc` | VTA XRT driver source. Built into `libvta.so` on the board. Adds done_vld polling after fetch idle, COR-clearing read before module start, optional `VTA_DUMP_INSN_DIR` instruction-stream dump. |

### Correctness harnesses (host-side, no board required)

| File | Description |
|------|-------------|
| `test_finn_mlp_infer.py` | Builds `libfinn_mlp_infer.so` with `gcc -Werror`, exercises pack/unpack byte-exactness and end-to-end mock inference against MNIST. |
| `test_finn_cnn_infer.py` | Same for `libfinn_cnn_infer.so`, plus QI-partition mock against a real deploy. Runs from repo root. |

## Power measurement

| File | Description |
|------|-------------|
| `fnb58_logger.py` | Host-side FNIRSI FNB58 USB-C inline meter logger. Streams V/I/P/T to CSV at 100 Hz over HID interface 3. Requires udev rule (see `fnb58_guide.md`). |
| `merge_power.py` | Post-hoc merge of FNB58 CSV with benchmark JSON by timestamp. Computes idle/active means, dynamic power, energy per inference. `--clock-offset` for board-host clock skew, `--plot` for power-timeline PNG. |
| `fnb58_guide.md` | Power measurement workflow: meter setup, udev rule, board clock sync, common failure modes. |

## Model export and weight extraction (host-side)

Compile and pack models for board-side execution. VTA exports cross-compile TVM
TE modules for aarch64 and write a self-contained directory with `.o` modules
plus weight/config/scale arrays.

| File | Description |
|------|-------------|
| `export_vta_model.py` | VTA MLP INT8 export (TE GEMM + ALU shift + clip schedule). |
| `export_vta_model_int4_v2.py` | VTA MLP INT4 export. Reads Brevitas `quant_weight().scale` per layer, `scaling_impl.value` per quantizer. Fully parameterized by `meta.json`. |
| `export_vta_cnn.py` | VTA CNN INT8 export. Generic per-layer config, `--force-m1` flag for single-tile module compilation, skip-add residual support, m-chunk loop. |
| `export_vta_cnn_int4_o8.py` | VTA CNN INT4-o8 export. Per-channel BN-fold, zero-point activation offset, INT8 output modules. |
| `extract_brevitas_weights.py` | Generic Brevitas → numpy extractor for MLP/CNN at INT8/INT4. |
| `extract_int4_brevitas.py` | INT4-specific Brevitas extractor with per-channel scale handling. |
| `prepare_cifar10_for_board.py` | CIFAR-10 host-side preprocessor. Emits uint8 HWC images + uint8 labels in board-ready binary format. |
| `vta_compile_all_sizes.py` | Bulk VTA compile across the model size sweep. |
| `load_vta_bitstream.py` | Bitstream loader helper (PYNQ Overlay wrapper). |

## DPU runtime + profiling (PetaLinux board side)

| File | Description |
|------|-------------|
| `run_dpu_transformer.py` | Custom Python orchestrator for the multi-subgraph transformer (VART for DPU subgraphs, numpy for CPU subgraphs — `libvart-cpu-runner.so` is missing from the PetaLinux 2024.1 BSP). |
| `probe_dpu_transformer.py` | VART API probe (subgraph attributes, runner availability). |
| `probe_subgraph_1.py`, `probe_subgraph_2.py` | xir attribute probes for individual subgraphs. |
| `profile_dpu_subgraphs.py` | Per-DPU-subgraph throughput via `xdputil benchmark -i`. |

## Debug, simulation, and diagnostics

These are kept in the tree as record of what was needed during bring-up; they
are not part of the production benchmark loop.

| File | Description |
|------|-------------|
| `debug_full_pipeline.py` | 31-stage VTA-transformer per-stage diagnostic against host-side reference. |
| `debug_vta_transformer.py` | Single-sample Q-projection diagnostic. |
| `vta_numpy_sim_int4.py` | Numpy reference simulator for VTA INT4 MLP (Modes A–D for sim-to-hardware gap analysis). |
| `vta_numpy_sim_int4_cnn_int8out.py` | Same for VTA INT4-o8 CNN (Modes E–G). |
| `calibrate_int4_shifts.py` | INT4 SHR calibration helper. |
| `diagnose_int4_v2.py` | Per-layer INT4 sim-vs-board divergence diagnostic. |
| `fingerprint_mlp_mt.py` | sha1 fingerprint of MLP MultiThreshold pred vectors (regression check for the `>` vs `>=` semantic fix). |
| `test_tiny_module.py` | Minimal VTA module smoke test. |
| `test_vta_cnn.py` | VTA CNN INT8 board test (im2col + GEMM + post-processing pipeline). |
| `test_vta_cnn_int4_o8.py` | VTA CNN INT4-o8 board test. |
| `test_vta_int4_minimal.py` | VTA INT4 cold-start probe (T1a/T1b/T1c discriminators for the first-GEMM-call artifact). |
| `verify_cifar10_data_path.py` | One-off check on CIFAR-10 binary path layout for the C runner. |

## Network setup (one-time host config)

| File | Description |
|------|-------------|
| `host_nat_setup.sh` | Host-side USB-Ethernet bringup + NAT for PYNQ board internet (auto-detects `enx*` interface, includes NetworkManager fix and MTU 900 setting). |
| `board_net_setup.sh` | Board-side static IP + USB-Ethernet bringup. |
| `rebuild_libvta.sh` | One-command rebuild of the VTA driver `.so` on board (driver only — runtime.cc changes need full cmake; see `docs/vta_build_guide.md`). |

## Documentation

| File | Description |
|------|-------------|
| `setup.md` | Board setup, credentials, SD-card layout, USB networking gotchas. |
| `fnb58_guide.md` | Power measurement workflow (meter, udev rule, merge). |
| `README.md` | This file. |

## Regression baselines

| File | Description |
|------|-------------|
| `regression/int8_baseline_pre_change.json` | INT8 MLP pred-vector + accuracy baseline before benchmark.py refactor. |
| `regression/int8_baseline_post_change.json` | After the `>=` MultiThreshold semantic fix. |
| `regression/int8_baseline_post_benchmark_refactor.json` | After the dual-closure refactor for INT8/INT4 dispatch. |
