# Energy Efficiency Comparison of Dataflow vs Overlay DNN Accelerators on FPGA

## Overview

This project compares three FPGA deployment strategies for deep neural networks on a single resource-constrained device (Zynq UltraScale+ ZU3EG):

- Dataflow (FINN): Custom hardware generated per model. Data flows through a dedicated pipeline with weights stored on-chip in BRAM. Each new model requires a full hardware build.
- Open-source overlay (VTA/TVM): A fixed accelerator with an open ISA. Models are deployed by loading new weights and microcode. Bitstream is reusable across models.
- Commercial overlay (Vitis AI DPU): AMD's production DNN processor with a closed ISA. Models are compiled to a proprietary instruction set and loaded at runtime.

All three frameworks are benchmarked at INT8 precision on the same physical board with external power measurement, isolating the architectural differences rather than quantization effects or platform variation. This addresses a gap in prior work (e.g., Hamanaka et al. 2023) where mismatched precision and estimation-based power made comparisons uninterpretable.

## Results

Board: AUP-ZU3 (Zynq UltraScale+ ZU3EG, 360 DSPs, 432 BRAM18, ~70K LUTs).
Power measurement: FNIRSI FNB58 USB inline meter (total board power, 100 Hz sampling).

### MLP, MNIST (784→64→32→10, INT8)

| Metric | FINN | VTA (C runner) | Vitis AI DPU |
|--------|------|----------------|--------------|
| Quantization | Brevitas QAT | Brevitas QAT | vai_q_pytorch PTQ |
| Accuracy | 96.58% | 96.45% | 97.14% |
| Throughput (FPS) | 241 | 1270 | 2905 |
| Latency (ms) | 4.15 | 0.79 | 0.34 |
| Dynamic power (W) | 0.28 | 0.12 | 0.23 |
| Energy/inference (mJ) | 14.84 | 3.49 | 1.57 |

FINN and VTA use the same Brevitas-trained weights. Vitis AI uses its own quantizer (each toolchain's native flow). VTA runs at 250 MHz, DPU at 300/600 MHz. FINN's first MatMul falls back to CPU on the ZU3, a resource constraint that limits its throughput on this device.

### CNN, MNIST (Conv [8,16] + FC, INT8)

| Metric | FINN | VTA (Python) |
|--------|------|--------------|
| Accuracy | 92.0% | 90.3% |
| Throughput (FPS) | 24.4 | 26.9 |
| Latency (ms) | 41.0 | 37.1 |
| Dynamic power (W) | 0.29 | 0.23 |
| Energy/inference (mJ) | 135.7 | 162.9 |

CNN tiny is intentionally undersized (float baseline 91.2%), selected to fit FINN's on-chip BRAM constraints at INT8 on the ZU3. DPU CNN results pending.

### Key Findings

- DPU dominance on small FPGAs is consistent with prior work. Hamanaka et al. predicted overlays would outperform dataflow on resource-constrained devices. Our results confirm this: FINN's CPU fallback on the ZU3 is the bottleneck, not a flaw in FINN's architecture.
- Dynamic power is nearly identical (~0.23-0.29 W) across all three frameworks. Energy per inference is throughput-dominated.
- Runtime overhead matters. VTA's C runner (1270 FPS) vs Python runner (257 FPS) shows that software stack overhead can dominate hardware execution time for tiny models.

## Hardware

- Board: AUP-ZU3 (Real Digital), XCZU3EG-SFVC784, 8 GB DDR4
- SD card 1 (PYNQ 3.1.1): FINN and VTA deployment. SSH/scp via USB networking.
- SD card 2 (PetaLinux 2024.1): Vitis AI DPU deployment. Serial console + USB gadget networking.
- Power meter: FNIRSI FNB58 (firmware V1.11), inline on board power input. Host-side logging at 100 Hz, merged with benchmark timestamps post-hoc.

## Toolchain Setup

### FINN (Dataflow)

Runs in the FINN Docker container. Uses Brevitas for quantization-aware training and the FINN compiler to generate a streaming dataflow bitstream.

```bash
# Train + export (in FINN Docker)
python finn/train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
# Compile to bitstream (uses Ultra96 board def, same ZU3EG die)
python finn/compile.py --model mlp_mnist_tiny.onnx --fps 1000
# Deploy: scp package to board, run benchmark.py
```

### VTA (Open-Source Overlay)

VTA bitstream is built via a split-tool flow: Vivado HLS 2020.1 (in Docker) for IP generation, Vivado 2022.2 for synthesis/implementation. This split is required because Vitis HLS 2022.2 has behavioral incompatibilities with VTA's HLS source, and Vivado 2020.1's implementation tools segfault on kernel 6.x.

TVM v0.12.0 is used for the host-side compiler and board-side runtime. Models are compiled using the manual TE path (GEMM + ALU shift + clip) because `relay.quantize` is broken for dense-only models in TVM v0.12.0.

```bash
# Export model for board-side execution (on host)
python board/export_vta_model.py   # MLP
python board/export_vta_cnn.py     # CNN
# scp model directory to board
# On board: link .o to .so, then run benchmark.py or vta_infer (C runner)
```

Pre-built VTA bitstreams (100 MHz, 250 MHz) are archived in `bitstreams/`.

### Vitis AI (Commercial Overlay)

DPUCZDX8G B512 deployed via Vivado 2024.1 block design + PetaLinux 2024.1. The DPU is accessed through the `/dev/dpu` kernel driver. XRT is not used (XRT 2.17 broke pynq-dpu binary compatibility with no migration path). VART 3.5.0 handles model loading and execution.

```bash
# Quantize + compile (in Vitis AI Docker)
conda activate vitis-ai-pytorch
python vitis_ai/train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
vai_c_xir -x quantize_result/MLP_int.xmodel \
  -a vitis_ai/arch_zu3_b512.json -o compiled -n mlp_mnist_tiny
# Deploy to PetaLinux SD card via serial + wget
```

See `docs/` for detailed setup guides.

## Benchmarking

All benchmarks use the same infrastructure:

1. `board/fnb58_logger.py`: host-side power logging to CSV at 100 Hz
2. `board/benchmark.py` or `board/vta_infer.c`: board-side inference with absolute UNIX timestamps per run
3. `board/merge_power.py`: aligns power CSV with benchmark JSON by timestamp, generates merged results and optional power timeline plots

Board clocks must be synced with the host before each run (no RTC on either SD image).

```bash
# Example: VTA MLP benchmark with power measurement
# Host: start power logger
python3 board/fnb58_logger.py -o results/vta/vta_mlp_power.csv
# Board: run benchmark
python3 benchmark.py --toolchain vta --model /home/xilinx/models/vta/mlp_mnist_tiny \
  --dataset mnist --runs 3 --stabilize 10 --idle 10
# Host: merge
python3 board/merge_power.py --benchmark /tmp/bench.json \
  --power results/vta/vta_mlp_power.csv --output results/vta/vta_mlp.json --plot
```

## Repo Structure

```
finn-vs-vitisai/
├── board/                  # Board deployment, benchmarking, and infrastructure
│   ├── benchmark.py            # Unified benchmark runner (FINN, VTA, DPU)
│   ├── vta_infer.c             # C inference runner (VTA MLP + CNN)
│   ├── export_vta_model.py     # Cross-compile VTA MLP for board
│   ├── export_vta_cnn.py       # Cross-compile VTA CNN for board
│   ├── pynq_driver_xrt.cc      # VTA XRT driver source (builds on board)
│   ├── rebuild_libvta.sh       # One-command driver rebuild on board
│   ├── fnb58_logger.py         # Host-side FNB58 power logger (100 Hz)
│   ├── merge_power.py          # Post-hoc power/benchmark timestamp merge
│   ├── fnb58_guide.md          # Power measurement workflow
│   ├── host_nat_setup.sh       # Host-side USB networking + NAT
│   ├── board_net_setup.sh      # Board-side internet access
│   └── setup.md                # Board setup and credentials
├── bitstreams/             # Archived VTA bitstreams (100 MHz, 250 MHz)
├── models/                 # Shared model definitions (mlp.py, cnn.py)
├── finn/                   # FINN pipeline: train (Brevitas QAT), export, compile
├── vitis_ai/               # Vitis AI pipeline: quantize, compile + DPU arch files
├── vta/                    # VTA host-side RPC test scripts (development/debugging)
├── results/                # Benchmark JSONs, power CSVs, timeline plots (gitignored)
│   ├── finn/
│   ├── vta/
│   └── vitis_ai/
└── docs/                   # Setup guides, troubleshooting
```

Tool repos (`finn-repo/`, `Vitis-AI/`) and datasets (`data/`) are cloned/downloaded locally but gitignored.

## Documentation

- `board/setup.md` - Board setup, connectivity, and deployment instructions
- `board/fnb58_guide.md` - FNB58 power measurement workflow
- `docs/dpu_setup_guide.md` - Building the DPU PetaLinux image for AUP-ZU3
- `docs/workflow.md` - End-to-end pipeline instructions *(predates VTA and PetaLinux DPU, update pending)*
- `docs/troubleshooting.md` - Common issues and solutions *(partial, covers FINN and basic board/VTA setup)*

## Status

### Complete
- Three-way MLP comparison (FINN, VTA, DPU) with matched INT8 precision and physical power measurement
- Two-way CNN comparison (FINN, VTA) with power measurement
- Board-side inference for all toolchains (no RPC overhead for VTA)
- FNB58 power measurement infrastructure (logger, merge, timeline plots)
- VTA C inference runner eliminating Python overhead

### In Progress
- Vitis AI CNN on DPU (completes three-way CNN comparison)
- VTA CNN C runner accuracy regression (86% vs 91% in Python, under investigation)

### Planned
- Larger model benchmarks (MLP small [128,64], CNN small [16,32]) if resources permit
- INT4 VTA bitstream (demonstrates overlay precision flexibility vs DPU's fixed INT8)
- RadioML signal classification (extends comparison beyond image classification)

## References

- [FINN](https://github.com/Xilinx/finn) - Xilinx dataflow compiler
- [Apache TVM / VTA](https://github.com/apache/tvm) - open-source DNN compiler + overlay
- [Vitis AI](https://github.com/Xilinx/Vitis-AI) - AMD production DNN toolchain
- [DPUCZDX8G](https://github.com/Xilinx/Vitis-AI/tree/master/dpu) - AMD DNN Processing Unit IP
- Hamanaka et al., "Comparison of DNN Deployment Frameworks on Resource-Constrained FPGA" (IEEE Access, 2023)
- Boutros, Arora & Betz, "FPGA Architecture for Deep Learning: Survey and Future Directions" (2024)
