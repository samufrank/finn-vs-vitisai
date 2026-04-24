# Energy Efficiency Comparison of Dataflow vs Overlay DNN Accelerators on FPGA

## Overview

This project compares three FPGA deployment strategies for deep neural networks on a single resource-constrained device (Zynq UltraScale+ ZU3EG):

- Dataflow (FINN): Custom hardware generated per model. Data flows through a dedicated pipeline with weights stored on-chip in BRAM. Each new model requires a full hardware build.
- Open-source overlay (VTA/TVM): A fixed accelerator with an open ISA. Models are deployed by loading new weights and microcode. Bitstream is reusable across models.
- Commercial overlay (Vitis AI DPU): AMD's production DNN processor with a closed ISA. Models are compiled to a proprietary instruction set and loaded at runtime.

All three frameworks are benchmarked at matched precision on the same physical board with external power measurement, isolating architectural differences from quantization effects or platform variation. Prior comparisons (e.g., Hamanaka et al. 2023) have used mismatched precision or estimation-based power, and this project is designed to complement that work by controlling both.

Runtime parity is a separate methodological consideration. Different toolchains ship with different drivers, ranging from Python wrappers to native C and C++. Reported throughput numbers can therefore include a substantial driver-overhead component that is not architectural. To make host-side overhead comparable across frameworks, we run native C inference for all three. FINN uses a ctypes-loaded C runner built on top of PYNQ's bitstream loader; VTA uses its C inference binary; DPU uses VART (which is natively C++).

## Results

Board: AUP-ZU3 (Zynq UltraScale+ ZU3EG, 360 DSPs, 432 BRAM18, 70K LUTs).
Power measurement: FNIRSI FNB58 USB inline meter (total board power, 100 Hz sampling).
All results use C/C++ runtimes on host side to isolate architectural comparison from host driver overhead.

### MLP, MNIST (784→64→32→10)

| Metric | FINN INT8 | FINN INT4 | VTA INT8 | VTA INT4 | Vitis AI DPU INT8 |
|--------|-----------|-----------|----------|----------|-------------------|
| Quantization | Brevitas QAT | Brevitas QAT | Brevitas QAT | Brevitas QAT | vai_q_pytorch PTQ |
| Accuracy | 96.58% | 97.29% | 96.45% | 93.08% | 97.14% |
| Throughput (FPS) | 1575 | 1811 | 1270 | 1266 | 2905 |
| Latency (ms) | 0.635 | 0.552 | 0.79 | 0.79 | 0.34 |
| Dynamic power (W) | 0.16 | 0.15 | 0.12 | 0.09 | 0.23 |
| Energy/inference (mJ) | 2.21 | 1.90 | 3.49 | 3.15 | 1.57 |

FINN and VTA use the same Brevitas-trained weights. Vitis AI uses its own quantizer (each toolchain's native flow). VTA runs at 250 MHz (INT8) and 200 MHz (INT4), DPU at 300/600 MHz, FINN at 100 MHz with auto-selected folding.

### CNN, MNIST (Conv [8,16] + FC)

| Metric | FINN INT8 | VTA INT8 (Python) | VTA INT4-o8 (Python) | Vitis AI DPU INT8 |
|--------|-----------|-------------------|----------------------|-------------------|
| Accuracy | 91.99% | 90.4% | 81.6% | 86.7% |
| Throughput (FPS) | 454 | 27.0 | 29.2 | 2910 |
| Latency (ms) | 2.205 | 37.0 | 34.3 | 0.34 |
| Dynamic power (W) | 0.18 | 0.24 | 0.28 | 0.23 |
| Energy/inference (mJ) | 7.59 | 170.8 | 142.0 | 1.58 |

CNN tiny is intentionally undersized (float baseline 91.2%), selected to fit FINN's on-chip BRAM constraints at INT8 on the ZU3. VTA CNN C runner has an open accuracy regression bug (approximately 86% in C vs 91% in Python) and is therefore reported in Python for accuracy until resolved. DPU accuracy loss (roughly 5% vs float) is from vai_q_pytorch post-training quantization on an already-small model.

### Key findings

DPU leads on throughput and energy for both MLP and CNN, achieving 1.57 and 1.58 mJ per inference respectively, but trades accuracy for it: post-training quantization on the small CNN drops accuracy about 5 points below FINN's 91.99%. Both numbers are legitimate depending on application constraints.

At matched INT8 precision and matched C runtime, FINN outperforms VTA on both MLP and CNN. MLP energy is 2.21 vs 3.49 mJ and CNN energy is 7.59 vs 170.8 mJ. The pattern extends to INT4 on MLP (1.90 vs 3.15 mJ). CNN matched-C comparison is currently incomplete pending resolution of a VTA CNN C runner accuracy regression.

FINN is currently CPU-bound rather than fabric-bound on these models, with the CPU first MatMul accounting for 82 to 94 percent of per-inference time depending on precision. The FINN compiler, at the throughput target we specified, leaves the first matrix multiply on the CPU and sets minimum folding on the fabric layers. Substantial headroom remains accessible by recompiling with more aggressive throughput targets. Experiments on this and other variables remain open, and headline rankings may shift as those experiments close.

### Transformer Deployment

The comparison has been extended to transformer workloads:

- FINN-T (Paderborn finn-plus 1.4.0) compiled a trained INT4 transformer for RadioML 2018 modulation classification (3 heads, D=96, T=128, 1 layer, 122k parameters, Brevitas QAT). End-to-end performance on ZU3EG: 72.12% accuracy, 1460.8 FPS, 2.76 mJ per inference.
- The Vitis AI DPU compiles linear projections but partitions all attention operations (Q@K^T, softmax, transpose, layer normalization) to CPU. The DPU only supports weight-by-activation matmuls, not the activation-by-activation pattern required by attention.
- VTA can execute all six transformer GEMM operations including the activation-by-activation matmuls the DPU rejects. VTA transformer deployment is in progress: compiled INT4 RadioML transformer modules have been validated bit-exactly against the Mode E host-side reference pipeline (70.53% accuracy), and the remaining work is the board-side inference driver.

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
# For matched-runtime benchmarks, use --finn-runtime c
```

A native C inference runner (`board/finn_mlp_infer.c`, `board/finn_cnn_infer.c`) replaces FINN's Python driver for hot-path inference. The runner loads FINN's bitstream through PYNQ but executes the DMA trigger, polling, cache operations, and CPU-partitioned layers in C via ctypes. The MLP runner supports INT8 and INT4; the CNN runner supports INT8.

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

Pre-built VTA bitstreams (100 MHz INT8, 250 MHz INT8, 200 MHz INT4, 166 MHz INT4-o8) are archived in `bitstreams/`.

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
# Example: FINN MLP C runner with power measurement
# Host: start power logger
python3 board/fnb58_logger.py -o results/finn/finn_mlp_int8_c_power.csv
# Board: run benchmark
python3 benchmark.py --toolchain finn --model /home/xilinx/models/finn/mlp_mnist_tiny/deploy \
  --dataset mnist --runs 3 --stabilize 10 --idle 10 --finn-runtime c
# Host: merge
python3 board/merge_power.py --benchmark /tmp/bench.json \
  --power results/finn/finn_mlp_int8_c_power.csv \
  --output results/finn/finn_mlp_int8_c.json --plot
```

## Repo Structure

```
finn-vs-vitisai/
├── board/                  # Board deployment, benchmarking, and infrastructure
│   ├── benchmark.py            # Unified benchmark runner (FINN, VTA, DPU, FINN-T)
│   ├── vta_infer.c             # C inference runner (VTA MLP + CNN)
│   ├── finn_mlp_infer.c        # C inference runner (FINN MLP, INT8 + INT4)
│   ├── finn_cnn_infer.c        # C inference runner (FINN CNN, INT8)
│   ├── finn_t_infer.c          # C inference runner (FINN-T transformer)
│   ├── test_finn_mlp_infer.py  # CPU-only harness for FINN MLP runner
│   ├── test_finn_cnn_infer.py  # CPU-only harness for FINN CNN runner
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
├── bitstreams/             # Archived VTA bitstreams (100/250 MHz INT8, 200 MHz INT4, 166 MHz INT4-o8)
├── models/                 # Shared model definitions (mlp.py, cnn.py)
├── finn/                   # FINN pipeline: train (Brevitas QAT), export, compile
├── finn-t/                 # FINN-T transformer (trained RadioML 2018 INT4, finn-plus 1.4.0)
├── vitis_ai/               # Vitis AI pipeline: quantize, compile + DPU arch files
├── vta/                    # VTA host-side RPC test scripts (development/debugging)
├── results/                # Benchmark JSONs, power CSVs, timeline plots (CSVs gitignored)
│   ├── finn/
│   ├── finn-t/
│   ├── vta/
│   └── vitis_ai/
└── docs/                   # Build guides (DPU, VTA)
```

Tool repos (`finn-repo/`, `Vitis-AI/`) and datasets (`data/`) are cloned/downloaded locally but gitignored.

## Documentation

- `board/setup.md` - Board setup, connectivity, and deployment instructions
- `board/fnb58_guide.md` - FNB58 power measurement workflow
- `docs/dpu_setup_guide.md` - Building the DPU PetaLinux image for AUP-ZU3 (Vivado block design, PetaLinux, VART)
- `docs/vta_build_guide.md` - Building the VTA bitstream and TVM runtime (split-tool HLS flow, board-side build)

## Status

### Complete
- Three-way MLP comparison (FINN, VTA, DPU) at matched INT8 and INT4 precision, matched C runtime, with physical power measurement
- Three-way CNN comparison at INT8 with FINN and DPU in C, VTA in Python (C runner accuracy regression under investigation)
- Trained-transformer FPGA deployment: FINN-T RadioML 2018 (72.12%, 1460 FPS, 2.76 mJ) on ZU3EG
- Board-side inference for all toolchains (no RPC overhead)
- FNB58 power measurement infrastructure (logger, merge, timeline plots)
- C inference runners for all four accelerator paths (VTA, FINN MLP, FINN CNN, FINN-T)
- DPU transformer compilation test (linear projections on DPU, attention ops partition to CPU)
- VTA transformer GEMM verification (all 6 dimensions tile correctly)

### In Progress
- VTA CNN C runner accuracy regression (86% vs 91% in Python, under investigation)
- FINN CNN at INT4 (pending Brevitas CNN-INT4 training and FINN compile)
- VTA transformer deployment: INT4 RadioML transformer, compiled modules bit-exactly validated against host-side reference at 70.53%, board-side inference driver remaining

### Planned
- FINN recompile with higher target_fps to move first MatMul onto fabric (tests remaining FINN throughput headroom on this device)
- Matched-precision transformer comparison (FINN-T vs VTA, once VTA transformer lands on board)

## References

- [FINN](https://github.com/Xilinx/finn) - Xilinx dataflow compiler
- [Apache TVM / VTA](https://github.com/apache/tvm) - open-source DNN compiler + overlay
- [Vitis AI](https://github.com/Xilinx/Vitis-AI) - AMD production DNN toolchain
- [DPUCZDX8G](https://github.com/Xilinx/Vitis-AI/tree/master/dpu) - AMD DNN Processing Unit IP
- [FINN-T](https://github.com/eki-project/FINN-T) - Transformer dataflow accelerator (Berganski et al., FPT 2024)
- Hamanaka et al., "An Exploration of State-of-the-Art Automation Frameworks for FPGA-Based DNN Acceleration" (IEEE Access, 2023)
- Boutros, Arora & Betz, "FPGA Architecture for Deep Learning: Survey and Future Directions" (TRETS, 2024)
- Machura et al., "Embedded Object Detection with Custom LittleNet, FINN and Vitis AI DCNN Accelerators" (JLPEA, 2022)
