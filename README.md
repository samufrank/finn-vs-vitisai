# Energy Efficiency Comparison of Overlay vs Dataflow DNN Deployment on FPGA

## Overview

This project compares two FPGA deployment strategies for deep neural networks:

- **Overlay:** A pre-built accelerator runs on the FPGA. Different models are deployed by loading new weights and instructions. Weights stored in DDR.
- **Dataflow (FINN):** Custom hardware is generated for each network. Data flows through a dedicated pipeline. Weights stored on-chip in BRAM.

Both tools target the same precision (INT8) so the comparison isolates the deployment method -- not the quantization level.

## Measurements

For each model deployed through each tool:
- Throughput (FPS)
- Latency (ms/image)
- Power consumption (W) via external USB power meter (FNB58) inline on board power input
- Energy efficiency (mJ/image)
- FPGA resource utilization (LUTs, DSPs, BRAM)
- On-chip thermal data (PS/PL temperature, VCCINT voltage) via SYSMON

We vary batch size (Vitis AI) and target FPS (FINN) to study how each deployment method scales. Input size ablations are also planned. Maybe some others too...

## Hardware

### Primary: AUP-ZU3 (Real Digital)
- **Chip:** XCZU3EG (Zynq UltraScale+ MPSoC), SFVC784 package
- **Resources:** 360 DSPs, 432 BRAM18, ~70K LUTs
- **PYNQ:** 3.1.1 (only available image for this board)
- **Connection:** USB networking only (no ethernet)
- **Power measurement:** External FNB58 USB power meter (no on-board INA260)

### Secondary: Kria KV260 (previously tested)
- **Chip:** K26 SoM (Zynq UltraScale+ MPSoC)
- **Resources:** 1,248 DSPs, 288 BRAM18, ~117K LUTs
- **DPU:** DPUCZDX8G_ISA1_B4096
- **Notes:** Vitis AI results from KV260 are in `results/vitis_ai/`. Vitis AI is currently blocked on the ZU3 due to XRT 2.17 incompatibility with pynq-dpu (see `docs/troubleshooting.md`).

## Key Constraints

Both tools must use the same precision for a fair comparison. The overlay tool's precision determines the constraint. For the DPU this is hardwired INT8. At INT8, FINN's on-chip BRAM fills up fast, especially for fully connected layers, making FINN the limiting factor on model size.

The ZU3's higher BRAM count (432 vs 288 BRAM18) gives FINN more headroom than the KV260, while the KV260's higher DSP count (1,248 vs 360) supports a larger DPU configuration. The boards have complementary strengths for the two tools.


## Repo Structure

```
├── models/
│   ├── mlp.py                    # MLP definitions (PyTorch + Brevitas), parameterized sizes
│   └── cnn.py                    # CNN definitions (PyTorch + Brevitas), parameterized sizes
├── vitis_ai/
│   ├── train_and_quantize.py     # Train, quantize (PTQ), export xmodel
│   ├── arch_zu3_pynq.json        # DPU arch file for AUP-ZU3 (target: B2304)
│   ├── arch_zu3_b1600_pynq.json  # DPU arch file for AUP-ZU3 (fingerprint: B1600)
│   └── arch_kv260_pynq.json      # DPU arch file for KV260 (fingerprint: B4096)
├── finn/
│   ├── train_and_export.py       # Train (QAT via Brevitas), export ONNX
│   ├── compile.py                # Full FINN build --> bitstream (--board Ultra96)
│   └── estimate_resources.py     # Quick resource check (pre-HLS, unreliable at INT8)
├── board/
│   ├── benchmark.py              # Benchmarking script (power + timing + accuracy)
│   ├── host_nat_setup.sh         # Host-side USB networking + NAT setup
│   ├── board_net_setup.sh        # Board-side internet access setup
│   └── setup.md                  # Board setup, credentials, directory layout
├── results/
│   ├── vitis_ai/                 # JSON benchmark results (KV260)
│   ├── finn/                     # JSON benchmark results (ZU3)
│   └── README.md                 # Naming convention and data format
└── docs/
    ├── workflow.md               # How to run each pipeline end-to-end
    ├── troubleshooting.md        # Common issues and solutions
    └── findings.md               # Key technical findings
```

Vitis-AI/ and finn-repo/ tool repos are cloned inside this repo but gitignored. See `docs/workflow.md` for setup instructions.

## Quick Start

### FINN (Dataflow) Pipeline
```bash
# In FINN Docker container
python finn/train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
python finn/compile.py --model mlp_mnist_tiny.onnx --fps 1000
# scp deployment package to board, run benchmark.py
```

### Vitis AI (Overlay) Pipeline
```bash
# In Vitis AI Docker container
conda activate vitis-ai-pytorch
python vitis_ai/train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
vai_c_xir -x quantize_result/MLP_int.xmodel -a vitis_ai/arch_zu3_b1600_pynq.json -o compiled -n mlp_mnist_tiny
# Note: Vitis AI deployment on AUP-ZU3 currently blocked (XRT incompatibility).
# KV260 results available in results/vitis_ai/. See docs/troubleshooting.md.
```

### Running Benchmarks
```bash
# On the board (as root, pynq-venv sourced)
python3 /home/xilinx/benchmark.py \
  --toolchain finn \
  --model /home/xilinx/models/finn/mlp_mnist_tiny/deploy \
  --name finn_mlp-64x32 --dataset mnist --runs 5
```

See `docs/workflow.md` for full instructions.

## Current Status

### Working
- FINN pipeline end-to-end: train --> export --> compile --> deploy --> benchmark
- FINN MLP on ZU3: 96.6% accuracy (MNIST), ~243 FPS, ~4.1ms latency
- FINN CNN on ZU3: deployed and confirmed running
- Vitis AI pipeline on KV260: MLP batch-size ablation (batch 1–64, MNIST + CIFAR-10) with power measurement
- Benchmark infrastructure: JSON output, SYSMON thermal logging, power-ready when FNB58 arrives

### Blocked
- Vitis AI on AUP-ZU3: XRT 2.17 incompatibility with pynq-dpu (requires XRT ≤2.15). Investigating alternative overlay architectures (VTA, Tensil) that don't depend on XRT.

### TODO
- Power measurements (FNB58 meter in transit)
- CNN Vitis AI results
- Transformer deployment (FINN-T / alternative)
- Full ablation studies
- Alternative overlay architecture validation (VTA on PYNQ 3.1.1)

## References

- [FINN](https://github.com/Xilinx/finn)
- [Vitis AI](https://github.com/Xilinx/Vitis-AI)
- [Apache TVM / VTA](https://github.com/apache/tvm)
- [DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ)
- [AUP-ZU3 PYNQ](https://xilinx.github.io/AUP-ZU3/)
