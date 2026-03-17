# Energy Efficiency Comparison of Overlay vs Dataflow DNN Deployment on FPGA

## Overview

This project compares two FPGA deployment strategies for deep neural networks:

- **Overlay (Vitis AI):** A pre-built accelerator (DPU) runs on the FPGA. Different models are deployed by loading new weights and instructions. Weights stored in DDR.
- **Dataflow (FINN):** Custom hardware is generated for each network. Data flows through a dedicated pipeline. Weights stored on-chip in BRAM.

Both target the same board (Kria KV260) at the same precision (INT8) so the comparison isolates the deployment method — not the hardware or quantization level.

## Measurements

For each model deployed through each tool:
- Throughput (FPS)
- Latency (ms/image)
- Power consumption (W) via on-board INA260 sensor
- Energy efficiency (mJ/image)
- FPGA resource utilization (LUTs, DSPs, BRAM)

We also vary batch size to study how each deployment method scales under different utilization levels. Other ablations such as input size will also be considered.

## Hardware

- **Board:** Kria KV260 Vision AI Starter Kit
- **Chip:** K26 SoM (Zynq UltraScale+ MPSoC)
- **DPU:** DPUCZDX8G_ISA1_B4096
- **Resources:** ~1,248 DSPs, 288 BRAM18, ~117K LUTs

## Key constraint

Vitis AI's DPU is hardwired for INT8. So both tools must target INT8 for a fair comparison. At INT8, FINN's on-chip BRAM fills up fast, especially for fully connected layers. This makes FINN the limiting factor on model size.

See `docs/findings.md` for more observations.

## Repo structure

```
├── models/
│   ├── mlp.py                  # MLP definitions (PyTorch + Brevitas), parameterized sizes
│   └── cnn.py                  # CNN definitions (PyTorch + Brevitas), parameterized sizes
├── vitis_ai/
│   ├── train_and_quantize.py   # Train, quantize (PTQ), export xmodel
│   └── arch_kv260_pynq.json    # DPU arch file (fingerprint 0x101000016010407)
├── finn/
│   ├── train_and_export.py     # Train (QAT via Brevitas), export ONNX
│   ├── compile.py              # Full FINN build --> bitstream
│   └── estimate_resources.py   # Quick resource check (pre-HLS, use with caution)
├── board/
│   ├── benchmark.py            # Benchmarking script (power + timing + accuracy)
│   └── setup.md                # Board setup, credentials, directory layout
├── results/
│   ├── vitis_ai/               # JSON benchmark results from Vitis AI runs
│   ├── finn/                   # JSON benchmark results from FINN runs (pending)
│   └── README.md               # Naming convention and data format
└── docs/
    ├── workflow.md             # How to run each pipeline end-to-end
    ├── troubleshooting.md      # Common issues and solutions
    └── findings.md             # Key technical findings so far
```

Vitis-AI/ and finn/ tool repos are cloned inside this repo but gitignored. See `docs/workflow.md` for setup instructions.

## Quick start

### Vitis AI (Overlay) Pipeline
```bash
# In Vitis AI Docker container
python vitis_ai/train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
vai_c_xir -x quantize_result/MLP_int.xmodel -a vitis_ai/arch_kv260_pynq.json -o compiled -n mlp_mnist_tiny
# scp compiled xmodel to board, run benchmark.py
```

### FINN (Dataflow) Pipeline
```bash
# In FINN Docker container
python finn/train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
python finn/compile.py --model mlp_mnist_tiny.onnx --fps 1000
# scp deployment package to board, run benchmark.py
```

See `docs/workflow.md` for full instructions including Docker setup, mounting, and board deployment.

## Current Status

### Done
- Vitis AI pipeline working end-to-end (train -> quantize -> compile -> deploy -> benchmark)
- MLP batch size ablation on MNIST and CIFAR-10 (batch 1–64), with power measurement
- FINN pipeline working through compilation (MLP tiny and CNN tiny bitstreams generated)
- Benchmark infrastructure with automated power measurement and JSON output

### TODO
- FINN deployment to board and benchmarking
- Finding largest model that fits both tools for direct comparison
- CNN deployment through both tools

- Vitis AI results for tiny models (to match FINN for head-to-head comparison)
- FINN side of benchmark.py (different inference API than Vitis AI)
- Transformer deployment via FINN-T/FINN+
- Full ablation studies with benchmarks for comparison

## References

- [Vitis AI](https://github.com/Xilinx/Vitis-AI)
- [FINN](https://github.com/Xilinx/finn)
- [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ)
