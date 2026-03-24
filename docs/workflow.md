# Workflow Guide

> **Note:** All commands assume `$REPO_ROOT` is the root of this repo.
> Set it with `export REPO_ROOT=~/dev/CEN571-final/finn-vs-vitisai` or substitute your path.

Both tools run in Docker containers on your host machine. The board is only needed for the final deployment and benchmarking step.

## Prerequisites

### Host Machine
- Linux (tested on Ubuntu 22.04/24.04)
- Docker installed
- Vivado 2022.2 installed at `/tools/Xilinx` (required for FINN)
- Add to `~/.bashrc`:
```bash
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2022.2
export FINN_DOCKER_EXTRA="-v /path/to/finn-vs-vitisai:/workspace/project"
```
Replace `/path/to/finn-vs-vitisai` with your actual `$REPO_ROOT`.

### Board Setup
See `board/setup.md` for full board setup instructions including:
- SD card flashing
- USB networking (NAT setup scripts)
- PYNQ environment configuration
- Directory structure

SSH access: `ssh 192.168.3.1` (user: xilinx, password: xilinx)

## Vitis AI Setup

> **Status:** Vitis AI deployment on AUP-ZU3 is currently blocked due to XRT 2.17
> incompatibility with pynq-dpu. Compilation works; board deployment does not.
> KV260 results available in `results/vitis_ai/`. See `troubleshooting.md`.

### First time setup
```bash
git clone https://github.com/Xilinx/Vitis-AI $REPO_ROOT/Vitis-AI
docker pull xilinx/vitis-ai-pytorch-cpu:latest
```

### Each session
```bash
docker run -it \
  -v $REPO_ROOT/Vitis-AI:/workspace \
  -v $REPO_ROOT:/workspace/project \
  xilinx/vitis-ai-pytorch-cpu:latest bash
conda activate vitis-ai-pytorch
```
Scripts at `/workspace/project/vitis_ai/`, models at `/workspace/project/models/`.

Alternatively, if you're already in `$REPO_ROOT`:
```bash
docker run -it \
  -v $(pwd)/Vitis-AI:/workspace \
  -v $(pwd):/workspace/project \
  xilinx/vitis-ai-pytorch-cpu:latest bash
conda activate vitis-ai-pytorch
```

## FINN Setup

### First time setup
```bash
git clone https://github.com/Xilinx/finn $REPO_ROOT/finn-repo
```

### Each session
```bash
cd $REPO_ROOT/finn-repo
bash run-docker.sh
```
Scripts at `/workspace/project/finn/`, models at `/workspace/project/models/`.

## Model Definitions

Models are defined in `models/mlp.py` and `models/cnn.py`. Each file contains:
- A standard PyTorch version (for Vitis AI PTQ)
- A Brevitas version (for FINN QAT)
- A config dict with named size presets

Brevitas requirements for FINN:
- Weights: `Int8WeightPerTensorFloat` (signed INT8)
- Activations after ReLU: `Uint8ActPerTensorFloat` (unsigned — FINN requires this)

### Model size configs
| Model | Config | Hidden sizes | FINN INT8 on ZU3 | Vitis AI |
|-------|--------|-------------|-------------------|----------|
| MLP | tiny | [64, 32] | ✓ fits | ✓ |
| MLP | small | [128, 64] | ✓ fits (confirmed) | ✓ |
| MLP | medium | [256, 128] | untested | ✓ |
| MLP | original | [256, 256, 128] | ✓ fits on ZU3 | ✓ |
| CNN | tiny | [8, 16] | ✓ fits | ✓ |
| CNN | small | [16, 32] | untested | ✓ |
| CNN | large | [32, 64, 128] | likely overflow | ✓ |

FINN is the limiting factor on model size at INT8. Always verify with full compilation.

## Vitis AI Pipeline

> Commands assume project mounted at `/workspace/project`.

### 1. Train and quantize
```bash
# Inside Vitis AI container
cd /workspace/project/vitis_ai
python train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
```

Key flags:
- `--target DPUCZDX8G_ISA1_B1600` — always specify for AUP-ZU3
- `--batch_size N` — batch size is baked into the compiled model

### 2. Compile
```bash
# For AUP-ZU3 (B1600):
vai_c_xir \
  -x quantize_result/MLP_int.xmodel \
  -a /workspace/project/vitis_ai/arch_zu3_b1600_pynq.json \
  -o compiled \
  -n mlp_mnist_tiny

# For KV260 (B4096):
vai_c_xir \
  -x quantize_result/MLP_int.xmodel \
  -a /workspace/project/vitis_ai/arch_kv260_pynq.json \
  -o compiled \
  -n mlp_mnist_tiny
```

### 3. Deploy to board
```bash
scp compiled/mlp_mnist_tiny.xmodel xilinx@192.168.3.1:~/models/vitis_ai/
# dpu.bit, dpu.hwh, dpu.xclbin must also be in ~/models/vitis_ai/ on the board
```

### Batch size ablation
```bash
for bs in 1 4 8 16 32 64; do
    python train_and_quantize.py --model mlp --dataset mnist --size tiny \
      --batch_size $bs --epochs 10
    vai_c_xir -x quantize_result/MLP_int.xmodel \
      -a /workspace/project/vitis_ai/arch_zu3_b1600_pynq.json \
      -o compiled -n mlp_mnist_tiny_b${bs}
done
```

## FINN Pipeline

> Commands assume project mounted at `/workspace/project`.

### 1. Train and export
```bash
# Inside FINN container
cd /workspace/project/finn
python train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
# Output: mlp_mnist_tiny.onnx, mlp_mnist_tiny.pth
```

### 2. Estimate Resources (optional, unreliable at INT8)
```bash
python estimate_resources.py --model mlp_mnist_tiny.onnx --board Ultra96
```

**Warning:** Pre-HLS estimates significantly underestimate BRAM at INT8. Use as a rough sanity check only.

### 3. Compile
```bash
python compile.py --model mlp_mnist_tiny.onnx --fps 1000
# Output: output_mlp_mnist_tiny/deploy/
```

Takes 20-60+ minutes. Uses `--board Ultra96` by default (same ZU3EG chip, compatible PS config).

### 4. Extract CPU layer weights (if partial hardware mapping)

FINN sometimes splits the network between CPU and FPGA. This happens when a layer
can't be mapped to hardware (e.g. FC layers with non-power-of-2 input sizes, first/last
layers of CNNs). Check `intermediate_models/dataflow_parent.onnx` to see the split.

If CPU layers exist, extract their weights and include them in the deploy package:

```bash
# Run from the finn/ directory, adjust prefix and output dir for your model
python3 -c "
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
m = ModelWrapper('output_mlp_mnist_tiny/intermediate_models/dataflow_parent.onnx')
prefix = 'mlp_'  # use 'cnn_' for CNN models
for init in m.model.graph.initializer:
    arr = np.frombuffer(init.raw_data, dtype=np.float32).copy() if init.raw_data \
          else np.array(init.float_data, dtype=np.float32)
    dims = list(init.dims) if init.dims else [1]
    arr = arr.reshape(dims)
    np.save(f'output_mlp_mnist_tiny/deploy/{prefix}{init.name}.npy', arr)
    print(f'Saved {prefix}{init.name}: {arr.shape}')
"
```

benchmark.py auto-detects these `.npy` files and handles the CPU/FPGA split transparently.

### 5. Deploy to board
```bash
# Create directory first
ssh xilinx@192.168.3.1 "mkdir -p ~/models/finn/mlp_mnist_tiny"

# Copy deploy package (includes .npy files if extracted above)
scp -r output_mlp_mnist_tiny/deploy/ \
    xilinx@192.168.3.1:~/models/finn/mlp_mnist_tiny/
```

## Running Benchmarks

```bash
# On the board as root
# (xrt_setup.sh + pynq_venv.sh auto-source if added to /root/.bashrc)

# FINN model
python3 /home/xilinx/benchmark.py \
  --toolchain finn \
  --model /home/xilinx/models/finn/mlp_mnist_tiny/deploy \
  --name finn_mlp-64x32 \
  --dataset mnist --batch 1 --runs 5

# Vitis AI model (run from models/vitis_ai/ so dpu.bit is found in cwd)
cd /home/xilinx/models/vitis_ai
python3 /home/xilinx/benchmark.py \
  --toolchain vitis_ai \
  --model /home/xilinx/models/vitis_ai/mlp_mnist_tiny.xmodel \
  --name vitisai_mlp-64x32 \
  --dataset mnist --batch 1 --runs 5
```

Result filename format: `{name}_{dataset}_b{batch}_{timestamp}.json`

Use consistent naming: `{tool}_{arch}-{sizes}`, e.g.:
- `finn_mlp-64x32`
- `vitisai_mlp-64x32`
- `finn_cnn-8x16`

## Copying Results to Repo

```bash
scp xilinx@192.168.3.1:~/results/*.json $REPO_ROOT/results/finn/
```

## What Doesn't Need the Board

- Model training and export (both tools)
- Vitis AI quantization and compilation
- FINN training, export, resource estimation, and compilation

The board is only needed for deployment and benchmarking.

## Tool Comparison

| Aspect | Vitis AI (Overlay) | FINN (Dataflow) |
|--------|-------------------|-----------------|
| Quantization | Post-training (PTQ) | Quantization-aware (QAT) |
| Weight storage | DDR (off-chip) | BRAM (on-chip) |
| Hardware | Fixed accelerator, shared across models | Custom hardware per model |
| Compile time | Seconds | 20-60+ minutes |
| Batch size | Set at quantization time | Set via target_fps |
| Parallelism | Fixed by accelerator config | PE/SIMD per layer |
| Resource constraint | DSP-limited (accelerator config) | BRAM-limited (weights on-chip) |
| Model swap | Load new compiled model, no rebuild | New bitstream required |
| CPU/FPGA split | Reshape/dequant on CPU (automatic) | May offload layers to CPU if mapping fails |
