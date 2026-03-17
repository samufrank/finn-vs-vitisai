# Workflow Guide

A reminder on the two FPGA deployment methods for neural networks:
- Vitis AI (Overlay): Uses a pre-built DPU accelerator. Weights stored in DDR.
- FINN (Dataflow): Generates custom hardware per network. Weights stored on-chip in BRAM.

Both target the Kria KV260 board (K26 chip, DPUCZDX8G_ISA1_B4096) at INT8 precision.

Vitis AI restricts our precision to INT8 (the DPU is hardwired for it). FINN restricts our model size, because at INT8 the on-chip BRAM fills up fast, especially for fully connected layers.

## Prerequisites

### PC Setup
- Linux recommended and probably required (tested on Ubuntu 24.04). Vitis is fully contained in its Docker image and works wherever Docker runs. FINN's Docker setup assumes a Linux host and requires Vivado 2022.2 installed on the host machine.
- Docker installed
- Specific requirements depend on the tool, see Vitis AI Setup or FINN Setup below.

> **Note:** All commands below assume `$REPO_ROOT` is the root of this repo.
> Set it or substitute your actual path, e.g. `export REPO_ROOT=~/dev/finn-vs-vitisai`

Both tools run in Docker containers. The instructions below include how to mount our project repo into each container so scripts can be run directly.

### Board Setup
- Kria KV260 with Ubuntu 22.04 and Kria-PYNQ pre-installed on SD card (already set up - no need to reflash)
- Network access (needed for SSH and file transfer): connect board to router via ethernet, then `ssh ubuntu@<board_ip>` (see `board/setup.md` for credentials and IP)
- Alternative for terminal-only access: serial console via micro-USB data cable
  (`minicom -D /dev/ttyUSB1 -b 115200`). Note: you still need network to transfer
  files to/from the board.
- See `board/setup.md` for full board setup instructions

## Vitis AI Setup

### First time setup
- Clone repo: `git clone https://github.com/Xilinx/Vitis-AI $REPO_ROOT/Vitis-AI`
- Pull image (first time only): `docker pull xilinx/vitis-ai-pytorch-cpu:latest`

### Each session
Without project mount (scripts must be copied into Vitis-AI/ directory):
```bash
cd $REPO_ROOT/Vitis-AI
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
conda activate vitis-ai-pytorch
```

With project mount (scripts accessible at /workspace/project/):
```bash
docker run -it \
  -v $REPO_ROOT/Vitis-AI:/workspace \
  -v $REPO_ROOT:/workspace/project \
  xilinx/vitis-ai-pytorch-cpu:latest bash
conda activate vitis-ai-pytorch
```
Scripts at `/workspace/project/vitis_ai/`, models at `/workspace/project/models/`.

## FINN Setup

### First time setup
- Clone repo (renamed to avoid collision with our `finn\` scripts folder): `git clone https://github.com/Xilinx/finn $REPO_ROOT/finn-repo`
- Install Vivado 2022.2 at `/tools/Xilinx`
- Add to `~/.bashrc`:
```bash
  export FINN_XILINX_PATH=/tools/Xilinx
  export FINN_XILINX_VERSION=2022.2
  export FINN_DOCKER_EXTRA="-v /path/to/finn-vs-vitisai:/workspace/project"
```
  Replace `/path/to/finn-vs-vitisai` with your actual `$REPO_ROOT` path
  (bash won't expand variables inside quotes in .bashrc exports).

### Each session
```bash
cd $REPO_ROOT/finn-repo
bash run-docker.sh
```
Scripts at `/workspace/project/finn/`, models at `/workspace/project/models/`.

## Alternative: Copy scripts instead of mounting
If mounting doesn't work or you just don't want to, then copy scripts into the tool workspace instead:
```bash
mkdir -p $REPO_ROOT/Vitis-AI/models $REPO_ROOT/finn/models
cp $REPO_ROOT/models/*.py $REPO_ROOT/Vitis-AI/models/
cp $REPO_ROOT/vitis_ai/*.py $REPO_ROOT/Vitis-AI/
cp $REPO_ROOT/models/*.py $REPO_ROOT/finn-repo/models/
cp $REPO_ROOT/finn/*.py $REPO_ROOT/finn-repo/
```
The downside is you need to re-copy back after any edits needed to push.

## Model definitions

Models are defined in `models/mlp.py` and `models/cnn.py`. Each file contains:
- A standard PyTorch version (for Vitis AI post-training quantization)
- A Brevitas version (for FINN quantization-aware training)
- A config dict with named size presets (tiny, small, medium, etc.)

Both tools must use the same architecture for a fair comparison. FINN is the limiting factor for model size at INT8.

### Fits Both Tools (can be used for comparison)
- MLP `tiny` [64, 32]: fits FINN INT8, fits Vitis AI
- CNN `tiny` [8, 16]: fits FINN INT8, fits Vitis AI

### Fits Vitis AI Only (FINN resource overflow)
- MLP `original` [256, 256, 128]: BRAM overflow on FINN
- CNN `large` [32, 64, 128]: LUT overflow on FINN

### Untested
- MLP `small` [128, 64]
- CNN `small` [16, 32]

## Vitis AI Pipeline (Overlay)

> Commands below assume the project is mounted at `/workspace/project` (see setup above).

### 1. Train and quantize
```bash
# Inside the Vitis AI container
cd /workspace/project/vitis_ai
python train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
```

This trains the model, runs post-training quantization with real calibration data, and exports a quantized xmodel to `quantize_result/`.

Key flags:
- `--batch_size N`: Sets the batch size baked into the compiled model (default 1)
- `--target DPUCZDX8G_ISA1_B4096`: Always specify this or shapes will be wrong

### 2. Compile for KV260
```bash
vai_c_xir \
  -x quantize_result/MLP_int.xmodel \
  -a /workspace/project/vitis_ai/arch_kv260_pynq.json \
  -o compiled \
  -n mlp_mnist_tiny
```

**Critical:** Use `arch_kv260_pynq.json` (fingerprint `0x101000016010407`), NOT the default arch.json in the Vitis AI container — that one has a different fingerprint and will fail on the board.

### 3. Deploy to board
```bash
scp compiled/mlp_mnist_tiny.xmodel ubuntu@<board_ip>:~/models/vitis_ai/
```

### Batch size ablation

Batch size is baked in at quantization time. To test multiple batch sizes, re-run quantize + compile for each:
```bash
for bs in 1 4 8 16 32 64; do
    python train_and_quantize.py --model mlp --dataset mnist --size tiny --batch_size $bs --epochs 10
    vai_c_xir -x quantize_result/MLP_int.xmodel \
      -a /workspace/project/vitis_ai/arch_kv260_pynq.json \
      -o compiled -n mlp_mnist_tiny_b${bs}
done
```

## FINN Pipeline (Dataflow)

> Commands below assume the project is mounted at `/workspace/project` (see setup above).

### 1. Train and export
```bash
# Inside the FINN container
cd /workspace/project/finn
python train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
```

This trains a Brevitas model with quantization-aware training and exports to ONNX.

Brevitas requirements for FINN:
- Weights: `Int8WeightPerTensorFloat` (signed INT8)
- Activations after ReLU: `Uint8ActPerTensorFloat` (unsigned — FINN requires this)

### 2. Estimate Resources (Optional)
```bash
python estimate_resources.py --model mlp_mnist_tiny.onnx --fps 1000
```

**Warning:** Pre-HLS estimates can significantly underestimate actual usage. An MLP estimated 49 BRAM18 pre-HLS but required 255+ after HLS synthesis. Use this as a quick sanity check, not final verification.

KV260 limits: 288 BRAM18, ~117K LUTs, 1,248 DSPs.

### 3. Compile (Generate Bitstream)
```bash
python compile.py --model mlp_mnist_tiny.onnx --fps 1000
```

This runs the full FINN build pipeline including Vivado synthesis. Takes 20-60+ minutes. Produces a deployment package with bitstream (.bit), hardware handoff (.hwh), and PYNQ driver.

### 4. Deploy to board
```bash
scp -r output_mlp_mnist_tiny/deploy/ ubuntu@<board_ip>:~/models/finn/mlp_mnist_tiny/
```

## Deploying to the Board

The board runs Ubuntu 22.04 with Kria-PYNQ pre-installed. Do not clone this repo onto the board. The board is a deployment target.

### Board directory structure
```
/home/ubuntu/
├── benchmark.py          # Main benchmarking script
├── models/
│   ├── vitis_ai/         # Compiled xmodels
│   └── finn/             # FINN deployment packages
├── results/              # JSON benchmark results
├── MNIST/                # Test datasets
├── cifar-10-batches-py/
└── archive/              # Old iteration scripts
```

### Running benchmarks
```bash
ssh ubuntu@<board_ip>
sudo su
source /etc/profile.d/pynq_venv.sh

# Vitis AI model
python3 /home/ubuntu/benchmark.py \
  --model /home/ubuntu/models/vitis_ai/mlp_mnist_tiny.xmodel \
  --name vitisai_mlp-64x32 \
  --dataset mnist --batch 1 --runs 5

# FINN model (not yet implemented - benchmark.py needs --toolchain finn flag)
```

The `--name` flag controls the result filename. Use the format: `{tool}_{arch}-{sizes}`, e.g.:
- `vitisai_mlp-256x256x128`
- `finn_mlp-64x32`
- `vitisai_cnn-8x16`

Results are saved as JSON to `/home/ubuntu/results/` with full config, per-run data, and summary stats.

### Copying results to repo

In a terminal other than the one running your containers or the board:

```bash
scp ubuntu@<board_ip>:/home/ubuntu/results/*.json ~/dev/CEN571-final/finn-vs-vitisai/results/vitis_ai/
```

## Power Measurement

Power is measured via the on-board INA260 sensor, read through the Linux hwmon sysfs interface at approximately 100 Hz during inference. This measures total board power (PS + PL + memory + I/O).

Benchmark protocol:
1. Thermal stabilization (10s wait)
2. Idle power baseline (10s, ~500 samples)
3. Warmup inference run (10 batches)
4. 5 measured runs with concurrent power sampling
5. Results include mean, std dev, and per-run breakdowns

## Differences between tools

| Aspect | Vitis AI (Overlay) | FINN (Dataflow) |
|--------|-------------------|-----------------|
| Quantization | Post-training (PTQ) | Quantization-aware training (QAT) |
| Weight storage | DDR (off-chip) | BRAM (on-chip) |
| Hardware | Fixed DPU, shared across models | Custom hardware per model |
| Compile time | Seconds | 30-60+ minutes |
| Batch size control | Set during quantization (dummy input shape) | Set via target_fps and folding config |
| Parallelism control | None (fixed by DPU config) | PE/SIMD per layer (folding parameters) |
| Resource constraint | Minimal (DPU pre-built) | Must fit in available BRAM/LUT/DSP |
| Model swap | Load new xmodel, no FPGA rebuild | New bitstream required |

## What doesn't need the board

Most work happens entirely on your PC:
- Model definition and training (both tools)
- Vitis AI quantization and compilation
- FINN training, export, resource estimation, and compilation
- All of the above produce files that get copied to the board at the end

The board is only needed for the final deployment and benchmarking step. The compilation will fail if your model will not fit on the target board.
