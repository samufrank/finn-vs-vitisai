#!/bin/bash
# Driver for compile timing — runs inside Vitis AI Docker.
# Steps: install brevitas/h5py, compile transformer, time CNN compile, regen+time MLP.
set -e
set -o pipefail

# Activate vitis-ai-pytorch conda env (Vitis AI 4.0 default).
# Try common conda init locations.
for f in /opt/vitis_ai/conda/etc/profile.d/conda.sh \
         /opt/conda/etc/profile.d/conda.sh \
         /home/vitis-ai-user/.bashrc ; do
    [ -f "$f" ] && source "$f" 2>/dev/null || true
done
# If conda still isn't a function, add via shell hook.
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found"
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate vitis-ai-pytorch
echo "  conda env: $CONDA_DEFAULT_ENV"
echo "  python:    $(which python3)"
echo "  vai_c_xir: $(which vai_c_xir)"

# Install missing deps. brevitas + h5py needed for the transformer script.
echo ""
echo "===== Installing brevitas + h5py (one-time) ====="
pip install --quiet brevitas==0.10.2 h5py 2>&1 | tail -5 || {
    echo "WARN: brevitas==0.10.2 install failed; trying without version pin"
    pip install --quiet brevitas h5py 2>&1 | tail -5 || echo "WARN: brevitas install failed; Gate B will be skipped"
}

cd /workspace/project/vitis_ai

# ===== Step 1: Transformer compile =====
echo ""
echo "===== STEP 1: Transformer compile (vai_q + vai_c_xir) ====="
SECONDS=0
python3 compile_dpu_transformer_radioml.py 2>&1 | tee transformer_compile.log
echo ""
echo "Step 1 wall-clock: ${SECONDS} seconds"

# ===== Step 2: CNN compile timing =====
echo ""
echo "===== STEP 2: CNN compile timing ====="
cd /workspace/project/vitis_ai/zu3_b512
ls -lh quantize_result/CNN_int.xmodel
mkdir -p compiled_timing_cnn
SECONDS=0
{ time vai_c_xir -x quantize_result/CNN_int.xmodel \
                 -a ../arch_zu3_b512.json \
                 -o compiled_timing_cnn \
                 -n cnn_mnist_tiny ; } 2>&1 | tee /workspace/project/vitis_ai/cnn_compile.log
echo "Step 2 wall-clock: ${SECONDS} seconds (vai_c_xir alone via 'time' above)"

# ===== Step 3: MLP regenerate + compile timing =====
echo ""
echo "===== STEP 3: MLP regen (train+quantize) + compile timing ====="
cd /workspace/project/vitis_ai/zu3_b512
SECONDS=0
python3 ../train_and_quantize.py --model mlp --dataset mnist --size tiny \
        --epochs 10 --target DPUCZDX8G_ISA1_B512 2>&1 | tee /workspace/project/vitis_ai/mlp_train.log
echo "Step 3a (train+quantize) wall-clock: ${SECONDS} seconds"
ls -lh quantize_result/MLP_int.xmodel
mkdir -p compiled_timing_mlp
SECONDS=0
{ time vai_c_xir -x quantize_result/MLP_int.xmodel \
                 -a ../arch_zu3_b512.json \
                 -o compiled_timing_mlp \
                 -n mlp_mnist_tiny ; } 2>&1 | tee /workspace/project/vitis_ai/mlp_compile.log
echo "Step 3b wall-clock: ${SECONDS} seconds (vai_c_xir alone via 'time' above)"

echo ""
echo "===== ALL DONE ====="
echo "Logs:"
echo "  /workspace/project/vitis_ai/transformer_compile.log"
echo "  /workspace/project/vitis_ai/cnn_compile.log"
echo "  /workspace/project/vitis_ai/mlp_train.log"
echo "  /workspace/project/vitis_ai/mlp_compile.log"
echo "Outputs:"
echo "  /workspace/project/vitis_ai/compiled_transformer_radioml/"
echo "  /workspace/project/vitis_ai/zu3_b512/compiled_timing_cnn/"
echo "  /workspace/project/vitis_ai/zu3_b512/compiled_timing_mlp/"
