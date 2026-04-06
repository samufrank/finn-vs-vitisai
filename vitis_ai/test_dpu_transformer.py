"""Vitis AI DPU Transformer Compilation Test.

Tests whether DPUCZDX8G (B512) can handle transformer attention ops,
or if they get partitioned to CPU.

Run inside Vitis AI Docker:
    cd ~/dev/CEN571-final/finn-vs-vitisai
    docker run -it \
      -v $(pwd)/Vitis-AI:/workspace \
      -v $(pwd):/workspace/project \
      xilinx/vitis-ai-pytorch-cpu:latest bash

    conda activate vitis-ai-pytorch
    cd /workspace/project/vitis_ai
    python test_dpu_transformer.py

Expected result: attention (softmax, Q@K^T matmul) gets partitioned to CPU.
The compiler log showing DPU vs CPU subgraph partitioning is the evidence.

Target: B512 DPU, fingerprint 0x101000016010400
Arch file: ../vitis_ai/arch_zu3_b512.json (relative to project root)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. Model definition
# ============================================================

class TinyTransformer(nn.Module):
    """Minimal transformer encoder for DPU compilation testing.
    1 encoder layer, small dimensions, classification head.
    """
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128,
                 seq_len=32, num_classes=10):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
            activation='relu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# ============================================================
# 2. Quantize with vai_q_pytorch
# ============================================================

def quantize_model(model, seq_len=32, d_model=64, calib_batches=20):
    """Quantize using Vitis AI PyTorch quantizer."""
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        print("ERROR: pytorch_nndct not available.")
        print("Make sure you're in the Vitis AI Docker with:")
        print("  conda activate vitis-ai-pytorch")
        sys.exit(1)

    device = torch.device('cpu')
    model = model.to(device)
    dummy_input = torch.randn(1, seq_len, d_model).to(device)

    output_dir = './quantize_result'
    os.makedirs(output_dir, exist_ok=True)

    # Calibration pass
    print("Creating quantizer (calib mode)...")
    quantizer = torch_quantizer(
        'calib', model, (dummy_input,),
        output_dir=output_dir, device=device,
    )
    quant_model = quantizer.quant_model

    print(f"Running {calib_batches} calibration batches...")
    with torch.no_grad():
        for i in range(calib_batches):
            calib_input = torch.randn(8, seq_len, d_model).to(device)
            quant_model(calib_input)

    quantizer.export_quant_config()
    print("Calibration complete.")

    # Export xmodel
    print("Creating quantizer (test mode) for xmodel export...")
    quantizer = torch_quantizer(
        'test', model, (dummy_input,),
        output_dir=output_dir, device=device,
    )
    quant_model = quantizer.quant_model

    with torch.no_grad():
        quant_model(dummy_input)

    quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
    print(f"xmodel exported to {output_dir}/")

    return output_dir


# ============================================================
# 3. Compile for DPU
# ============================================================

def compile_for_dpu(xmodel_dir):
    """Compile the quantized xmodel for B512 DPU.

    The compiler output shows which ops map to DPU vs CPU subgraphs.
    """
    # Find xmodel
    xmodel_file = None
    for f in os.listdir(xmodel_dir):
        if f.endswith('.xmodel'):
            xmodel_file = os.path.join(xmodel_dir, f)
            break

    if xmodel_file is None:
        print(f"ERROR: No .xmodel in {xmodel_dir}")
        print(f"Contents: {sorted(os.listdir(xmodel_dir))}")
        sys.exit(1)

    print(f"xmodel: {xmodel_file}")

    # Find arch file — try several locations
    arch_file = None
    candidates = [
        'arch_zu3_b512.json',                    # same dir (vitis_ai/)
        '../vitis_ai/arch_zu3_b512.json',        # from project root
        '/workspace/project/vitis_ai/arch_zu3_b512.json',
    ]
    for c in candidates:
        if os.path.exists(c):
            arch_file = c
            break

    if arch_file is None:
        # Create inline
        arch_file = '/tmp/arch_zu3_b512.json'
        with open(arch_file, 'w') as f:
            json.dump({"fingerprint": "0x101000016010400"}, f)
        print(f"Created arch file at {arch_file}")
    else:
        print(f"Arch file: {arch_file}")

    output_dir = './compiled_transformer'
    os.makedirs(output_dir, exist_ok=True)

    compile_cmd = (
        f"vai_c_xir "
        f"-x {xmodel_file} "
        f"-a {arch_file} "
        f"-o {output_dir} "
        f"-n transformer_tiny "
    )

    print(f"\nCompiling for DPU B512...")
    print(f"Command: {compile_cmd}")
    print("=" * 70)

    ret = os.system(compile_cmd + " 2>&1")

    print("=" * 70)
    print(f"Compiler exit code: {ret}")

    if os.path.exists(output_dir):
        print(f"\nCompiled output:")
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f}  ({size} bytes)")

    return output_dir, ret


# ============================================================
# 4. Analyze subgraph partitioning
# ============================================================

def analyze_compiled_model(output_dir):
    """Load compiled xmodel and check DPU vs CPU subgraphs."""
    try:
        import xir
    except ImportError:
        print("\nNOTE: xir not available in this Docker image.")
        print("The compiler log above is the primary evidence.")
        print("You can also analyze on-board with:")
        print("  python3 -c \"import xir; g=xir.Graph.deserialize('transformer_tiny.xmodel'); ...")
        return

    xmodel_file = None
    for f in os.listdir(output_dir):
        if f.endswith('.xmodel'):
            xmodel_file = os.path.join(output_dir, f)
            break

    if xmodel_file is None:
        print(f"No compiled .xmodel in {output_dir}")
        return

    print(f"\n=== Subgraph Analysis ===")
    graph = xir.Graph.deserialize(xmodel_file)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

    dpu_sgs = []
    cpu_sgs = []

    for sg in subgraphs:
        device = sg.get_attr("device")
        name = sg.get_name()
        ops = [op.get_type() for op in sg.get_ops()]
        op_counts = {}
        for t in ops:
            op_counts[t] = op_counts.get(t, 0) + 1

        info = {"name": name, "device": device,
                "ops": op_counts, "num_ops": len(ops)}
        (dpu_sgs if device == "DPU" else cpu_sgs).append(info)

    print(f"Total subgraphs: {len(subgraphs)}")
    print(f"  DPU: {len(dpu_sgs)}, CPU: {len(cpu_sgs)}")

    for label, sgs in [("DPU", dpu_sgs), ("CPU", cpu_sgs)]:
        print(f"\n--- {label} Subgraphs ---")
        for sg in sgs:
            print(f"  {sg['name']} ({sg['num_ops']} ops)")
            for op_type, count in sorted(sg['ops'].items()):
                print(f"    {op_type}: {count}")

    # Key question
    all_cpu_ops = set()
    for sg in cpu_sgs:
        all_cpu_ops.update(sg['ops'].keys())

    print(f"\n=== Key Finding ===")
    if 'softmax' in all_cpu_ops:
        print("CONFIRMED: Softmax partitioned to CPU (not DPU)")
    else:
        print("Softmax not found as separate CPU op — may be fused or renamed")
        print("Check compiler log above for subgraph details")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Vitis AI DPU Transformer Compilation Test")
    print("  Target: DPUCZDX8G B512 (fingerprint 0x101000016010400)")
    print("  Model:  1 encoder layer, 4 heads, D=64, FFN=128, T=32")
    print("=" * 70)

    # Step 1: Create model
    print("\n[1/3] Creating model...")
    model = TinyTransformer()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    {name}: Linear({mod.in_features}, {mod.out_features})")
        elif isinstance(mod, nn.MultiheadAttention):
            print(f"    {name}: MHA(d={mod.embed_dim}, h={mod.num_heads})")
        elif isinstance(mod, nn.LayerNorm):
            print(f"    {name}: LayerNorm({list(mod.normalized_shape)})")

    dummy = torch.randn(1, 32, 64)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Forward pass OK, output shape: {out.shape}")

    # Step 2: Quantize
    print("\n[2/3] Quantizing model...")
    xmodel_dir = quantize_model(model)

    # Step 3: Compile
    print("\n[3/3] Compiling for DPU...")
    output_dir, ret = compile_for_dpu(xmodel_dir)

    # Step 4: Analyze (best-effort)
    analyze_compiled_model(output_dir)

    print("\n" + "=" * 70)
    print("Test complete. Key evidence is in the compiler output above.")
    print("Look for lines showing DPU vs CPU subgraph assignment.")
    if ret != 0:
        print("\nCompilation returned non-zero — check for errors above.")
        print("If vai_c_xir failed, the quantizer output itself is still useful:")
        print("  - Did torch_quantizer trace through attention successfully?")
        print("  - Check quantize_result/ for which ops were quantized")
    print("=" * 70)
