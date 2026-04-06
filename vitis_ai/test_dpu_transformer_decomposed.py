"""Vitis AI DPU Transformer Compilation Test — Decomposed Attention.

The standard nn.MultiheadAttention uses torch.chunk (multi-output op)
which XIR can't represent. This version decomposes attention into
explicit ops to see which individual operations the DPU accepts vs
rejects.

Decomposition:
  - Separate nn.Linear for Q, K, V projections (no chunk needed)
  - Explicit reshape for head splitting (single-output, should be fine)
  - torch.matmul for Q @ K^T
  - Manual softmax
  - torch.matmul for attn_weights @ V
  - Output projection as nn.Linear

Run inside Vitis AI Docker (same as before):
    cd ~/dev/CEN571-final/finn-vs-vitisai
    docker run -it \
      -v $(pwd)/Vitis-AI:/workspace \
      -v $(pwd):/workspace/project \
      xilinx/vitis-ai-pytorch-cpu:latest bash

    conda activate vitis-ai-pytorch
    cd /workspace/project/vitis_ai
    python test_dpu_transformer_decomposed.py

If this also fails, try the --linear-only flag which strips everything
except the linear layers (guaranteed DPU-compatible) to establish a
baseline:
    python test_dpu_transformer_decomposed.py --linear-only
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Model variants
# ============================================================

class DecomposedAttention(nn.Module):
    """Multi-head attention with explicit ops — no torch.chunk."""
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        # Separate projections — avoids chunk
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x):
        batch, seq, _ = x.shape

        # Project Q, K, V separately
        q = self.q_proj(x)  # (batch, seq, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (batch, seq, nhead, d_head) -> (batch, nhead, seq, d_head)
        q = q.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)

        # Attention: Q @ K^T / sqrt(d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum: attn @ V
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, nhead, seq, d_head) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        # Output projection
        return self.out_proj(attn_output)


class DecomposedTransformerBlock(nn.Module):
    """Single transformer encoder block with decomposed attention."""
    def __init__(self, d_model=64, nhead=4, d_ff=128):
        super().__init__()
        self.attn = DecomposedAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention + residual + layernorm
        x = self.norm1(x + self.attn(x))
        # FFN + residual + layernorm
        x = self.norm2(x + self.ff2(F.relu(self.ff1(x))))
        return x


class DecomposedTransformer(nn.Module):
    """Full decomposed transformer for DPU compilation test."""
    def __init__(self, d_model=64, nhead=4, d_ff=128, seq_len=32,
                 num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        self.block = DecomposedTransformerBlock(d_model, nhead, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.block(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class LinearOnlyTransformer(nn.Module):
    """Transformer with ONLY linear layers — strips all non-CNN ops.

    Use this as a baseline: if even this doesn't compile, there's a
    toolchain issue unrelated to transformer ops. If this compiles
    but the decomposed version doesn't, the diff tells you which
    specific ops the DPU can't handle.
    """
    def __init__(self, d_model=64, nhead=4, d_ff=128, seq_len=32,
                 num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        # Q/K/V projections (same as in attention, but no attention matmul)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # FFN
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        # Just run linear layers sequentially — no attention pattern
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.out_proj(q + k + v)  # simple sum instead of attention
        x = self.ff2(F.relu(self.ff1(x)))
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# ============================================================
# Quantize and compile (same logic as original test)
# ============================================================

def quantize_model(model, seq_len=32, d_model=64, calib_batches=20):
    from pytorch_nndct.apis import torch_quantizer

    device = torch.device('cpu')
    model = model.to(device)
    dummy_input = torch.randn(1, seq_len, d_model).to(device)

    output_dir = './quantize_result'
    os.makedirs(output_dir, exist_ok=True)

    # Calibration
    print("  Quantizer (calib mode)...")
    quantizer = torch_quantizer(
        'calib', model, (dummy_input,),
        output_dir=output_dir, device=device,
    )
    quant_model = quantizer.quant_model

    print(f"  Running {calib_batches} calibration batches...")
    with torch.no_grad():
        for _ in range(calib_batches):
            quant_model(torch.randn(8, seq_len, d_model).to(device))
    quantizer.export_quant_config()

    # Export xmodel
    print("  Quantizer (test mode) for export...")
    quantizer = torch_quantizer(
        'test', model, (dummy_input,),
        output_dir=output_dir, device=device,
    )
    with torch.no_grad():
        quantizer.quant_model(dummy_input)

    quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
    return output_dir


def compile_for_dpu(xmodel_dir, model_name="transformer_decomposed"):
    # Find xmodel
    xmodel_file = None
    for f in os.listdir(xmodel_dir):
        if f.endswith('.xmodel'):
            xmodel_file = os.path.join(xmodel_dir, f)
            break

    if xmodel_file is None:
        print(f"  ERROR: No .xmodel in {xmodel_dir}")
        print(f"  Contents: {sorted(os.listdir(xmodel_dir))}")
        print(f"\n  xmodel conversion failed — same class of error as before.")
        print(f"  Check warnings above for which op caused the failure.")
        return None, 1

    print(f"  xmodel: {xmodel_file}")

    # Arch file
    arch_file = None
    for c in ['arch_zu3_b512.json',
              '/workspace/project/vitis_ai/arch_zu3_b512.json']:
        if os.path.exists(c):
            arch_file = c
            break
    if arch_file is None:
        arch_file = '/tmp/arch_zu3_b512.json'
        with open(arch_file, 'w') as f:
            json.dump({"fingerprint": "0x101000016010400"}, f)

    output_dir = f'./compiled_{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    cmd = (f"vai_c_xir -x {xmodel_file} -a {arch_file} "
           f"-o {output_dir} -n {model_name}")

    print(f"  Compiling: {cmd}")
    print("  " + "=" * 66)
    ret = os.system(cmd + " 2>&1")
    print("  " + "=" * 66)
    print(f"  Compiler exit code: {ret}")

    if os.path.exists(output_dir):
        contents = sorted(os.listdir(output_dir))
        if contents:
            print(f"  Output: {contents}")
        else:
            print(f"  Output directory empty")

    return output_dir, ret


def analyze_xmodel(output_dir):
    """Best-effort xmodel analysis."""
    try:
        import xir
    except ImportError:
        print("  (xir not available — use compiler log above)")
        return

    xmodel_file = None
    for f in os.listdir(output_dir):
        if f.endswith('.xmodel'):
            xmodel_file = os.path.join(output_dir, f)
            break
    if not xmodel_file:
        return

    graph = xir.Graph.deserialize(xmodel_file)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

    print(f"\n  === Subgraph Analysis ===")
    for sg in subgraphs:
        device = sg.get_attr("device")
        name = sg.get_name()
        ops = {}
        for op in sg.get_ops():
            t = op.get_type()
            ops[t] = ops.get(t, 0) + 1
        print(f"  [{device}] {name} ({sum(ops.values())} ops)")
        for op_type, count in sorted(ops.items()):
            print(f"        {op_type}: {count}")


# ============================================================
# Main
# ============================================================

def run_variant(name, model):
    """Run quantize + compile for one model variant."""
    print(f"\n{'=' * 70}")
    print(f"  Variant: {name}")
    print(f"{'=' * 70}")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # List layers
    for lname, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    {lname}: Linear({mod.in_features}, {mod.out_features})")
        elif isinstance(mod, nn.LayerNorm):
            print(f"    {lname}: LayerNorm({list(mod.normalized_shape)})")

    # Verify forward pass
    dummy = torch.randn(1, 32, 64)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Forward pass OK, output: {out.shape}")

    # Clean previous quantize results
    import shutil
    if os.path.exists('./quantize_result'):
        shutil.rmtree('./quantize_result')

    # Quantize
    print(f"\n  --- Quantizing ---")
    try:
        xmodel_dir = quantize_model(model)
    except Exception as e:
        print(f"  Quantization FAILED: {e}")
        return False

    # Check if xmodel was produced
    has_xmodel = any(f.endswith('.xmodel') for f in os.listdir(xmodel_dir))
    if not has_xmodel:
        print(f"\n  xmodel conversion FAILED — no .xmodel produced")
        print(f"  Files in quantize_result: {sorted(os.listdir(xmodel_dir))}")
        return False

    # Compile
    print(f"\n  --- Compiling for DPU ---")
    output_dir, ret = compile_for_dpu(xmodel_dir, model_name=name)

    if output_dir and ret == 0:
        analyze_xmodel(output_dir)

    return ret == 0


if __name__ == "__main__":
    print("=" * 70)
    print("  Vitis AI DPU Transformer — Decomposed Attention Test")
    print("  Target: DPUCZDX8G B512")
    print("=" * 70)

    linear_only = "--linear-only" in sys.argv

    if linear_only:
        # Just run the linear-only baseline
        model = LinearOnlyTransformer()
        run_variant("linear_only", model)
    else:
        # Run decomposed transformer first
        success = run_variant("decomposed", DecomposedTransformer())

        if not success:
            # If decomposed failed, try linear-only as baseline
            print("\n\n" + "#" * 70)
            print("  Decomposed model failed. Trying linear-only baseline")
            print("  to isolate whether the issue is transformer-specific.")
            print("#" * 70)
            run_variant("linear_only", LinearOnlyTransformer())

    print("\n" + "=" * 70)
    print("  Test complete.")
    print("  Compare the results above to identify which ops block DPU mapping.")
    print("=" * 70)
