"""
FINN-T model export — standalone version (no DVC dependency).
Reads params.yaml, creates a DummyTransformer, calibrates, exports to ONNX.

Run inside FINN Docker or with brevitas/torch/qonnx installed.
"""
import os
import yaml
import numpy as np
import torch
from tqdm import trange
from brevitas.export import export_qonnx
from model import DummyTransformer

import onnx
from onnx import helper as oh, numpy_helper, TensorProto


def split_fused_key_transpose(model_path):
    """Fix fused key transposes so FINN's InferScaledDotProductAttention matches.

    Brevitas QuantMultiheadAttention applies two transposes to K:
      1) transpose(0,1) — head rearrange: [seq, heads, dim] → [heads, seq, dim]
      2) transpose(-2,-1) — key transpose:  [heads, seq, dim] → [heads, dim, seq]

    Torch's ONNX exporter fuses these into a single Transpose(perm=[1,2,0]).
    FINN's pattern matcher expects a separate Transpose(perm=[0,2,1]) feeding
    the QK MatMul's key input, with the Quant node preceding it.

    This fixup finds the pattern  Transpose(1,2,0) → Quant → MatMul  and
    rewrites it to            Transpose(1,0,2) → Quant → Transpose(0,2,1) → MatMul,
    which is safe because the per-tensor quantizer commutes with transpose.
    """
    m = onnx.load(model_path)
    g = m.graph
    changed = False

    # Index: tensor name → consumer nodes
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    for node in list(g.node):
        if node.op_type != "Transpose":
            continue
        perm = list(node.attribute[0].ints)
        if perm != [1, 2, 0]:
            continue

        # Check if output feeds exactly one Quant node
        quant_nodes = [c for c in consumers.get(node.output[0], [])
                       if c.op_type == "Quant"]
        if len(quant_nodes) != 1:
            continue
        quant = quant_nodes[0]

        # Check if Quant output feeds a MatMul (the QK matmul)
        matmul_nodes = [c for c in consumers.get(quant.output[0], [])
                        if c.op_type == "MatMul"]
        if len(matmul_nodes) != 1:
            continue

        # Rewrite: change this Transpose to perm=[1,0,2] (head rearrange only)
        node.attribute[0].CopyFrom(oh.make_attribute("perm", [1, 0, 2]))

        # Insert a new Transpose(perm=[0,2,1]) between Quant and MatMul
        new_out = quant.output[0] + "_pre_kt"
        kt_node = oh.make_node(
            "Transpose",
            inputs=[quant.output[0]],
            outputs=[new_out],
            perm=[0, 2, 1],
            name=node.name + "_key_transpose",
        )

        # Rewire: MatMul now reads from the new Transpose output
        matmul = matmul_nodes[0]
        for i, inp in enumerate(matmul.input):
            if inp == quant.output[0]:
                matmul.input[i] = new_out

        # Insert the new node right after Quant in the graph
        idx = list(g.node).index(quant)
        g.node.insert(idx + 1, kt_node)
        changed = True

    if changed:
        onnx.save(m, model_path)
        print(f"Fixed fused key transpose(s) in {model_path}")
    else:
        print(f"No fused key transposes found in {model_path}")
    return changed


def seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def is_norm_layer(module):
    norm_layers = {
        torch.nn.modules.batchnorm._NormBase,
        torch.nn.LayerNorm,
    }
    return any(isinstance(module, norm) for norm in norm_layers)


def patch_non_affine_norms(model):
    for name, module in model.named_modules():
        if is_norm_layer(module):
            if hasattr(module, "weight") and module.weight is None:
                if hasattr(module, "running_var"):
                    module.weight = torch.nn.Parameter(
                        torch.ones_like(module.running_var)
                    )
            if hasattr(module, "bias") and module.bias is None:
                if hasattr(module, "running_mean"):
                    module.bias = torch.nn.Parameter(
                        torch.zeros_like(module.running_var)
                    )
    return model


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    seed(params["seed"])
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    model = DummyTransformer(**params["model"])
    seq = params["model"]["seq_len"]
    dim = params["model"]["emb_dim"]

    print(f"Model config: {params['model']}")
    print(f"Input shape: (1, {seq}, {dim})")

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        for _ in trange(0, params["calibration_passes"], desc="calibrating"):
            model(torch.rand(32, seq, dim, device=device))
        model = model.cpu()

    model = patch_non_affine_norms(model)
    model = model.eval()

    x = torch.rand(1, seq, dim)
    o = model(x)

    os.makedirs("outputs/", exist_ok=True)
    np.save("outputs/inp.npy", x.detach().numpy())
    np.save("outputs/out.npy", o.detach().numpy())
    export_qonnx(model, (x,), "outputs/model.onnx", **params["export"])
    print(f"Exported to outputs/model.onnx")

    # Fix fused key transposes for FINN pattern matcher compatibility
    split_fused_key_transpose("outputs/model.onnx")
