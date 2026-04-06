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
