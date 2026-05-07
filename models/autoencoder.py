"""FC autoencoder for MLPerf Tiny anomaly detection (DCASE 2020 ToyCar).

Default architecture: MLPerf Tiny v0.5 canonical
  640 → 128 → 128 → 128 → 128 → 8 → 128 → 128 → 128 → 128 → 640
  (4 encoder × 128, 8-dim bottleneck, 4 decoder × 128, output 640)
~268K parameters. Per-recording reconstruction MSE → AUC-ROC.

Input: standardized log-mel feature vectors (640-dim). Standardization (mean
and std computed on the training set) is applied OUTSIDE the model.

Output: raw reconstruction (no ReLU on the output) so reconstruction MSE
sees the full signed range.

Pass `hidden_dims=(128, 128, 128)` to recover the earlier 4-layer variant.
"""
import torch
import torch.nn as nn


CANONICAL_HIDDEN = (128, 128, 128, 128, 8, 128, 128, 128, 128)


class FCAutoencoder(nn.Module):
    """Plain PyTorch FC autoencoder for Vitis AI PTQ flow.

    BatchNorm1d + ReLU between every Linear except after the output layer.
    """
    def __init__(self, input_dim=640, hidden_dims=CANONICAL_HIDDEN):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, input_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

    class FCAutoencoder_Brevitas(nn.Module):
        """Brevitas FC autoencoder for FINN/VTA QAT.

        INT8 weights per-tensor, UINT8 ReLU activations. Mirrors the existing
        MLP_Brevitas class's idiom (no input QuantIdentity — FINN auto-inserts
        input quantization at compile time). Output QuantLinear has no
        post-activation so the reconstruction can take signed values.
        """
        def __init__(self, input_dim=640, hidden_dims=CANONICAL_HIDDEN):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(qnn.QuantLinear(prev, h, bias=True,
                                              weight_quant=Int8WeightPerTensorFloat))
                layers.append(nn.BatchNorm1d(h))
                layers.append(qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat))
                prev = h
            layers.append(qnn.QuantLinear(prev, input_dim, bias=True,
                                          weight_quant=Int8WeightPerTensorFloat))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

except ImportError:
    pass
