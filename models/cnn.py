"""
CNN model definitions for overlay vs dataflow comparison.
Same architecture for both VTA, Vitis AI (standard PyTorch) and FINN (Brevitas).

For fair comparison, all tools must use the same architecture.
FINN is the limiting factor for model size at INT8.
"""
import torch
import torch.nn as nn

def get_cnn_config(size='tiny'):
    """Return channel sizes for a given configuration."""
    configs = {
        'tiny':   [8, 16],          # Fits FINN at INT8
        'small':  [16, 32],         
        'medium': [32, 64],
        'deep_3': [16, 32, 64],
        'large':  [32, 64, 128],    
    }
    if size not in configs:
        raise ValueError(f"Unknown config '{size}'. Options: {list(configs.keys())}")
    return configs[size]


class CNN(nn.Module):
    """Standard PyTorch CNN for Vitis AI post-training quantization."""
    def __init__(self, in_channels=3, num_classes=10, channels=None):
        super().__init__()
        if channels is None:
            channels = get_cnn_config('tiny')
        
        features = []
        prev_ch = in_channels
        for ch in channels:
            features.extend([
                nn.Conv2d(prev_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            prev_ch = ch
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_ch, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


try:
    import brevitas.nn as qnn
    from brevitas.quant import (
        Int8WeightPerTensorFloat,
        Int8WeightPerChannelFloat,
        Uint8ActPerTensorFloat,
    )
    
    class CNN_Brevitas(nn.Module):
        """Brevitas CNN for FINN quantization-aware training.
        Weights: INT8 signed. Activations after ReLU: UINT8 unsigned (FINN requirement).
        """
        def __init__(self, in_channels=3, num_classes=10, channels=None):
            super().__init__()
            if channels is None:
                channels = get_cnn_config('tiny')

            features = []
            prev_ch = in_channels
            for ch in channels:
                features.extend([
                    qnn.QuantConv2d(prev_ch, ch, 3, padding=1, bias=False,
                                  weight_quant=Int8WeightPerTensorFloat),
                    nn.BatchNorm2d(ch),
                    qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat),
                    nn.MaxPool2d(2),
                ])
                prev_ch = ch
            features.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                qnn.QuantLinear(prev_ch, num_classes, bias=True,
                              weight_quant=Int8WeightPerTensorFloat),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class CNN_Brevitas_INT4(nn.Module):
        """Brevitas CNN at INT4 for QAT retrain.

        Same topology as CNN_Brevitas: Conv→BN→ReLU→MaxPool blocks, AvgPool,
        Linear classifier. BN modules stay as plain nn.BatchNorm2d (float,
        unquantized) - deploy path handles BN on CPU post-VTA. 
        Extraction reads unfolded BN params from the state_dict.
        """
        def __init__(self, in_channels=3, num_classes=10, channels=None):
            super().__init__()
            if channels is None:
                channels = get_cnn_config('tiny')

            Int4W = Int8WeightPerTensorFloat.let(bit_width=4)
            Uint4A = Uint8ActPerTensorFloat.let(bit_width=4)

            features = []
            prev_ch = in_channels
            for ch in channels:
                features.extend([
                    qnn.QuantConv2d(prev_ch, ch, 3, padding=1, bias=False,
                                  weight_quant=Int4W),
                    nn.BatchNorm2d(ch),
                    qnn.QuantReLU(act_quant=Uint4A),
                    nn.MaxPool2d(2),
                ])
                prev_ch = ch
            features.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                qnn.QuantLinear(prev_ch, num_classes, bias=True,
                              weight_quant=Int4W),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class CNN_Brevitas_INT4_NoBN(nn.Module):
        """Brevitas CNN at INT4 with NO batch normalization.

        BN removed from both feature blocks. Architecture:
            Conv2d(in, c1, 3, pad=1, bias=True) -> QuantReLU -> MaxPool(2) ->
            Conv2d(c1, c2, 3, pad=1, bias=True) -> QuantReLU -> MaxPool(2) ->
            AdaptiveAvgPool2d(1) -> Flatten -> QuantLinear(c2, num_classes).

        Motivation: BN is the root cause of the CNN INT4
        VTA deploy failure. VTA's per-tensor SHR + 16-level int4 DMA output
        cannot accommodate BN's per-channel γ/sqrt(var+eps) magnitude spread
        MLP INT4 works at 93.08% because MLP has no BN; FINN-T deploys at 
        INT4 with `norm=none`.
        Removing BN trades some expressivity (tiny CNN relies on BN for
        training stability) for VTA deployability via the MLP INT4 pattern.

        Bias=True on conv since BN's β is no longer available as the de-facto
        bias. Per-tensor weight quantization (Int8WeightPerTensorFloat at
        bit_width=4) matches MLP INT4 recipe. Per-channel scales no longer
        needed without the BN fold magnitude spread.
        """
        def __init__(self, in_channels=3, num_classes=10, channels=None):
            super().__init__()
            if channels is None:
                channels = get_cnn_config('tiny')

            Int4W = Int8WeightPerTensorFloat.let(bit_width=4)
            Uint4A = Uint8ActPerTensorFloat.let(bit_width=4)

            features = []
            prev_ch = in_channels
            for ch in channels:
                features.extend([
                    qnn.QuantConv2d(prev_ch, ch, 3, padding=1, bias=True,
                                  weight_quant=Int4W),
                    qnn.QuantReLU(act_quant=Uint4A),
                    nn.MaxPool2d(2),
                ])
                prev_ch = ch
            features.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                qnn.QuantLinear(prev_ch, num_classes, bias=True,
                              weight_quant=Int4W),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class CNN_Brevitas_INT4_NoBN_Wide(nn.Module):
        """Wider-channel variant of CNN_Brevitas_INT4_NoBN: channels [16, 32].

        Same pipeline and quantizers as CNN_Brevitas_INT4_NoBN, with channel
        counts doubled. Motivation: 1420-param tiny CNN may not have enough
        capacity/redundancy to absorb per-tensor SHR quantization noise at
        INT4. A wider model gives more parameters and potentially more
        scale-robustness per channel.

        Conv2 reduction dim grows from 72 (8*3*3) to 144 (16*3*3), which
        tiles as n=9 at BLOCK_IN=16 (vs n=5 for tiny). The INT4 sweep
        established n=5 threshold at o≤96, and n=1 at o≥256;
        n=9 threshold is untested. For sim, not a concern -- only matters
        for future board deploy.
        """
        def __init__(self, in_channels=3, num_classes=10, channels=None):
            super().__init__()
            if channels is None:
                channels = [16, 32]

            Int4W = Int8WeightPerTensorFloat.let(bit_width=4)
            Uint4A = Uint8ActPerTensorFloat.let(bit_width=4)

            features = []
            prev_ch = in_channels
            for ch in channels:
                features.extend([
                    qnn.QuantConv2d(prev_ch, ch, 3, padding=1, bias=True,
                                  weight_quant=Int4W),
                    qnn.QuantReLU(act_quant=Uint4A),
                    nn.MaxPool2d(2),
                ])
                prev_ch = ch
            features.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                qnn.QuantLinear(prev_ch, num_classes, bias=True,
                              weight_quant=Int4W),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class CNN_Brevitas_INT4_PerChan(nn.Module):
        """Brevitas CNN at INT4 with PER-CHANNEL weight quantization.

        Single change vs CNN_Brevitas_INT4: weight quantizer is
        Int8WeightPerChannelFloat (per-output-channel scale) instead of
        Int8WeightPerTensorFloat. Per-channel scales are required at INT4 CNN
        because BN gamma/sqrt(var+eps) varies up to ~6.5x (Conv1) / ~23x (Conv2)
        across output channels in the tiny [8,16] MNIST model. After post-train
        merge_bn folds BN into conv weights, per-tensor scales lose precision
        on small-fold-scale channels (3a result: 88.42% -> 80.02% post-fold).
        Per-channel scales absorb the per-channel BN spread directly: each
        output channel gets its own scale via per-channel max-abs.

        BN modules remain plain nn.BatchNorm2d during training; merge_bn is
        applied post-training. The per-channel weight quantizer's
        init_tensor_quant() then re-initializes scales as per-channel max-abs
        of the folded weights (verified).

        Use this class for FINN INT4 CNN training -- the per-channel weight
        quantization decision is precision-driven, not deploy-target-driven.
        """
        def __init__(self, in_channels=3, num_classes=10, channels=None):
            super().__init__()
            if channels is None:
                channels = get_cnn_config('tiny')

            Int4WPerCh = Int8WeightPerChannelFloat.let(bit_width=4)
            Uint4A = Uint8ActPerTensorFloat.let(bit_width=4)

            features = []
            prev_ch = in_channels
            for ch in channels:
                features.extend([
                    qnn.QuantConv2d(prev_ch, ch, 3, padding=1, bias=False,
                                  weight_quant=Int4WPerCh),
                    nn.BatchNorm2d(ch),
                    qnn.QuantReLU(act_quant=Uint4A),
                    nn.MaxPool2d(2),
                ])
                prev_ch = ch
            features.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                qnn.QuantLinear(prev_ch, num_classes, bias=True,
                              weight_quant=Int4WPerCh),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

except ImportError:
    pass
