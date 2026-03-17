"""
CNN model definitions for overlay vs dataflow comparison.
Same architecture for both Vitis AI (standard PyTorch) and FINN (Brevitas).

Confirmed working configurations on KV260:
  - FINN INT8: channels=[8, 16], no FC hidden layer (fits)
  - FINN INT8: channels=[32, 64, 128] (does NOT fit, BRAM + LUT overflow)
  - Vitis AI INT8: untested yet, expected to work at any size (DPU stores weights in DDR)

For fair comparison, both tools must use the same architecture.
FINN is the limiting factor for model size at INT8.
"""
import torch
import torch.nn as nn


def get_cnn_config(size='tiny'):
    """Return channel sizes for a given configuration."""
    configs = {
        'tiny':   [8, 16],          # Fits FINN INT8 on KV260
        'small':  [16, 32],         # Untested on FINN
        'medium': [32, 64],         # Untested on FINN
        'large':  [32, 64, 128],    # Does NOT fit FINN INT8 on KV260
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
    from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
    
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

except ImportError:
    pass
