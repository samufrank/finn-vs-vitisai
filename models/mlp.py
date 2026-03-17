"""
MLP model definitions for overlay vs dataflow comparison.
Same architecture for both Vitis AI (standard PyTorch) and FINN (Brevitas).

Confirmed working configurations on KV260:
  - FINN INT8: 784->64->32->10 (fits)
  - FINN INT8: 784->256->256->128->10 (does NOT fit, BRAM overflow)
  - Vitis AI INT8: 784->256->256->128->10 (fits, DPU stores weights in DDR)
  - Vitis AI INT8: 3072->256->256->128->10 (fits)

For fair comparison, both tools must use the same architecture.
FINN is the limiting factor for model size at INT8.
"""
import torch
import torch.nn as nn


def get_mlp_config(size='tiny'):
    """Return hidden layer sizes for a given configuration."""
    configs = {
        'tiny':     [64, 32],        # Fits FINN INT8 on KV260
        'small':    [128, 64],       # Untested on FINN
        'medium':   [256, 128],      # Does NOT fit FINN INT8 on KV260
        'original': [256, 256, 128], # Does NOT fit FINN INT8 on KV260
    }
    if size not in configs:
        raise ValueError(f"Unknown config '{size}'. Options: {list(configs.keys())}")
    return configs[size]


class MLP(nn.Module):
    """Standard PyTorch MLP for Vitis AI post-training quantization."""
    def __init__(self, input_size=784, num_classes=10, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = get_mlp_config('tiny')
        
        layers = [nn.Flatten()]
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
    
    class MLP_Brevitas(nn.Module):
        """Brevitas MLP for FINN quantization-aware training.
        Weights: INT8 signed. Activations after ReLU: UINT8 unsigned (FINN requirement).
        """
        def __init__(self, input_size=784, num_classes=10, hidden_sizes=None):
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = get_mlp_config('tiny')
            
            layers = [nn.Flatten()]
            prev = input_size
            for h in hidden_sizes:
                layers.append(qnn.QuantLinear(prev, h, bias=True,
                             weight_quant=Int8WeightPerTensorFloat))
                layers.append(qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat))
                prev = h
            layers.append(qnn.QuantLinear(prev, num_classes, bias=True,
                         weight_quant=Int8WeightPerTensorFloat))
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layers(x)

except ImportError:
    pass
