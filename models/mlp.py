"""
MLP model definitions for overlay vs dataflow comparison.
Same architecture for both VTA, Vitis AI (standard PyTorch) and FINN (Brevitas).

For fair comparison, all tools must use the same architecture.
FINN is the limiting factor for model size at INT8.
"""
import torch
import torch.nn as nn


def get_mlp_config(size='tiny'):
    """Return hidden layer sizes for a given configuration."""
    configs = {
        'tiny':         [64, 32],       # Fits FINN at INT8 and INT4
        'tiny_plus':    [96, 48],                 
        'small':        [128, 64],      
        'small_plus':   [192, 96],      
        'medium':       [256, 128],
        'large':        [512, 256],
        'original':     [256, 256, 128], 
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

    class MLP_Brevitas_INT4(nn.Module):
        """Brevitas MLP at INT4 for QAT retrain."""
        def __init__(self, input_size=784, num_classes=10, hidden_sizes=None):
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = get_mlp_config('tiny')
            
            Int4W = Int8WeightPerTensorFloat.let(bit_width=4)
            Uint4A = Uint8ActPerTensorFloat.let(bit_width=4)
            
            layers = [nn.Flatten()]
            prev = input_size
            for h in hidden_sizes:
                layers.append(qnn.QuantLinear(prev, h, bias=True, weight_quant=Int4W))
                layers.append(qnn.QuantReLU(act_quant=Uint4A))
                prev = h
            layers.append(qnn.QuantLinear(prev, num_classes, bias=True, weight_quant=Int4W))
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layers(x)

except ImportError:
    pass
