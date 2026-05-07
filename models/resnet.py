"""
ResNet-8 model definitions for recognized-benchmark cross-framework comparison.

MLPerf Tiny canonical ResNet-8 for CIFAR-10 (Banbury et al. 2021):
- 3x3 stem conv (3->16) at 32x32
- 3 residual stages, 1 block each (Conv-BN-ReLU-Conv-BN + skip, ReLU after add)
- Stage 1: 16ch, stride=1, identity skip
- Stage 2: 32ch, stride=2 in conv1, 1x1 downsample skip
- Stage 3: 64ch, stride=2 in conv1, 1x1 downsample skip
- GlobalAvgPool -> FC(64, 10)
~78K parameters. Reference target: 85% top-1 on CIFAR-10.

Brevitas variant uses INT8 weights per-tensor + UINT8 ReLU activations,
matching the CNN_Brevitas idioms in cnn.py. Skip-add is plain `+` between
QuantTensors; FINN's streamlining decides whether to absorb it into
dataflow or partition to CPU.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Plain PyTorch residual block: Conv-BN-ReLU-Conv-BN + skip, ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu2(out)
        return out


class ResNet8(nn.Module):
    """Plain PyTorch ResNet-8 for Vitis AI post-training quantization."""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = BasicBlock(16, 16, stride=1)
        self.stage2 = BasicBlock(16, 32, stride=2)
        self.stage3 = BasicBlock(32, 64, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

    class BasicBlock_Brevitas(nn.Module):
        """Brevitas residual block: QuantConv-BN-QuantReLU-QuantConv-BN + skip, QuantReLU."""
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = qnn.QuantConv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                                         bias=False, weight_quant=Int8WeightPerTensorFloat)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.relu1 = qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat)
            self.conv2 = qnn.QuantConv2d(out_ch, out_ch, 3, stride=1, padding=1,
                                         bias=False, weight_quant=Int8WeightPerTensorFloat)
            self.bn2 = nn.BatchNorm2d(out_ch)
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    qnn.QuantConv2d(in_ch, out_ch, 1, stride=stride, bias=False,
                                    weight_quant=Int8WeightPerTensorFloat),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.downsample = nn.Identity()
            self.relu2 = qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat)

        def forward(self, x):
            identity = self.downsample(x)
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + identity
            out = self.relu2(out)
            return out

    class ResNet8_Brevitas(nn.Module):
        """Brevitas ResNet-8 for FINN/VTA QAT.

        Weights: INT8 signed per-tensor. Activations after ReLU: UINT8 unsigned.
        Skip-add is plain `+` between QuantTensors; the post-add QuantReLU
        requantizes the sum. FINN's streamlining will reveal whether this
        residual structure can be absorbed into streaming dataflow or must
        partition to CPU.
        """
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            self.stem_conv = qnn.QuantConv2d(in_channels, 16, 3, stride=1, padding=1,
                                             bias=False, weight_quant=Int8WeightPerTensorFloat)
            self.stem_bn = nn.BatchNorm2d(16)
            self.stem_relu = qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat)
            self.stage1 = BasicBlock_Brevitas(16, 16, stride=1)
            self.stage2 = BasicBlock_Brevitas(16, 32, stride=2)
            self.stage3 = BasicBlock_Brevitas(32, 64, stride=2)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = qnn.QuantLinear(64, num_classes, bias=True,
                                      weight_quant=Int8WeightPerTensorFloat)

        def forward(self, x):
            x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.gap(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

except ImportError:
    pass
