"""
FINN-deployable ResNet-8 for CIFAR-10 (parallel track; mainline FINN @ 8ac41e46).

This is a FORK of models/resnet.py's ResNet8_Brevitas, rebuilt to satisfy FINN's
residual-add constraints. It does NOT replace resnet.py (canonical). The structural
fix follows AMD's reference QuantBasicBlock
(deps/brevitas/src/brevitas_examples/bnn_pynq/models/resnet.py, brevitas 0.10.0):

  - A matched, UNSIGNED QuantReLU terminates BOTH lanes *before* the add
    (relu2 on the main lane; the downsample's trailing QuantReLU on the skip lane).
  - That terminating QuantReLU is a SHARED INSTANCE threaded across blocks, so both
    add operands carry the same scale and same sign  -> InferAddStreamsLayer's
    idt0==idt1 + is_integer() guards pass.
  - return_quant_tensor=True everywhere on the residual path; the add is asserted to
    be between QuantTensors.
  - A separate relu_out QuantReLU AFTER the add (this is the "extra" relu AMD adds for
    FINN; our canonical resnet.py had only the post-add relu, which is why its joins
    stayed float).

Deviations from AMD, deliberate:
  - Topology is OUR ResNet-8: stem 3->16, three stages 16/32/64, one block each,
    strides 1/2/2, GAP over an 8x8 map, FC 64->10  (AMD's class is ResNet-18-shaped).
  - TruncAvgPool kernel = 8 (our pre-pool map is 8x8), not AMD's hardcoded 4.
  - An input QuantIdentity(bit_width=8) is prepended (cnn.py:82-85 idiom) so the stem
    conv input edge is int-annotated and the stem lands on fabric. AMD assumes a
    pre-quantized 8b input and has no input quantizer.
  - Per-TENSOR INT8 weights (Int8WeightPerTensorFloat), matching our cnn.py/mlp.py
    baseline and the matched-INT8 comparison intent (AMD defaults to per-channel).

Accuracy is NOT the point of this file; structural FINN-compilability is. See
docs/resnet8_finn_recon.md and the Gate 1/2 session narrative.
"""
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import (
    Int8WeightPerTensorFloat,
    Int8ActPerTensorFloat,
    IntBias,
    TruncTo8bit,
)
from brevitas.quant_tensor import QuantTensor

ACT_BIT = 8
WEIGHT_BIT = 8
WEIGHT_QUANT = Int8WeightPerTensorFloat


def _conv(in_ch, out_ch, k, stride=1, padding=0, bias=False):
    return qnn.QuantConv2d(
        in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=bias,
        weight_quant=WEIGHT_QUANT, weight_bit_width=WEIGHT_BIT)


class QuantBasicBlock(nn.Module):
    """Residual block with FINN-correct same-sign, shared-scale residual add.

    Mirrors AMD's QuantBasicBlock construction (incl. the shared_quant_act trick).
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shared_quant_act=None):
        super().__init__()
        self.conv1 = _conv(in_planes, planes, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=ACT_BIT, return_quant_tensor=True)
        self.conv2 = _conv(planes, planes, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                _conv(in_planes, self.expansion * planes, 1, stride=stride, padding=0),
                nn.BatchNorm2d(self.expansion * planes),
                # same-sign requirement on residual adds -> terminate skip lane in ReLU
                qnn.QuantReLU(bit_width=ACT_BIT, return_quant_tensor=True),
            )
            # main-lane pre-add quantizer SHARES the downsample's QuantReLU instance
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=ACT_BIT, return_quant_tensor=True)
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(bit_width=ACT_BIT, return_quant_tensor=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if len(self.downsample):
            x = self.downsample(x)
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        out = out + x
        out = self.relu_out(out)
        return out


class ResNet8_Brevitas_FINN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # input quantizer (cnn.py:82-85 idiom) so the stem lands on fabric
        self.inp_quant = qnn.QuantIdentity(
            bit_width=8, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

        self.stem_conv = _conv(in_channels, 16, 3, stride=1, padding=1)
        self.stem_bn = nn.BatchNorm2d(16)
        shared = qnn.QuantReLU(bit_width=ACT_BIT, return_quant_tensor=True)
        self.stem_relu = shared

        # one block per stage; thread the shared activation exactly like AMD _make_layer
        self.stage1 = QuantBasicBlock(16, 16, stride=1, shared_quant_act=shared)
        shared = self.stage1.relu_out
        self.stage2 = QuantBasicBlock(16, 32, stride=2, shared_quant_act=shared)
        shared = self.stage2.relu_out
        self.stage3 = QuantBasicBlock(32, 64, stride=2, shared_quant_act=shared)

        # pre-pool feature map is 8x8 -> kernel 8 (FINN-supported truncating avgpool)
        self.final_pool = qnn.TruncAvgPool2d(
            kernel_size=8, trunc_quant=TruncTo8bit, float_to_int_impl_type='FLOOR')
        self.fc = qnn.QuantLinear(
            64, num_classes, weight_bit_width=WEIGHT_BIT, bias=True,
            bias_quant=IntBias, weight_quant=WEIGHT_QUANT)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.inp_quant(x)
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
