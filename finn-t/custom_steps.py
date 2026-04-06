"""Replicate FINN-T build_steps.py step_convert_attention_to_hw ordering."""
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.composed import ComposedTransformation
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.attention import (
    InferScaledDotProductAttention,
    AbsorbMultiThresholdIntoScaledDotProductAttention,
)
from finn.transformation.fpgadataflow.attention_heads import (
    InferMultiHeads,
    UnrollMultiHeadAttention,
    MoveSplitMultiHeadsPastMultiThreshold,
    MoveMergeMultiHeadsPastMultiThreshold,
)
from finn.transformation.squeeze import Squeeze
from finn.transformation.streamline.absorb import AbsorbAddIntoMultiThreshold
from finn.transformation.streamline.reorder import (
    MoveTransposePastFork,
    MoveTransposePastEltwise,
    MoveTransposePastJoinMul,
    MoveTransposePastJoinAdd,
    MoveTransposePastSplit,
    MoveTransposePastJoinConcat,
    MoveSqueezePastMultiThreshold,
    MoveSqueezePastMatMul,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedTranspose
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape,
)


def step_convert_attention_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Exact replication of FINN-T build_steps.py step_convert_attention_to_hw."""
    # Infer multi-head split/merge from Reshape+Transpose patterns
    model = model.transform(InferMultiHeads())
    # Move multi-head splitting past thresholds
    model = model.transform(MoveSplitMultiHeadsPastMultiThreshold())
    # Absorb adds exposed by moving heads past thresholds
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # Infer the fused ScaledDotProductAttention custom op
    model = model.transform(InferScaledDotProductAttention())
    # Unroll multi-head attention into per-head operations
    model = model.transform(UnrollMultiHeadAttention())
    # Move merge past thresholds
    model = model.transform(MoveMergeMultiHeadsPastMultiThreshold())
    # Absorb final thresholds into attention operator
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())
    # Squeeze out batch-1 dimension — this kills the problematic Transpose
    model = model.transform(Squeeze())
    # Clean up transposes and reshapes exposed by squeezing
    model = model.transform(ComposedTransformation([
        MoveTransposePastFork(),
        MoveTransposePastSplit(),
        MoveTransposePastJoinConcat(),
        MoveTransposePastEltwise(),
        MoveTransposePastJoinMul(),
        MoveTransposePastJoinAdd(),
        CollapseRepeatedTranspose(),
        RemoveIdentityTranspose(),
        RemoveIdentityReshape(),
        MoveSqueezePastMatMul(),
        MoveSqueezePastMultiThreshold(),
    ]))
    # Second pass of absorptions after squeezing
    model = model.transform(AbsorbAddIntoMultiThreshold())
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())
    # Second round of streamlining after squeeze (matches FINN-T)
    from finn.transformation.streamline.streamline_plus import StreamlinePlus
    model = model.transform(StreamlinePlus())
    # Fix datatype annotations after Squeeze — ReplicateStream drops them
    model = model.transform(InferDataTypes())
    # Convert Squeeze/Unsqueeze to FINN hardware ops
    from finn.transformation.fpgadataflow.convert_to_hw_layers import InferSqueeze, InferUnsqueeze
    model = model.transform(InferSqueeze())
    model = model.transform(InferUnsqueeze())
    return model.transform(GiveUniqueNodeNames())


def step_convert_elementwise_binary_to_hw(model, cfg):
    """FINN-T: Convert elementwise binary ops to hardware."""
    from finn.transformation.fpgadataflow.convert_to_hw_layers import (
        InferElementwiseBinaryOperation
    )
    return model.transform(InferElementwiseBinaryOperation(
        InferElementwiseBinaryOperation.reject_output_dequant
    ))

def step_replicate_streams(model, cfg):
    """FINN-T: Convert fork nodes to ReplicateStream hardware ops."""
    from finn.transformation.fpgadataflow.replicate_stream import InferReplicateStream
    return model.transform(InferReplicateStream())
