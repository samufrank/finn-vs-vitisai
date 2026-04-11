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

def _read_model_dims():
    """Read seq_len and emb_dim from params.yaml in cwd."""
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params["model"]["seq_len"], params["model"]["emb_dim"]


def step_set_fifo_depths(model, cfg):
    """
    FINN-T set_fifo_depths adapted for ZU3EG (no URAM -> BRAM).
    Sources seq_len from params.yaml.
    """
    import yaml
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
    from finn.transformation.fpgadataflow.set_fifo_depths import (
        RemoveShallowFIFOs, SplitLargeFIFOs,
    )
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
    from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
    from custom.apply_config import ApplyConfig

    # finn-plus --start passes a PosixPath, not a loaded model
    if not hasattr(model, "graph"):
        model = ModelWrapper(str(model))

    seq_len, _emb_dim = _read_model_dims()
    uram_threshold = seq_len  # matches FINN-T build.py default

    # --- Pass 1: per-node FIFO depth hints for attention and residuals ---
    for node in model.graph.node:
        inst = getCustomOp(node)
        in_depths = inst.get_nodeattr("inFIFODepths")
        out_depths = inst.get_nodeattr("outFIFODepths")
        num_inputs = len(node.input)
        num_outputs = len(node.output)

        if in_depths == [2] and num_inputs > 1:
            in_depths = num_inputs * [2]
        if out_depths == [2] and num_outputs > 1:
            out_depths = num_outputs * [2]

        # Attention: each folded input stream needs full buffering
        if node.op_type == "ScaledDotProductAttention_hls":
            in_depths = [inst.get_number_input_values(i) for i in range(num_inputs)]

        # Residual joins: buffer T^2 cycles on both branches
        if node.op_type == "ElementwiseAdd_hls" and model.is_join_node(node):
            in_depths = [seq_len ** 2, seq_len ** 2]

        # ReplicateStream: deepen output FIFOs to prevent fork-join deadlock
        if node.op_type == "ReplicateStream_hls":
            out_depths = [seq_len ** 2] * num_outputs

        # Cast to native Python int — FINN's set_nodeattr rejects numpy ints
        inst.set_nodeattr("inFIFODepths", [int(x) for x in in_depths])
        inst.set_nodeattr("outFIFODepths", [int(x) for x in out_depths])

    # --- Insert DWCs/FIFOs and re-specialize ---
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # --- Re-apply folding config (clobbered by InsertFIFO/SpecializeLayers) ---
    if cfg.folding_config_file is not None:
        with open(cfg.folding_config_file, "r") as f:
            folding_cfg = yaml.safe_load(f)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(folding_cfg))

    # --- Pass 2: deep FIFOs go to BRAM (NOT URAM — ZU3EG has 0 URAM) ---
    for node in model.graph.node:
        if node.op_type == "StreamingFIFO_rtl":
            inst = getCustomOp(node)
            if inst.get_nodeattr("depth") >= uram_threshold:
                inst.set_nodeattr("impl_style", "vivado")
                inst.set_nodeattr("ram_style", "block")  # CHANGED from "ultra"

    # --- Save final hw config for reproducibility ---
    hw_attrs = {
        "PE", "SIMD", "parallel_window", "ram_style", "ram_style_thresholds",
        "ram_style_mask", "depth", "impl_style", "resType", "mac_resource",
        "mem_mode", "runtime_writeable_weights", "inFIFODepths", "outFIFODepths",
        "depth_trigger_uram", "depth_trigger_bram",
    }
    out_config = {"defaults": {}}
    for node in model.graph.node:
        inst = getCustomOp(node)
        out_config[node.name] = {}
        for key in hw_attrs:
            try:
                out_config[node.name][key] = inst.get_nodeattr(key)
            except AttributeError:
                pass
        if not out_config[node.name]:
            del out_config[node.name]

    with open(str(cfg.output_dir) + "/final_hw_config.yaml", "w") as f:
        yaml.safe_dump(out_config, f)

    # --- FIFO post-processing ---
    if getattr(cfg, "split_large_fifos", False):
        model = model.transform(SplitLargeFIFOs())
    model = model.transform(RemoveShallowFIFOs())

    # --- Re-synthesize newly inserted FIFOs/DWCs ---
    model = model.transform(
        PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
    )
    model = model.transform(HLSSynthIP())

    return model
