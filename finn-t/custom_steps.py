"""Replicate FINN-T build_steps.py step_convert_attention_to_hw ordering."""
import numpy as np
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.composed import ComposedTransformation
from qonnx.util.basic import get_by_name
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


class RemoveIdentityAveragePool(Transformation):
    """Remove AveragePool nodes whose kernel_shape and strides are all 1 with
    no padding. These arise when AdaptiveAvgPool2d's target size equals the
    incoming spatial size (e.g. patches=(1,64) applied to a 1x64 map). The op
    is semantically identity but survives export; after Squeeze reduces its
    input to 2D, ONNX shape inference fails because AveragePool requires rank
    >=3. Running this pre-Squeeze prevents the downstream shape-loss cascade.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        for node in list(graph.node):
            if node.op_type != "AveragePool":
                continue
            ks = get_by_name(node.attribute, "kernel_shape")
            if ks is None or not all(x == 1 for x in ks.ints):
                continue
            st = get_by_name(node.attribute, "strides")
            if st is not None and not all(x == 1 for x in st.ints):
                continue
            pads = get_by_name(node.attribute, "pads")
            if pads is not None and any(x != 0 for x in pads.ints):
                continue
            inp = node.input[0]
            out = node.output[0]
            for other in graph.node:
                for i, t in enumerate(other.input):
                    if t == out:
                        other.input[i] = inp
            for go in graph.output:
                if go.name == out:
                    go.name = inp
            graph.node.remove(node)
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False


class RestoreGAPRank(Transformation):
    """After Squeeze strips size-1 dims from GlobalAveragePool's input, the rank
    may fall below ONNX GAP's required rank >= 3 (NCH/NCHW). When this happens
    ONNX shape inference silently returns the stale pre-Squeeze shape, which
    cascades a shape mismatch into the classifier-head MatMul. Prepend an
    Unsqueeze(axes=[0]) to restore a batch dimension so GAP's expected layout
    holds again, then wipe stale shape annotations downstream of GAP so
    InferShapes re-derives them from the restored rank.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        for node in list(graph.node):
            if node.op_type != "GlobalAveragePool":
                continue
            inp = node.input[0]
            in_shape = model.get_tensor_shape(inp)
            if in_shape is None or len(in_shape) >= 3:
                continue
            new_tensor = inp + "_unsq_for_gap"
            axes_init = inp + "_unsq_axes"
            model.set_initializer(axes_init, np.asarray([0], dtype=np.int64))
            unsq = oh.make_node(
                "Unsqueeze",
                inputs=[inp, axes_init],
                outputs=[new_tensor],
                name=node.name + "_rank_restore",
            )
            node.input[0] = new_tensor
            idx = list(graph.node).index(node)
            graph.node.insert(idx, unsq)
            # Wipe stale annotations on GAP output and everything downstream
            # so InferShapes re-derives them from the restored rank instead of
            # preserving the pre-Unsqueeze (wrong) shapes.
            inits = {i.name for i in graph.initializer}
            to_wipe = {node.output[0]}
            changed = True
            while changed:
                changed = False
                for consumer in graph.node:
                    if any(t in to_wipe for t in consumer.input):
                        for out_t in consumer.output:
                            if out_t not in to_wipe and out_t not in inits:
                                to_wipe.add(out_t)
                                changed = True
            for t in to_wipe:
                model.set_tensor_shape(t, None)
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False


class AnnotateSDPAOutputDtype(Transformation):
    """Set each ScaledDotProductAttention node's output datatype to integer.

    Two changes are required and it is critical that BOTH are made:
      1. The node's ``OType`` attribute, via ``inst.set_nodeattr("OType", ...)``.
      2. The output tensor's datatype annotation, via
         ``model.set_tensor_datatype(node.output[i], ...)``.

    Annotating only the tensor is NOT sufficient. SDPA's ``infer_node_datatype``
    (finn/custom_op/fpgadataflow/attention.py, in the FINN+ 1.4.0 sources the
    call is at the end of that method, around line 282) always ends with::

        model.set_tensor_datatype(node.output[0], DataType[self.get_nodeattr('OType')])

    That line overwrites any manual tensor-datatype annotation on the next
    call to ``InferDataTypes``. Since ``step_convert_to_hw`` runs
    ``InferDataTypes`` multiple times (directly and via per-pass
    post-processing), any tensor-only annotation is reverted to whatever the
    node's ``OType`` attribute says. Setting the attribute itself makes the
    change persistent.

    Why this matters here
    ---------------------
    The trained transformer has ``output_quant=None`` on Brevitas'
    ``QuantMultiheadAttention``, so the exported SDPA gets ``OType='FLOAT32'``
    even though the values flowing out of attention are integer-valued
    (everything upstream was quantized to ``bits=4``/``bits=8``). Downstream
    ``StreamingConcat.get_output_datatype()`` (finn/custom_op/fpgadataflow/
    concat.py around line 114) reads each input tensor's datatype and, for
    FLOAT32, computes::

        max_abs_input = max(-FLOAT32.min(), 1 + FLOAT32.max()) ≈ 3.4e38
        out_bit_width = ceil(log2(3.4e38) + 1) = 129
        odt = DataType['INT129']

    INT129 then propagates to the next MultiThreshold/Thresholding's input
    dtype annotation, and ``RoundAndClipThresholds`` (finn/transformation/
    streamline/round_thresholds.py line 86) crashes with::

        Exception: Could not find a suitable int datatype for
                   -340282366920938463463374607431768211457

    because ``-(2**128) - 1`` has no representation in any signed int type
    ≤ 64 bits.

    Earlier investigation wrongly attributed this to MVAU accumulator
    widening and looked for emb_dim sensitivity; actually the cascade
    originates in StreamingConcat's handling of a FLOAT32 input, and
    emb_dim does not appear in its width formula. Fixing SDPA's ``OType``
    attribute upstream of StreamingConcat resolves it at any emb_dim.

    Upstream fix for future models
    -------------------------------
    Setting ``output_quant=act_quantizer(bits)`` on
    ``QuantMultiheadAttention`` in the trained model would produce a proper
    integer ``OType`` in the exported SDPA, making this pass unnecessary.
    Until then, keep this pass in the pipeline.
    """

    def apply(self, model: ModelWrapper):
        from qonnx.core.datatype import DataType
        from qonnx.custom_op.registry import getCustomOp
        for node in model.graph.node:
            if node.op_type != "ScaledDotProductAttention":
                continue
            inst = getCustomOp(node)
            otype = inst.get_nodeattr("OType")
            if not DataType[otype].is_integer():
                # Set the node attribute — this is what survives across
                # InferDataTypes invocations.
                inst.set_nodeattr("OType", "INT8")
            # Also set the tensor annotation for the immediate subsequent pass
            # (InferConcatLayerIntegerOrFloat) that reads the tensor's current
            # datatype to decide StreamingConcat's inputDataTypes at creation.
            for out in node.output:
                current = model.get_tensor_datatype(out)
                if current is None or not current.is_integer():
                    model.set_tensor_datatype(out, DataType["INT8"])
        return model, False


class InferConcatLayerIntegerOrFloat(Transformation):
    """Local replacement for InferConcatLayer that converts Concat over the
    last axis to StreamingConcat regardless of input datatype. Upstream
    InferConcatLayer skips non-integer inputs, but our trained transformer's
    SDPA custom op annotates its output as FLOAT32 (despite carrying
    integer-valued data), which leaves the attention-head Concat unconverted
    and blocks step_create_dataflow_partition. Annotating the SDPA output as
    INT8/INT32 would cascade into RoundAndClipThresholds producing overflow
    on downstream threshold tensors (extreme float-sentinel values that
    don't fit any signed int relative to the widened input range). Avoiding
    that cascade by simply bypassing the integer check here keeps the rest
    of the graph's dtype annotations intact. The StreamingConcat HLS op
    itself transports whatever bit-width the input stream carries; the
    integer check in the upstream pass was a conservative filter, not a
    hard requirement.
    """

    def apply(self, model: ModelWrapper):
        from onnx import helper
        from qonnx.util.basic import get_by_name
        from qonnx.core.datatype import DataType
        graph = model.graph
        graph_modified = False
        for ni, node in enumerate(list(graph.node)):
            if node.op_type != "Concat":
                continue
            ishape = model.get_tensor_shape(node.input[0])
            axis_attr = get_by_name(node.attribute, "axis")
            if ishape is None or axis_attr is None:
                continue
            axis = axis_attr.i
            last_axis = len(ishape) - 1
            if axis != -1 and axis != last_axis:
                continue
            if any(model.get_initializer(x) is not None for x in node.input):
                continue
            def _dtype_name(t):
                dt = model.get_tensor_datatype(t)
                if dt is None or not dt.is_integer():
                    return "INT8"  # safe default matching the trained model's bit-width
                return dt.name
            channels_per_stream = [model.get_tensor_shape(x)[-1] for x in node.input]
            inp_vec = list(model.get_tensor_shape(node.input[0])[:-1])
            new_node = helper.make_node(
                "StreamingConcat",
                node.input,
                node.output,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                name="StreamingConcat_" + node.name,
                SIMD=1,
                ChannelsPerStream=channels_per_stream,
                inputDataTypes=[_dtype_name(x) for x in node.input],
                numInputVectors=inp_vec,
                inFIFODepths=[2] * len(node.input),
                cpp_interface="hls_vector",
                hls_style="freerunning",
            )
            graph.node.insert(ni + 1, new_node)
            graph.node.remove(node)
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False


class HarmonizeMultiThresholdLayouts(Transformation):
    """Align a MultiThreshold's input and output data_layout annotations when
    they disagree. After streamlining absorbs an input Transpose (e.g.,
    PyTorch's `Rearrange("b h w c -> b c h w")` at the start of a model), the
    MT's `data_layout` attribute and its output annotation reflect the new
    (post-absorption) layout, but the producer's output / graph input can be
    left with the stale pre-absorption layout. FINN's `convert_to_hw_layers`
    rejects MTs whose input/output layouts form NCHW-in/NHWC-out (or the
    reverse) — it can't auto-resolve that combination. This pass harmonizes
    the stale side to match the MT's attribute, but only when the non-stale
    side already agrees with the attribute (so we never flip a genuinely
    distinct layout).
    """

    _STR_TO_LIST = {
        "NHWC": ["N", "H", "W", "C"],
        "NCHW": ["N", "C", "H", "W"],
        "NWC":  ["N", "W", "C"],
        "NC":   ["N", "C"],
    }

    def apply(self, model: ModelWrapper):
        graph_modified = False
        for node in model.graph.node:
            if node.op_type != "MultiThreshold":
                continue
            attr = get_by_name(node.attribute, "data_layout")
            if attr is None:
                continue
            target = self._STR_TO_LIST.get(attr.s.decode())
            if target is None:
                continue
            inp = node.input[0]
            out = node.output[0]
            in_lay = model.get_tensor_layout(inp)
            out_lay = model.get_tensor_layout(out)
            if in_lay != target and out_lay == target:
                model.set_tensor_layout(inp, target)
                graph_modified = True
            elif out_lay != target and in_lay == target:
                model.set_tensor_layout(out, target)
                graph_modified = True
        return model, False


class AnnotateElementwiseOutputShapes(Transformation):
    """Explicitly set Add/Mul/Sub/Div output shapes by broadcasting the input
    shapes when both inputs are shaped but the output annotation is missing.
    qonnx InferShapes relies on ONNX shape_inference, which only fills in
    missing shapes and does not overwrite existing (possibly stale) ones; in
    topologies with intermediate finn custom ops whose shape-compatible
    stand-ins fail to propagate, some plain ONNX Add/Mul outputs end up with
    empty shape annotations even though both inputs are known. That empty
    shape later trips InferElementwiseBinaryOperation when it copies the
    output shape into the new ElementwiseAdd/ElementwiseMul node's out_shape
    attribute (helper.make_attribute fails with an empty iterator). This pass
    fills the gap by computing the broadcasted output shape directly.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            if node.op_type not in ("Add", "Mul", "Sub", "Div"):
                continue
            if len(node.input) != 2:
                continue
            out = node.output[0]
            out_shape = model.get_tensor_shape(out)
            if out_shape is not None and len(out_shape) > 0:
                continue
            s0 = model.get_tensor_shape(node.input[0])
            s1 = model.get_tensor_shape(node.input[1])
            if s0 is None or s1 is None or len(s0) == 0 or len(s1) == 0:
                continue
            try:
                bshape = np.broadcast_shapes(tuple(s0), tuple(s1))
            except ValueError:
                continue
            model.set_tensor_shape(out, list(bshape))
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False


class AnnotateSliceOutputShape(Transformation):
    """Fill in Slice output shape by applying the slice parameters to a dummy
    tensor of the input shape. Same class of problem as
    AnnotateElementwiseOutputShapes: qonnx InferShapes does not overwrite
    stale empty annotations, and an upstream shape loss (e.g. the MLP
    residual Add) that was later papered over with explicit annotation on
    that specific node leaves downstream Slice nodes still shapeless. When
    Reshape is the direct consumer, the Reshape recovers via its static
    target, but FINN's RemoveCNVtoFCFlatten reads Reshape's *input* shape
    (not output) and does `ishape[0]`, which raises IndexError on empty.
    """

    def apply(self, model: ModelWrapper):
        graph_modified = False
        for node in model.graph.node:
            if node.op_type != "Slice":
                continue
            out = node.output[0]
            out_shape = model.get_tensor_shape(out)
            if out_shape is not None and len(out_shape) > 0:
                continue
            in_shape = model.get_tensor_shape(node.input[0])
            if in_shape is None or len(in_shape) == 0:
                continue
            if len(node.input) < 3:
                continue
            starts = model.get_initializer(node.input[1])
            ends = model.get_initializer(node.input[2])
            if starts is None or ends is None:
                continue
            starts = np.atleast_1d(starts)
            ends = np.atleast_1d(ends)
            axes = (model.get_initializer(node.input[3])
                    if len(node.input) > 3 else None)
            if axes is None:
                axes = np.arange(len(starts), dtype=np.int64)
            else:
                axes = np.atleast_1d(axes)
            steps = (model.get_initializer(node.input[4])
                     if len(node.input) > 4 else None)
            if steps is None:
                steps = np.ones(len(starts), dtype=np.int64)
            else:
                steps = np.atleast_1d(steps)
            slicer = [slice(None)] * len(in_shape)
            for ax, st, en, sp in zip(axes, starts, ends, steps):
                ax = int(ax)
                if ax < 0:
                    ax += len(in_shape)
                slicer[ax] = slice(int(st), int(en), int(sp))
            try:
                new_shape = list(np.zeros(in_shape)[tuple(slicer)].shape)
            except Exception:
                continue
            model.set_tensor_shape(out, new_shape)
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False


class RestoreChannelwiseBroadcast(Transformation):
    """After Squeeze's np.squeeze() has collapsed channelwise [1, C, 1]
    initializers to [C], detect Mul/Add nodes where the resulting 1D init
    cannot broadcast against a 2D [C, W] runtime tensor and reshape the init
    to [C, 1]. [C, 1] broadcasts safely against both [C, W] and [B, C, W], so
    this does not regress higher-rank cases. Runs post-Squeeze to undo the
    broadcast-breaking part of np.squeeze() without disturbing other
    initializers Squeeze correctly collapsed.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        init_names = {i.name for i in graph.initializer}
        for node in graph.node:
            if node.op_type not in ("Mul", "Add"):
                continue
            if len(node.input) != 2:
                continue
            a_name = t_name = None
            for inp in node.input:
                if inp in init_names:
                    a_name = inp
                else:
                    t_name = inp
            if a_name is None or t_name is None:
                continue
            a_val = model.get_initializer(a_name)
            t_shape = model.get_tensor_shape(t_name)
            if a_val is None or t_shape is None or len(t_shape) == 0:
                continue
            if a_val.ndim != 1 or len(t_shape) != 2:
                continue
            if a_val.shape[0] != t_shape[0]:
                continue
            try:
                np.broadcast_shapes(tuple(t_shape), a_val.shape)
                continue
            except ValueError:
                pass
            reshaped = a_val.reshape(a_val.shape[0], 1)
            model.set_initializer(a_name, reshaped)
            model.set_tensor_shape(a_name, list(reshaped.shape))
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False


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
    # Remove identity AveragePool (kernel=strides=1, no pad) before Squeeze.
    # After Squeeze reduces rank, a 2D AveragePool with 2D kernel fails ONNX
    # shape inference and the shape loss cascades to InferUnsqueeze's inp_shape.
    model = model.transform(RemoveIdentityAveragePool())
    # Squeeze out batch-1 dimension — this kills the problematic Transpose
    model = model.transform(Squeeze())
    # Squeeze collapses [1, C, 1] channelwise inits to [C] via np.squeeze,
    # which then can't broadcast against [C, W] runtime tensors. Restore rank
    # to [C, 1] so broadcast semantics survive.
    model = model.transform(RestoreChannelwiseBroadcast())
    # Squeeze may reduce GlobalAveragePool's input below the rank>=3 GAP
    # requires. Prepend an Unsqueeze to bring the rank back up so downstream
    # MatMul at the classifier head can infer its shape.
    model = model.transform(RestoreGAPRank())
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
    # Run again: StreamlinePlus and the CT cleanup above can move a Transpose
    # past a channelwise Add/Mul, which flips the tensor's channel axis and
    # re-breaks the [C] vs [C, W] broadcast that the first RestoreChannelwise-
    # Broadcast call fixed. Re-running here catches whatever was re-introduced.
    model = model.transform(RestoreChannelwiseBroadcast())
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
    # Some Add/Mul nodes reach this step with shapeless output annotations due
    # to prior InferShapes passes not propagating through custom-op stand-ins
    # (e.g., the MLP residual Add). InferElementwiseBinaryOperation copies the
    # output shape into the new HLS node's attribute and fails on empty lists.
    # Explicitly annotate the broadcasted output shape first.
    model = model.transform(AnnotateElementwiseOutputShapes())
    return model.transform(InferElementwiseBinaryOperation(
        InferElementwiseBinaryOperation.reject_output_dequant
    ))

def step_replicate_streams(model, cfg):
    """FINN-T: Convert fork nodes to ReplicateStream hardware ops."""
    from finn.transformation.fpgadataflow.replicate_stream import InferReplicateStream
    from finn.transformation.fpgadataflow.convert_to_hw_layers import (
        InferSplitLayer, InferConcatLayer
    )
    from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
    model = model.transform(InferReplicateStream())
    # Fix 2B (session 19 reorder): run CancelQKVTransposes BEFORE InferSplitLayer.
    # The previous ordering let the stale [96, 96] MT_6/7/8 shape annotation
    # bleed into InferSplitLayer, which baked numInputVectors=[96] into the
    # resulting StreamingSplit_hls nodes and later blew up InsertFIFO (step 18,
    # session 19 diagnosis). CancelQKVTransposes now also rewrites the
    # downstream MT output shapes to [seq, emb]; running it first lets
    # InferSplitLayer read the correct seq dimension when computing
    # StreamingSplit attributes.
    from cancel_qkv_transposes import CancelQKVTransposes
    model = model.transform(CancelQKVTransposes())
    # Standard step_convert_to_hw (the next step in the build) does NOT call
    # InferSplitLayer / InferConcatLayer — those come from upstream's
    # transformer_adhoc.step_convert_to_hw which our build doesn't invoke.
    # Without them, QKV Split and attention-head Concat stay in ONNX domain
    # and step_create_dataflow_partition rejects them as unmapped layers
    # between FINN operators.
    model = model.transform(InferSplitLayer())
    # Fix 1: annotate SDPA outputs as integer BEFORE custom Concat runs, so
    # that StreamingConcat's get_output_datatype sees an integer min/max
    # range (width = ceil(log2(MAX_FLOAT32)+1) = 129 when the annotation is
    # FLOAT32, which was the real source of the earlier RoundAndClipThresholds
    # overflow — not MVAU widening).
    model = model.transform(AnnotateSDPAOutputDtype())
    model = model.transform(InferConcatLayerIntegerOrFloat())
    # Collapse pairs of transposes that cancel (e.g. absorbed-then-reinserted
    # layout conversions around MVAU projections) before the partitioner sees
    # them. Reduces the set of transposes sandwiched between FINN ops.
    model = model.transform(AbsorbConsecutiveTransposes())
    # Last custom step before standard FINN's step_convert_to_hw. That step
    # rejects MTs whose input/output layouts form an NCHW/NHWC mismatch; in
    # the trained PatchEmbedding model a stale NCHW annotation survives on
    # global_in even though the first MT has already absorbed the input
    # Rearrange and processes NHWC. Harmonize here so step_convert_to_hw sees
    # consistent annotations.
    model = model.transform(HarmonizeMultiThresholdLayouts())
    # step_convert_to_hw also runs RemoveCNVtoFCFlatten, which reads each
    # Reshape/Flatten's *input* shape and indexes `ishape[0]`. Slice nodes
    # downstream of a shape-loss point (e.g. the MLP residual Add whose
    # output we annotated explicitly earlier) can still be left shapeless,
    # which propagates into the next Reshape's input. Fill those in.
    model = model.transform(AnnotateSliceOutputShape())
    # Fix 3a: remove Slice nodes that were identity-by-intent but became
    # truncating after Squeeze failed to re-index axes. See remove_stale_slices.py.
    from remove_stale_slices import RemoveStaleSlices
    model = model.transform(RemoveStaleSlices())
    # Fix 3b: cancel the MLP-path Transpose pair (Transpose_4 ↔ Transpose_5)
    # around the channelwise bias Add. See cancel_mlp_transposes.py.
    from cancel_mlp_transposes import CancelMLPTransposePair
    model = model.transform(CancelMLPTransposePair())
    return model


def step_fix_streaming_split_vectors(model, cfg):
    """Re-derive StreamingSplit/Concat numInputVectors from their producers.

    Unblocks InsertFIFO (step 18): the stale [96, 96] tensor annotations left
    by step_convert_attention_to_hw caused InferSplitLayer to compute
    numInputVectors=[96] on the per-head StreamingSplit_hls nodes, when the
    producer Thresholding_rtl nodes actually emit seq=64 vectors. Part 2
    (in cancel_qkv_transposes.py) fixes the upstream drift; this step is
    defense in depth for any drift that still slips through.
    """
    from fix_streaming_split_vectors import FixStreamingSplitInputVectors
    return model.transform(FixStreamingSplitInputVectors())


def step_absorb_dequant_sdpa(model, cfg):
    """Fix 5: absorb pre-SDPA dequant Muls; rescale QK thresholds.

    Deletes the 9 ElementwiseMul_hls nodes (3 per SDPA head) that sit between
    QKV-projection Split outputs and SDPA Q/K/V inputs in the trained
    Brevitas graph. Sets QType/KType/VType=INT4 on each SDPA and rescales
    param0 (QK thresholds) by 1/(s_Q * s_K). See absorb_dequant_sdpa.py
    docstring for the full analysis and Gate 1/2 findings.

    Placement: after step_specialize_layers, before step_minimize_bit_width,
    so that minimize recomputes AccQKMatMul/AccAVMatMul from the new INT4
    Q/K/V instead of the FLOAT32 fallback widths.
    """
    from absorb_dequant_sdpa import AbsorbDequantIntoSDPA
    return model.transform(AbsorbDequantIntoSDPA())


def _read_model_dims_from_graph(model):
    """Derive seq_len and emb_dim from the SDPA node attributes in the graph.

    params.yaml in the working directory may belong to a different sweep
    (e.g. sweep_A with seq_len=16) than the model actually being built
    (trained model with seq_len=64). Reading from the graph is authoritative.
    """
    from qonnx.custom_op.registry import getCustomOp
    for node in model.graph.node:
        if node.op_type == "ScaledDotProductAttention_hls":
            inst = getCustomOp(node)
            seq_len = inst.get_nodeattr("QLen")
            emb_dim = inst.get_nodeattr("QKDim") * 3  # 3 heads × head_dim
            return seq_len, emb_dim
    # Fallback: read from params.yaml (legacy path for non-transformer models)
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

    seq_len, _emb_dim = _read_model_dims_from_graph(model)
    uram_threshold = seq_len  # matches FINN-T build.py default
    print(f"  FIFO depths: seq_len={seq_len} (from graph), depth={seq_len**2}")

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

        # StreamingSplit: same fork-join deadlock risk as ReplicateStream.
        # Each output feeds a different SDPA head; if one head stalls, the
        # shallow default depth-2 FIFO fills and blocks the upstream,
        # deadlocking the entire pipeline. Session 19 diagnosis.
        if node.op_type == "StreamingSplit_hls":
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
