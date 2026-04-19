"""Fix 2B: Cancel the Transpose_0 ↔ Transpose_1/2/3 quartet in the trained
transformer's post-streamline ONNX graph.

Background
----------
After Squeeze removes the batch/singleton dims, the attention pre-projection
chain acquires a round-trip layout conversion:

    Mul_1_out0 [64, 96] (seq × emb, NC)
      → Transpose_0 perm=[1,0] → [96, 64] (emb × seq, CN)
        → MultiThreshold_5 (per-channel thresh [96, 15])
          → ReplicateStream (3 copies)
            → Transpose_1/2/3 perm=[1,0] → [64, 96] (back to NC)
              → MatMul_1/2/3 (QKV projections)

The 4 Transposes form a net-identity NC→CN→NC with MultiThreshold_5 and
ReplicateStream between them. Those intermediate ops are layout-flexible:

- MultiThreshold: its data_layout attribute governs which axis is "channel".
  Changing from "NHC" (C at last for rank 3) to "NC" (C at axis 1 for rank 2)
  makes it threshold on axis 1 = emb = 96, matching the threshold tensor's
  shape [96, 15]. Verified via qonnx source in Gate 1 pre-work.

- ReplicateStream: purely replicates input N times. Its ``num_inputs``
  attribute records the input's leading dimensions for shape-inference
  stand-in. Must be updated from [96] → [64] when input changes from
  [96, 64] to [64, 96].

Patch actions:
1. Delete Transpose_0, Transpose_1, Transpose_2, Transpose_3.
2. Rewire MultiThreshold_5.input[0] from Transpose_0_out0 to Mul_1_out0.
3. Rewire MatMul_1/2/3.input[0] from Transpose_{1,2,3}_out0 to the three
   ReplicateStream output tensors.
4. Update MultiThreshold_5's data_layout attribute from "NHC" to "NC".
5. Update ReplicateStream's num_inputs attribute from [96] to [64].
6. Clear stale shape annotations on affected intermediate tensors so
   InferShapes re-derives them.

Numerical verification
-----------------------
qonnx's execute_onnx cannot run the mid-pipeline step_replicate_streams.onnx
(hybrid FINN/ONNX custom-op graph with multiple stale annotations). Gate 2a'
attempted: PyTorch model forward succeeded (75% argmax accuracy on 16 test
samples) but ONNX execution hit cascading shape mismatches in onnxruntime
(stale [96,96] on RS outputs → rank mismatch at StreamingConcat). The
mathematical argument for correctness of this transform is:
- The 4 Transposes form a round-trip identity on the data layout.
- Removing them all preserves semantics if and only if the ops between them
  are layout-flexible (verified: MT uses data_layout attribute, RS is
  agnostic, MatMul weights are square [96,96]).
Full build verification (Gate 6) is the de-facto correctness check.
"""

import numpy as np
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import get_by_name


class CancelQKVTransposes(Transformation):
    """Delete the Transpose_0 ↔ Transpose_1/2/3 quartet around the QKV
    pre-projection path and update the intermediate ops' attributes to match
    the new (un-transposed) data layout.

    Guards: fires only when it finds the EXACT pattern (4 Transpose nodes with
    perm=[1,0], connected via a MultiThreshold + ReplicateStream chain as
    described above). On any other graph topology (e.g., the DummyTransformer
    which has no PatchEmbedding and hence no pre-QKV layout swap), the pass
    is a no-op.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        producer = {o: n for n in graph.node for o in n.output}
        consumers = {}
        for n in graph.node:
            for t in n.input:
                consumers.setdefault(t, []).append(n)

        graph_modified = False

        for node in list(graph.node):
            # Step 1: find a Transpose perm=[1,0] whose output feeds a
            # MultiThreshold
            if node.op_type != "Transpose":
                continue
            perm_attr = get_by_name(node.attribute, "perm")
            if perm_attr is None or list(perm_attr.ints) != [1, 0]:
                continue

            t0 = node  # candidate Transpose_0

            # Consumer must be a single MultiThreshold
            t0_consumers = consumers.get(t0.output[0], [])
            if len(t0_consumers) != 1 or t0_consumers[0].op_type != "MultiThreshold":
                continue
            mt = t0_consumers[0]

            # MT's sole consumer must be a ReplicateStream
            mt_consumers = consumers.get(mt.output[0], [])
            if len(mt_consumers) != 1 or mt_consumers[0].op_type != "ReplicateStream":
                continue
            rs = mt_consumers[0]

            # RS must have 3 outputs, each consumed by exactly 1 Transpose perm=[1,0]
            if len(rs.output) != 3:
                continue
            t123 = []
            for rs_out in rs.output:
                rs_out_consumers = consumers.get(rs_out, [])
                if len(rs_out_consumers) != 1:
                    break
                cand = rs_out_consumers[0]
                if cand.op_type != "Transpose":
                    break
                cand_perm = get_by_name(cand.attribute, "perm")
                if cand_perm is None or list(cand_perm.ints) != [1, 0]:
                    break
                t123.append(cand)
            if len(t123) != 3:
                continue

            # Each of those 3 Transposes must feed exactly 1 MatMul
            matmuls = []
            for t in t123:
                t_consumers = consumers.get(t.output[0], [])
                if len(t_consumers) != 1 or t_consumers[0].op_type != "MatMul":
                    break
                matmuls.append(t_consumers[0])
            if len(matmuls) != 3:
                continue

            # === Pattern matched. Apply the patch. ===
            t0_input = t0.input[0]  # the tensor BEFORE Transpose_0

            # (a) Rewire MT input from Transpose_0 output → Transpose_0 input
            mt.input[0] = t0_input

            # (b) Rewire each MatMul input from Transpose_1/2/3 output → RS output
            for t, rs_out in zip(t123, rs.output):
                for mm in matmuls:
                    for i, inp in enumerate(mm.input):
                        if inp == t.output[0]:
                            mm.input[i] = rs_out

            # (c) Delete the 4 Transpose nodes
            for t in [t0] + t123:
                graph.node.remove(t)

            # (d) Update MT data_layout (NHC → NC for rank-2 channels-last)
            mt_inst = getCustomOp(mt)
            old_layout = mt_inst.get_nodeattr("data_layout")
            mt_inst.set_nodeattr("data_layout", "NC")

            # (e) Update RS num_inputs: old was [emb_dim] (channels-first
            #     leading axis); new is [seq_len] (channels-last leading axis)
            rs_inst = getCustomOp(rs)
            old_num_inputs = rs_inst.get_nodeattr("num_inputs")
            # Derive new num_inputs from the input tensor (now channels-last)
            t0_input_shape = model.get_tensor_shape(t0_input)
            if t0_input_shape is not None and len(t0_input_shape) >= 2:
                new_num_inputs = list(t0_input_shape[:-1])
            else:
                new_num_inputs = old_num_inputs
            rs_inst.set_nodeattr("num_inputs", new_num_inputs)

            # (f) Clear stale annotations, then EXPLICITLY propagate correct
            # shapes. Stray InferShapes calls crash later in the pipeline
            # (FINN custom op Squeeze is not in the ONNX opset), so we cannot
            # rely on it. Before session 19 this block only cleared and ran
            # InferShapes; when the latter silently failed, MT_6/7/8 outputs
            # retained their [96, 96] drift, which cascaded into StreamingSplit
            # numInputVectors and blew up InsertFIFO at step 18.
            stale_tensors = {t0.output[0], mt.output[0]}
            for rs_out in rs.output:
                stale_tensors.add(rs_out)
            for t in t123:
                stale_tensors.add(t.output[0])
            for mm in matmuls:
                stale_tensors.add(mm.output[0])
            for tname in stale_tensors:
                model.set_tensor_shape(tname, None)

            # Now re-derive using known upstream shape. t0_input is the
            # channels-last [seq, emb] tensor before Transpose_0; RS, MT and
            # MatMul inputs all share that rank-2 shape after cancellation.
            if t0_input_shape is not None and len(t0_input_shape) == 2:
                for tname in (mt.output[0], *rs.output):
                    model.set_tensor_shape(tname, list(t0_input_shape))
                for mm in matmuls:
                    w_init = model.get_initializer(mm.input[1])
                    if w_init is not None and w_init.ndim == 2:
                        mm_out_shape = [t0_input_shape[0], w_init.shape[-1]]
                        model.set_tensor_shape(mm.output[0], mm_out_shape)
                        # Walk forward through elementwise ops (MT / Add / Mul)
                        # that preserve rank-2 shape, so downstream consumers
                        # see the correct [seq, emb] annotation instead of the
                        # drifted [emb, emb].
                        cur_out = mm.output[0]
                        for _ in range(8):  # safety bound
                            consumers_cur = consumers.get(cur_out, [])
                            if len(consumers_cur) != 1:
                                break
                            ec = consumers_cur[0]
                            if ec.op_type not in (
                                "MultiThreshold", "Thresholding_rtl",
                                "Thresholding_hls", "Add", "Mul",
                            ):
                                break
                            ec_out = ec.output[0]
                            model.set_tensor_shape(ec_out, mm_out_shape)
                            cur_out = ec_out

            graph_modified = True
            break  # only one quartet per graph expected

        if graph_modified:
            try:
                model = model.transform(InferShapes())
            except Exception:
                pass  # expected to fail on FINN custom ops; (f) covered it
            model = model.transform(InferDataTypes())
        return model, False
