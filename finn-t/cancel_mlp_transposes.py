"""Fix 3b: Cancel the MLP-path Transpose pair (Transpose_4 ↔ Transpose_5 in
the Fix 2B numbering).

The chain is:
    Mul_11_out0 [64, 96] (NC)
      → Transpose perm=[1,0] → [96, 64] (CN)
        → ElementwiseAdd(+ bias [96, 1])
      → Transpose perm=[1,0] → [64, 96] (NC)
        → MultiThreshold → MatMul (MLP entry)

Round-trip NC→CN→NC with a single channelwise ElementwiseAdd between. The
``[96, 1]`` bias broadcasts at axis 0 in CN layout; transposing it to
``[1, 96]`` makes it broadcast at axis 1 in NC layout — same per-channel
addition. ElementwiseAdd's shape attributes also need to flip.
"""

import numpy as np
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import get_by_name


class CancelMLPTransposePair(Transformation):
    """Delete Transpose→ElementwiseAdd([C,1])→Transpose pairs (perm=[1,0])
    and reshape the channelwise bias initializer to [1,C].

    Guard: fires only when:
    - First node is Transpose perm=[1,0], single consumer is ElementwiseAdd
    - ElementwiseAdd's second input is an initializer with shape [C, 1]
    - ElementwiseAdd's single consumer is Transpose perm=[1,0]
    No-op on graphs without this specific pattern.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        consumers = {}
        for n in graph.node:
            for t in n.input:
                consumers.setdefault(t, []).append(n)

        graph_modified = False

        for node in list(graph.node):
            if node.op_type != "Transpose":
                continue
            perm = get_by_name(node.attribute, "perm")
            if perm is None or list(perm.ints) != [1, 0]:
                continue

            t_first = node
            t_first_consumers = consumers.get(t_first.output[0], [])
            if len(t_first_consumers) != 1:
                continue
            ew_add = t_first_consumers[0]
            if ew_add.op_type != "ElementwiseAdd":
                continue

            init_name = None
            for inp in ew_add.input:
                init = model.get_initializer(inp)
                if init is not None and init.ndim == 2:
                    if init.shape[1] == 1 and init.shape[0] > 1:
                        init_name = inp
                        break
            if init_name is None:
                continue

            ew_add_consumers = consumers.get(ew_add.output[0], [])
            if len(ew_add_consumers) != 1:
                continue
            t_second = ew_add_consumers[0]
            if t_second.op_type != "Transpose":
                continue
            perm2 = get_by_name(t_second.attribute, "perm")
            if perm2 is None or list(perm2.ints) != [1, 0]:
                continue

            # === Pattern matched ===
            t_first_input = t_first.input[0]
            t_second_output = t_second.output[0]

            # (a) Rewire ElementwiseAdd input: bypass first Transpose
            for i, inp in enumerate(ew_add.input):
                if inp == t_first.output[0]:
                    ew_add.input[i] = t_first_input
                    break

            # (b) Rewire consumer of second Transpose → ElementwiseAdd output
            for consumer in graph.node:
                for i, inp in enumerate(consumer.input):
                    if inp == t_second_output:
                        consumer.input[i] = ew_add.output[0]
            for go in graph.output:
                if go.name == t_second_output:
                    go.name = ew_add.output[0]

            # (c) Delete both Transposes
            graph.node.remove(t_first)
            graph.node.remove(t_second)

            # (d) Transpose the channelwise bias initializer [C, 1] → [1, C]
            old_init = model.get_initializer(init_name)
            new_init = old_init.T.copy()
            model.set_initializer(init_name, new_init)
            model.set_tensor_shape(init_name, list(new_init.shape))

            # (e) Update ElementwiseAdd shape attributes
            try:
                ew_inst = getCustomOp(ew_add)
                old_lhs = ew_inst.get_nodeattr("lhs_shape")
                old_rhs = ew_inst.get_nodeattr("rhs_shape")
                old_out = ew_inst.get_nodeattr("out_shape")
                ew_inst.set_nodeattr("lhs_shape", list(reversed(old_lhs)))
                ew_inst.set_nodeattr("rhs_shape", list(new_init.shape))
                ew_inst.set_nodeattr("out_shape", list(reversed(old_out)))
            except (AttributeError, KeyError):
                pass

            # (f) Clear stale shapes
            for tname in (t_first.output[0], ew_add.output[0],
                          t_second.output[0]):
                model.set_tensor_shape(tname, None)

            graph_modified = True
            break

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False
