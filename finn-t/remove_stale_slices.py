"""Fix 3a: Remove Slice nodes that were identity-by-intent but became
truncating after Squeeze failed to re-index their axes attribute.

Background: einops ``pack([x], "b * d")`` on a single tensor emits
``unpack`` as Slice + Reshape. Pre-Squeeze the Slice operated on the
sequence axis (axis=1 in rank-3 ``[batch, seq, emb]``), selecting the
full range ``[0:seq_len]`` — an identity. Squeeze removed the batch dim
without decrementing Slice's ``axes`` attribute, so post-Squeeze the Slice
now targets the embedding axis (axis=1 in rank-2 ``[seq, emb]``), slicing
``[0:seq_len]`` out of ``emb`` — a truncation if ``emb > seq_len``.

Detection: Slice with ``starts=0``, ``steps=1``, and
``input_shape[axis] > ends`` (the slice truncates because the axis got
longer after Squeeze shuffled which dim the axis points to). The original
intent was identity (``input_shape[axis] == ends`` on the PRE-Squeeze
axis); the truncation proves the axes weren't re-indexed.

Fix: delete the Slice and rewire its single consumer to the Slice's input.
"""

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes


class RemoveStaleSlices(Transformation):
    """Remove Slice nodes that are stale-axis identity slices from einops
    unpack. See module docstring for detection criteria."""

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False

        for node in list(graph.node):
            if node.op_type != "Slice":
                continue
            if len(node.input) < 3:
                continue

            in_shape = model.get_tensor_shape(node.input[0])
            if in_shape is None or len(in_shape) == 0:
                continue

            starts = model.get_initializer(node.input[1])
            ends = model.get_initializer(node.input[2])
            if starts is None or ends is None:
                continue
            starts = int(np.atleast_1d(starts).item())
            ends = int(np.atleast_1d(ends).item())

            axes = None
            if len(node.input) > 3:
                axes_init = model.get_initializer(node.input[3])
                if axes_init is not None:
                    axes = int(np.atleast_1d(axes_init).item())
            if axes is None:
                axes = 0

            steps = 1
            if len(node.input) > 4:
                steps_init = model.get_initializer(node.input[4])
                if steps_init is not None:
                    steps = int(np.atleast_1d(steps_init).item())

            if starts != 0 or steps != 1:
                continue

            if axes < 0:
                axes += len(in_shape)

            dim_at_axis = in_shape[axes]
            if dim_at_axis <= ends:
                continue

            inp = node.input[0]
            out = node.output[0]
            for consumer in graph.node:
                for i, t in enumerate(consumer.input):
                    if t == out:
                        consumer.input[i] = inp
            for go in graph.output:
                if go.name == out:
                    go.name = inp
            graph.node.remove(node)
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return model, False
