"""FixStreamingSplitInputVectors: re-derive numInputVectors from producer.

Bug (session 19 Gate 1 / InsertFIFO step 18 failure): tensor shape annotations
on the QKV-projection path drifted to [96, 96] / [96, 32] during earlier
custom passes (likely the transpose-cancellation cluster in step_replicate_streams).
InferSplitLayer in step_convert_to_hw read the stale [96, 96] input annotation
and computed numInputVectors=[96] on StreamingSplit, which is wrong -- actual
data flow has seq=64, so numInputVectors should be [64]. The drift survived
every downstream stage because MVAU/SDPA/Thresholding read their own attrs
rather than the tensor annotation, but InsertFIFO's size-equality check fires
on it.

Fix: for each StreamingSplit_hls / StreamingConcat_hls, look at the direct
producer of each input. If the producer is an fpgadataflow node with a
numInputVectors attribute, adopt that as the canonical value and overwrite
this node's numInputVectors. Also correct the input/output tensor shape
annotations so downstream consumers see consistent shapes.

Defensive coverage: the actual drift in fix6 is only on StreamingSplit; the
three StreamingConcat_hls_0 inputs are already [64]. Including StreamingConcat
in the scan costs nothing and hardens against future drift in the same region.
"""
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


def _get_num_input_vectors(node):
    for a in node.attribute:
        if a.name == "numInputVectors" and a.type == 7:
            return list(a.ints)
    return None


def _set_num_input_vectors(node, value):
    for a in node.attribute:
        if a.name == "numInputVectors" and a.type == 7:
            del a.ints[:]
            a.ints.extend(int(x) for x in value)
            return True
    return False


class FixStreamingSplitInputVectors(Transformation):
    """Align StreamingSplit / StreamingConcat numInputVectors and connected
    tensor shapes with their actual upstream producers."""

    def apply(self, model: ModelWrapper):
        graph = model.graph
        modified = False
        self.changes = []

        for node in graph.node:
            if node.op_type not in ("StreamingSplit_hls", "StreamingConcat_hls"):
                continue
            current_nv = _get_num_input_vectors(node)

            producer_nv = None
            for inp in node.input:
                p = model.find_producer(inp)
                if p is None:
                    continue
                p_nv = _get_num_input_vectors(p)
                if p_nv is not None and len(p_nv) > 0:
                    producer_nv = p_nv
                    break

            if producer_nv is None:
                continue
            if current_nv == producer_nv:
                continue

            _set_num_input_vectors(node, producer_nv)

            seq = int(producer_nv[0]) if len(producer_nv) == 1 else None

            retagged = []
            if seq is not None:
                if node.op_type == "StreamingSplit_hls":
                    for inp in node.input:
                        shp = model.get_tensor_shape(inp)
                        if shp and len(shp) >= 2 and shp[0] != seq:
                            new_shp = [seq] + list(shp[1:])
                            model.set_tensor_shape(inp, new_shp)
                            retagged.append((inp, shp, new_shp))
                    for out in node.output:
                        shp = model.get_tensor_shape(out)
                        if shp and len(shp) >= 2 and shp[0] != seq:
                            new_shp = [seq] + list(shp[1:])
                            model.set_tensor_shape(out, new_shp)
                            retagged.append((out, shp, new_shp))
                elif node.op_type == "StreamingConcat_hls":
                    for inp in node.input:
                        shp = model.get_tensor_shape(inp)
                        if shp and len(shp) >= 2 and shp[0] != seq:
                            new_shp = [seq] + list(shp[1:])
                            model.set_tensor_shape(inp, new_shp)
                            retagged.append((inp, shp, new_shp))
                    for out in node.output:
                        shp = model.get_tensor_shape(out)
                        if shp and len(shp) >= 2 and shp[0] != seq:
                            new_shp = [seq] + list(shp[1:])
                            model.set_tensor_shape(out, new_shp)
                            retagged.append((out, shp, new_shp))

            self.changes.append({
                "node": node.name,
                "op_type": node.op_type,
                "numInputVectors": f"{current_nv} -> {producer_nv}",
                "retagged_tensors": retagged,
            })
            modified = True

        return model, modified
