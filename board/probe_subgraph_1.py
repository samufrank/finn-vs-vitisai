"""xir API probe — walk subgraph [1] (simplest CPU subgraph: 3 ops, 1 add + 2 const).

Prints every attribute of every op, every input op pointer, every output
tensor's metadata + attrs, and tries to read const data via attr keys
'data' and 'value'. Goal: figure out the exact xir Python API surface
that the full orchestrator will need to walk all 12 CPU subgraphs.

Usage on board:
    python3 probe_subgraph_1.py
    python3 probe_subgraph_1.py /path/to/other.xmodel
"""
import sys
import xir
import numpy as np

XMODEL = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/petalinux/models/dpu/transformer_radioml/transformer_radioml.xmodel'

graph = xir.Graph.deserialize(XMODEL)
subs = graph.get_root_subgraph().toposort_child_subgraph()
print(f"Total subgraphs: {len(subs)}")

sg = subs[1]
device = sg.get_attr('device') if sg.has_attr('device') else 'UNKNOWN'
print(f"\n== Subgraph[1] ==")
print(f"  name:   {sg.get_name()}")
print(f"  device: {device}")
print(f"  in_tensors  ({len(sg.get_input_tensors())}): "
      f"{[t.name for t in sg.get_input_tensors()]}")
print(f"  out_tensors ({len(sg.get_output_tensors())}): "
      f"{[t.name for t in sg.get_output_tensors()]}")

# Walk ops — prefer topo order, fall back to get_ops
try:
    ops = sg.toposort_child_op()
    print(f"  toposort_child_op OK")
except Exception as e:
    print(f"  toposort_child_op FAILED: {e}; using get_ops")
    ops = list(sg.get_ops())
print(f"  {len(ops)} ops total\n")

def _abbrev(s, n=120):
    s = repr(s)
    return s if len(s) <= n else s[:n-3] + '...'

for i, op in enumerate(ops):
    print(f"--- Op[{i}] type={op.get_type()} name={op.get_name()} ---")

    # All attrs
    try:
        attrs = dict(op.get_attrs())
    except Exception as e:
        attrs = {}
        print(f"  get_attrs FAILED: {e}")
    for k in sorted(attrs):
        v = attrs[k]
        print(f"  attr[{k}] : {type(v).__name__} = {_abbrev(v)}")

    # Input ops (parent ops, keyed by argument name)
    try:
        in_ops = op.get_input_ops()
    except Exception as e:
        in_ops = {}
        print(f"  get_input_ops FAILED: {e}")
    for arg, parents in in_ops.items():
        for p in parents:
            print(f"  in[{arg}] <- type={p.get_type()} name='{p.get_name()}'")

    # Output tensor + its attrs (fix-point params usually live here)
    try:
        t = op.get_output_tensor()
        print(f"  output tensor: name='{t.name}' dims={t.dims} dtype={t.dtype}")
        try:
            t_attrs = dict(t.get_attrs()) if hasattr(t, 'get_attrs') else {}
        except Exception as e:
            t_attrs = {}
            print(f"    tensor.get_attrs FAILED: {e}")
        for k in sorted(t_attrs):
            v = t_attrs[k]
            print(f"    tensor.attr[{k}] : {type(v).__name__} = {_abbrev(v)}")
    except Exception as e:
        print(f"  get_output_tensor FAILED: {e}")

    # For constants, try to read the actual data array
    if op.get_type() in ('const', 'const-fix', 'data', 'data-fix'):
        for key in ('data', 'value'):
            try:
                if op.has_attr(key):
                    raw = op.get_attr(key)
                    arr = np.array(raw)
                    print(f"  CONST data via attr[{key}]: shape={arr.shape} "
                          f"dtype={arr.dtype} first3={arr.flatten()[:3].tolist()}")
                    break
            except Exception as e:
                print(f"  CONST attr[{key}] read FAILED: {e}")
    print()

# Cross-check: do subgraph[1]'s input tensor names appear as
# subgraph[0]'s output tensor names? Confirms tensor-passing convention.
print("\n== Cross-check: subgraph[0] output tensors vs subgraph[1] input tensors ==")
sg0 = subs[0]
out0 = [t.name for t in sg0.get_output_tensors()]
in1  = [t.name for t in sg.get_input_tensors()]
print(f"  sg[0] outputs: {out0}")
print(f"  sg[1] inputs:  {in1}")
print(f"  intersection:  {sorted(set(out0) & set(in1))}")
