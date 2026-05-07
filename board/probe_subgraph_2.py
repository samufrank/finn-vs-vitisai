"""xir probe of CPU subgraph [2] — 10 ops, the first real CPU compute.

Layout:
  const-fix×1, float2fix×2, eltwise-fix×1, fix2float×2,
  aten::round×1, transpose×2, nndct_clamp×1

Goal: extract the exact attribute keys we need to implement these in numpy.
Specifically:
  - float2fix / fix2float / const-fix: which attr stores the Q-format
    fix_point (and is it on the op, or on the output tensor)?
  - eltwise-fix: which attr discriminates add vs mul vs sub?
  - transpose: where is the permutation order?
  - nndct_clamp: where are min/max?
  - const-fix: how to decode the data bytes (dtype + shape)?

Run on board:
    python3 probe_subgraph_2.py
"""
import sys
import xir
import numpy as np
from collections import Counter

XMODEL = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/petalinux/models/dpu/transformer_radioml/transformer_radioml.xmodel'

subs = xir.Graph.deserialize(XMODEL).get_root_subgraph().toposort_child_subgraph()
sg = subs[2]
print(f"sg[2] device={sg.get_attr('device')}  name={sg.get_name()}")
print(f"  in_tensors:  {[t.name for t in sg.get_input_tensors()]}")
print(f"  out_tensors: {[t.name for t in sg.get_output_tensors()]}")

ops = list(sg.get_ops())
print(f"  {len(ops)} ops, types: {dict(Counter(o.get_type() for o in ops))}\n")


def short(v, n=140):
    if isinstance(v, (bytes, bytearray)):
        return f"bytes[{len(v)}]"
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + '...'


# Walk every op, dump op attrs + tensor attrs + parents
for i, op in enumerate(ops):
    t_out = op.get_output_tensor()
    print(f"[{i}] {op.get_type():16s} -> '{t_out.name}' "
          f"dims={t_out.dims} dtype={t_out.dtype}")
    try:
        a_op = dict(op.get_attrs())
    except Exception as e:
        a_op = {}
        print(f"   op.get_attrs FAIL: {e}")
    for k in sorted(a_op):
        print(f"   op.attr[{k}] : {type(a_op[k]).__name__} = {short(a_op[k])}")
    try:
        a_t = dict(t_out.get_attrs()) if hasattr(t_out, 'get_attrs') else {}
    except Exception as e:
        a_t = {}
        print(f"   t.get_attrs FAIL: {e}")
    for k in sorted(a_t):
        print(f"   t.attr[{k}] : {type(a_t[k]).__name__} = {short(a_t[k])}")
    try:
        for arg, parents in op.get_input_ops().items():
            for p in parents:
                print(f"   in[{arg}] <- {p.get_type():14s} '{p.get_name()}'")
    except Exception as e:
        print(f"   inputs FAIL: {e}")

# Decode any const-fix bytes — try int8/int32 to figure out the storage
print("\n=== const-fix data decode test ===")
for op in ops:
    if op.get_type() == 'const-fix' and op.has_attr('data'):
        raw = op.get_attr('data')
        if not isinstance(raw, (bytes, bytearray)):
            print(f"  {op.get_name()}: attr[data] is not bytes; type={type(raw).__name__}")
            continue
        out_t = op.get_output_tensor()
        print(f"  {op.get_name()}: dims={out_t.dims} dtype={out_t.dtype} "
              f"raw_len={len(raw)}")
        n_elem = int(np.prod(out_t.dims)) if out_t.dims else 1
        print(f"    expected elements: {n_elem}, bytes/elem: "
              f"{len(raw)/n_elem if n_elem else '?'}")
        for npdt in (np.int8, np.uint8, np.int32, np.float32):
            if len(raw) % np.dtype(npdt).itemsize == 0:
                try:
                    arr = np.frombuffer(raw, dtype=npdt)
                    if arr.size == n_elem or n_elem == 0:
                        print(f"    as {npdt.__name__}: shape={arr.shape} "
                              f"first5={arr[:5].tolist()}")
                except Exception:
                    pass
        break  # one example is enough

# Boundary tensor flow: confirm tensor names match upstream/downstream
print("\n=== Boundary tensor name flow ===")
print(f"  sg[0] outputs: {[t.name for t in subs[0].get_output_tensors()]}")
print(f"  sg[2] inputs:  {[t.name for t in sg.get_input_tensors()]}")
print(f"  sg[2] outputs: {[t.name for t in sg.get_output_tensors()]}")
print(f"  sg[3] inputs:  {[t.name for t in subs[3].get_input_tensors()]}")
# Look for name-prefix matches across boundaries
def overlap(a, b):
    pairs = []
    for x in a:
        for y in b:
            if x == y:
                pairs.append(('exact', x, y))
            elif x.startswith(y) or y.startswith(x):
                pairs.append(('prefix', x, y))
    return pairs
print(f"  sg[0]→sg[2] matches: {overlap([t.name for t in subs[0].get_output_tensors()], [t.name for t in sg.get_input_tensors()])}")
print(f"  sg[2]→sg[3] matches: {overlap([t.name for t in sg.get_output_tensors()], [t.name for t in subs[3].get_input_tensors()])}")
