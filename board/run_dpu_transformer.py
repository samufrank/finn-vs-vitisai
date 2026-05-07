"""run_dpu_transformer.py — manual orchestration of the 21-subgraph DPU
transformer xmodel. DPU subgraphs run via vart.Runner; CPU subgraphs run
by walking xir ops in numpy.

Tensor passing is by xir tensor name into a global buffer dict. All buffer
values are float32 — int-representations are stored as their dequantized
float values (int_val * 2^-fix_point), so subsequent ops just read floats.
Round-trips through float2fix/fix2float collapse to the rounding+clamping
the model expected.

Skipped: sg[0] (USER, data-fix is applied manually pre-loop), sg[1] (BN
counter, no data-path connection per probe).

Usage:
    python3 run_dpu_transformer.py
    python3 run_dpu_transformer.py --n 10000 --xmodel ... --data ...
"""
import os, sys, time, json, argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import xir, vart


# ============================================================
# CPU op implementations (numpy)
# ============================================================
def _quantize(x, fp, bw, sgn):
    """DPU_ROUND + clip; returns float on integer grid (= dequantized int)."""
    lo = -(1 << (bw - 1)) if sgn else 0
    hi = (1 << (bw - 1)) - 1 if sgn else (1 << bw) - 1
    s = 2.0 ** fp
    # round-half-away-from-zero (DPU_ROUND); np.round is banker's
    rounded = np.copysign(np.floor(np.abs(x * s) + 0.5), x)
    return np.clip(rounded, lo, hi) / s


def _quant_out(x, t_a):
    """Apply output quantization if t_a carries fix_point."""
    if 'fix_point' in t_a:
        return _quantize(x, t_a['fix_point'], t_a.get('bit_width', 8),
                         t_a.get('if_signed', True))
    return x


def op_fix2float(inp, op_a, t_a):
    return inp[0].astype(np.float32)


def op_float2fix(inp, op_a, t_a):
    return _quantize(inp[0], op_a['fix_point'], op_a['bit_width'],
                     op_a['if_signed']).astype(np.float32)


def op_eltwise_fix(inp, op_a, t_a):
    global _DIV_DEBUGGED
    t = op_a.get('type', 'ADD')
    if t == 'DIV' and not _DIV_DEBUGGED:
        # Verify operand order: probe showed inp[0]=activation, inp[1]=const_scale.
        # If swapped, division would be 1/x and accuracy would be ruined.
        a, b = inp[0], inp[1]
        a_first = a.flatten()[:5].tolist() if a.size else []
        b_first = b.flatten()[:5].tolist() if b.size else []
        print(f"\n  [eltwise-fix DIV first call] op_attrs={dict(op_a)}")
        print(f"    inp[0] shape={a.shape} (numerator?)   first5={a_first}")
        print(f"    inp[1] shape={b.shape} (denominator?) first5={b_first}")
        print(f"    inp[0] mean|.|={float(np.abs(a).mean()):.4f}, "
              f"inp[1] mean|.|={float(np.abs(b).mean()):.4f}")
        # If inp[0] is the small constant (mean ~0.013) and inp[1] is the wide
        # activation (mean ~5+), the operands are swapped.
        _DIV_DEBUGGED = True
    if t == 'ADD':   r = inp[0] + inp[1]
    elif t == 'SUB': r = inp[0] - inp[1]
    elif t == 'MUL': r = inp[0] * inp[1]
    elif t == 'DIV': r = inp[0] / inp[1]
    else: raise ValueError(f"eltwise-fix type={t}")
    return _quant_out(r, t_a).astype(np.float32)


def op_const_fix(inp, op_a, t_a):
    arr = np.frombuffer(op_a['data'], dtype=np.int8).reshape(list(op_a['shape']))
    return arr.astype(np.float32) / (2.0 ** t_a['fix_point'])


def op_const(inp, op_a, t_a):
    raw = op_a.get('data')
    if isinstance(raw, (bytes, bytearray)):
        dt = op_a.get('data_type', 'FLOAT32').upper()
        npdt = {'FLOAT32': np.float32, 'INT32': np.int32,
                'INT8': np.int8, 'XINT8': np.int8}.get(dt, np.float32)
        arr = np.frombuffer(raw, dtype=npdt)
        if op_a.get('shape'): arr = arr.reshape(op_a['shape'])
        arr = arr.astype(np.float32)
    else:
        arr = np.array(raw, dtype=np.float32)
    # Ensure at least 1-d: 0-d scalars break run_dpu_subgraph's a.shape[0]
    # access, and broadcast just as well as 1-d in eltwise ops.
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def op_reshape_fix(inp, op_a, t_a):
    return inp[0].reshape(list(op_a.get('shape', list(inp[0].shape))))


def op_strided_slice_fix(inp, op_a, t_a):
    """XIR strided_slice-fix. Tries common attribute key spellings."""
    global _STRIDED_DEBUG_COUNT
    a = inp[0]
    b = _attr(op_a, 'begin', 'start', 'starts',
              default=[0] * a.ndim)
    e = _attr(op_a, 'end',   'stop',  'stops',
              default=list(a.shape))
    s = _attr(op_a, 'strides', 'stride', 'step', 'steps',
              default=[1] * a.ndim)
    out = a[tuple(slice(int(x), int(y), int(z)) for x, y, z in zip(b, e, s))]
    if _STRIDED_DEBUG_COUNT < 9:
        print(f"  [strided_slice-fix #{_STRIDED_DEBUG_COUNT}] "
              f"in={tuple(a.shape)} begin={list(b)} end={list(e)} "
              f"strides={list(s)} → out={tuple(out.shape)}")
        if _STRIDED_DEBUG_COUNT == 0:
            print(f"    full op_attrs: {dict(op_a)}")
        _STRIDED_DEBUG_COUNT += 1
    return out


def op_matmul(inp, op_a, t_a):
    """XIR matmul. Honors `transpose_a` / `transpose_b` attributes — these
    are how XIR encodes the K-transpose for Q @ K^T in the attention block
    (no separate transpose op gets emitted, which is why sg[8] has 0
    transpose ops in its op-type breakdown)."""
    global _MATMUL_DEBUGGED
    a, b = inp[0], inp[1]
    ta = bool(_attr(op_a, 'transpose_a', 'trans_a', 'transposeA', default=False))
    tb = bool(_attr(op_a, 'transpose_b', 'trans_b', 'transposeB', default=False))
    if not _MATMUL_DEBUGGED:
        print(f"  [matmul first call] inp[0]={tuple(a.shape)} "
              f"inp[1]={tuple(b.shape)} transpose_a={ta} transpose_b={tb}")
        print(f"    op_attrs: {dict(op_a)}")
        _MATMUL_DEBUGGED = True
    if ta:
        a = np.swapaxes(a, -2, -1)
    if tb:
        b = np.swapaxes(b, -2, -1)
    return np.matmul(a, b).astype(np.float32)


def op_concat_fix(inp, op_a, t_a):
    return _quant_out(np.concatenate(inp, axis=op_a.get('axis', 0)), t_a).astype(np.float32)


def op_softmax(inp, op_a, t_a):
    ax = op_a.get('axis', -1)
    x = inp[0] - np.max(inp[0], axis=ax, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=ax, keepdims=True)).astype(np.float32)


def op_aten_mean(inp, op_a, t_a):
    """np.mean over the axis the model intended.

    Bug history: this was using `op_a.get('axes') or op_a.get('dim')`,
    which returns None for `dim=0` (0 is falsy) — and the original GAP for
    our model is `x.mean(dim=1)`. If vai_q_pytorch traces under a key that
    happens to hold `0`, the OR-chain skipped it and we fell into the
    full-reduction scalar fallback. Now uses explicit `key in op_a` checks
    and falls back to inferring the reduced axis from the output tensor's
    shape (set by xir's auto-definition for unknown ops)."""
    global _MEAN_DEBUGGED
    in_arr = inp[0]
    in_shape = tuple(in_arr.shape)

    # 1. Direct axis lookup — accept 0 / empty list as valid (don't use `or`)
    ax = None
    for key in ('axes', 'dim', 'dims', 'axis'):
        if key in op_a:
            ax = op_a[key]
            break

    # 2. keepdim lookup (same care: don't use `or`)
    kd = False
    for key in ('keepdim', 'keepdims', 'keep_dims'):
        if key in op_a:
            kd = bool(op_a[key])
            break

    # 3. Normalize axis to a tuple of ints
    if ax is not None:
        if isinstance(ax, int):
            ax_t = (ax,)
        elif hasattr(ax, '__iter__'):
            ax_t = tuple(int(x) for x in ax)
        else:
            ax_t = (int(ax),)
    else:
        ax_t = None

    # 4. Fallback: infer reduced axis from output shape if attribute is absent.
    #    XIR's auto-definition for unknown ops sets t_a['shape'] to the
    #    expected output shape — use that to back out the axis.
    if ax_t is None:
        out_shape = t_a.get('shape') or op_a.get('shape')
        if out_shape and len(out_shape) < len(in_shape):
            in_dims = list(in_shape)
            out_dims = list(out_shape)
            inferred = []
            j = 0
            for i, d_in in enumerate(in_dims):
                if j < len(out_dims) and d_in == out_dims[j]:
                    j += 1
                else:
                    inferred.append(i)
            if len(inferred) == len(in_dims) - len(out_dims):
                ax_t = tuple(inferred)

    if ax_t is None:
        raise ValueError(
            f"aten::mean: cannot determine reduction axis. "
            f"in_shape={in_shape} op_attrs={dict(op_a)} t_attrs={dict(t_a)}")

    out = np.mean(in_arr, axis=ax_t, keepdims=kd).astype(np.float32)
    if out.ndim == 0:
        out = out.reshape(1)

    if not _MEAN_DEBUGGED:
        print(f"\n  [aten::mean first call] in_shape={in_shape}")
        print(f"    op_attrs: {dict(op_a)}")
        print(f"    t_attrs:  {dict(t_a)}")
        print(f"    parsed axis={ax_t} keepdim={kd} → out_shape={tuple(out.shape)}")
        _MEAN_DEBUGGED = True

    return out


def op_fix(inp, op_a, t_a):
    fp = t_a.get('fix_point', op_a.get('fix_point', 0))
    bw = t_a.get('bit_width', op_a.get('bit_width', 8))
    sgn= t_a.get('if_signed', op_a.get('if_signed', True))
    return _quantize(inp[0], fp, bw, sgn).astype(np.float32)


def op_nndct_clamp(inp, op_a, t_a):
    """Real clamping. nndct_clamp sits between aten::round and float2fix in
    the fake-quant chain `(x/scale).round().clamp(lo, hi) * scale`. The
    bound depends on the bit width of the surrounding quantizer:
      INT8 signed → [-128, 127]
      INT4 signed → [-8, 7]
    But xir's nndct_clamp op carries no min/max attrs (probe confirmed).
    We default to [-128, 127] — exact for INT8, redundant for INT4 because
    the downstream float2fix re-clamps to [-8, 7]. Pass-through (the
    previous behavior) let value-range overflow corrupt the classifier
    logits, manifesting as -128 saturation across most output classes.
    """
    bw = t_a.get('bit_width', op_a.get('bit_width', 8))
    sgn = t_a.get('if_signed', op_a.get('if_signed', True))
    if sgn:
        lo, hi = -(1 << (bw - 1)), (1 << (bw - 1)) - 1
    else:
        lo, hi = 0, (1 << bw) - 1
    return np.clip(inp[0], lo, hi).astype(np.float32)


def _attr(op_a, *keys, default=None):
    """Look up the first key present in op_a; falls back to default."""
    for k in keys:
        if k in op_a:
            return op_a[k]
    return default


# First-call debug flags for shape-sensitive ops. Each fires once so the
# inference loop doesn't get spammed with 100+ identical dumps.
_CONV2D_DEBUGGED        = False
_MATMUL_DEBUGGED        = False
_MEAN_DEBUGGED          = False
_DIV_DEBUGGED           = False
_STRIDED_DEBUG_COUNT    = 0    # prints first 9 strided_slice-fix calls in sg[8]
_DEBUG_SG_IDX           = 8    # subgraph to focus diagnostics on (attention)


def op_conv2d_fix(inp, op_a, t_a):
    """NHWC 2D conv with quantized output.

    XIR convention (verified against Vitis AI source):
      - kernel, stride, dilation stored as [W, H], NOT [H, W]
      - pad stored as [pad_left, pad_right, pad_top, pad_bottom]
    This is the OPPOSITE of PyTorch's [H, W] ordering. Misreading [H, W]
    on patch_conv (kernel=[16,1] in xir = [kW=16,kH=1]) made H_out come
    out as 0 because the math used kH=16 against H=1.
    """
    global _CONV2D_DEBUGGED

    kernel   = _attr(op_a, 'kernel',   'kernel_size', 'kernels', default=[1, 1])
    stride   = _attr(op_a, 'stride',   'strides',                default=[1, 1])
    pad      = _attr(op_a, 'pad',      'padding',                default=[0, 0, 0, 0])
    dilation = _attr(op_a, 'dilation',                           default=[1, 1])

    # XIR W,H ordering — unpack accordingly.
    kW, kH = int(kernel[0]), int(kernel[1])
    sW, sH = int(stride[0]), int(stride[1])
    dW, dH = int(dilation[0]), int(dilation[1])

    # Normalize pad to 4-elem [left, right, top, bottom] (XIR convention).
    if len(pad) == 2:
        # 2-elem form: [pad_w, pad_h] → [pad_w, pad_w, pad_h, pad_h]
        pad = [pad[0], pad[0], pad[1], pad[1]]
    elif len(pad) != 4:
        pad = [0, 0, 0, 0]
    pad_l, pad_r, pad_t, pad_b = (int(pad[0]), int(pad[1]),
                                   int(pad[2]), int(pad[3]))

    # ---- Identify inputs ----
    bias = next((a for a in inp if a.ndim == 1), None)
    four_d = [a for a in inp if a.ndim == 4]
    if len(four_d) != 2:
        raise ValueError(
            f"conv2d-fix: expected exactly 2 4D inputs, got {len(four_d)}; "
            f"all input shapes: {[a.shape for a in inp]}; op_attrs: {dict(op_a)}")

    x = w = None
    if bias is not None:
        cout = int(bias.shape[0])
        if int(four_d[0].shape[0]) == cout:
            w, x = four_d[0], four_d[1]
        elif int(four_d[1].shape[0]) == cout:
            w, x = four_d[1], four_d[0]
    if w is None:
        # Fallback: kernel-shape matching on OHWC weight (kH at dim 1, kW at dim 2)
        for cand_w, cand_x in (four_d, four_d[::-1]):
            if int(cand_w.shape[1]) == kH and int(cand_w.shape[2]) == kW:
                w, x = cand_w, cand_x
                break
    if x is None or w is None:
        raise ValueError(
            f"conv2d-fix: cannot identify input/weight tensors.\n"
            f"  4D input shapes: {[a.shape for a in four_d]}\n"
            f"  bias shape: {bias.shape if bias is not None else None}\n"
            f"  parsed kernel(W,H)=({kW},{kH}) stride=({sW},{sH}) "
            f"pad(L,R,T,B)=({pad_l},{pad_r},{pad_t},{pad_b}) "
            f"dilation=({dW},{dH})\n"
            f"  full op_attrs: {dict(op_a)}")

    # ---- One-shot diagnostic dump ----
    if not _CONV2D_DEBUGGED:
        _CONV2D_DEBUGGED = True
        print(f"\n  [conv2d-fix first-call diagnostic]")
        print(f"    raw op_attrs: {dict(op_a)}")
        print(f"    raw kernel={kernel} stride={stride} pad={pad} dilation={dilation}")
        print(f"    parsed (XIR W,H ordering): "
              f"kW={kW} kH={kH} sW={sW} sH={sH} dW={dW} dH={dH} "
              f"pad_l={pad_l} pad_r={pad_r} pad_t={pad_t} pad_b={pad_b}")
        print(f"    x.shape (NHWC): {x.shape}")
        print(f"    w.shape (OHWC): {w.shape}")
        print(f"    bias.shape: {bias.shape if bias is not None else None}")
        # Predict output shape to verify the parse before running the conv
        pH = x.shape[1] + pad_t + pad_b
        pW = x.shape[2] + pad_l + pad_r
        Hout_p = (pH - (kH - 1) * dH - 1) // sH + 1
        Wout_p = (pW - (kW - 1) * dW - 1) // sW + 1
        print(f"    predicted output: (N={x.shape[0]}, H={Hout_p}, "
              f"W={Wout_p}, C={w.shape[0]})")
        if Hout_p <= 0 or Wout_p <= 0:
            print(f"    WARNING: predicted H or W <= 0; "
                  f"kernel/stride parse is likely wrong")

    # ---- Convolution (NHWC) ----
    if pad_l or pad_r or pad_t or pad_b:
        x = np.pad(x, [(0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)])
    N, H, W, Cin = x.shape
    Cout = w.shape[0]
    Hout = (H - (kH - 1) * dH - 1) // sH + 1
    Wout = (W - (kW - 1) * dW - 1) // sW + 1
    y = np.zeros((N, Hout, Wout, Cout), dtype=np.float32)
    for i in range(Hout):
        for j in range(Wout):
            patch = x[:, i*sH:i*sH+kH, j*sW:j*sW+kW, :]
            y[:, i, j, :] = np.einsum('nhwc,ohwc->no', patch, w)
    if bias is not None:
        y += bias.reshape(1, 1, 1, -1)
    # Apply fused nonlinearity. xir conv2d-fix's `nonlinear` attribute can
    # be NONE / RELU / HSIGMOID / HSWISH. Our patch_conv (sg[4]) has RELU
    # because the compiler folded patch_conv → BN → ReLU into one op; not
    # applying it here was the silent accuracy killer.
    nonlinear = op_a.get('nonlinear', 'NONE')
    if nonlinear == 'RELU':
        y = np.maximum(y, 0)
    elif nonlinear not in ('NONE', '', None):
        raise NotImplementedError(
            f"conv2d-fix: unsupported nonlinear='{nonlinear}'")
    return _quant_out(y, t_a).astype(np.float32)


CPU_OPS = {
    'fix2float':         op_fix2float,
    'float2fix':         op_float2fix,
    'eltwise-fix':       op_eltwise_fix,
    'const-fix':         op_const_fix,
    'const':             op_const,
    'reshape-fix':       op_reshape_fix,
    'strided_slice-fix': op_strided_slice_fix,
    'concat-fix':        op_concat_fix,
    'softmax':           op_softmax,
    'aten::mean':        op_aten_mean,
    'fix':               op_fix,
    'data-fix':          op_fix,                                    # same logic as 'fix'
    'aten::round':       lambda inp, op_a, t_a: np.round(inp[0]).astype(np.float32),
    'nndct_clamp':       op_nndct_clamp,
    'transpose':         lambda inp, op_a, t_a: np.transpose(inp[0], axes=op_a['order']),
    'matmul':            op_matmul,
    'add':               lambda inp, op_a, t_a: (inp[0] + inp[1]).astype(np.float32),
    'mul':               lambda inp, op_a, t_a: (inp[0] * inp[1]).astype(np.float32),
    'div':               lambda inp, op_a, t_a: (inp[0] / inp[1]).astype(np.float32),
    'conv2d-fix':        op_conv2d_fix,
}


# ============================================================
# Subgraph executors
# ============================================================
def run_cpu_subgraph(sg, buffers, sg_idx=None):
    """Walk ops in dataflow order via repeated-pass execution: each pass
    runs every op whose parents' outputs are all in `buffers`, deferring
    the rest. Stops when no ops remain or no progress is made (latter is
    a hard error). xir's get_ops() does not guarantee topological order,
    so this is required."""
    pending = list(sg.get_ops())
    while pending:
        progress = False
        deferred = []
        for op in pending:
            # Probe parent output names — if any missing, defer this op
            input_arrs = []
            ready = True
            for arg, parents in op.get_input_ops().items():
                for p in parents:
                    nm = p.get_output_tensor().name
                    if nm not in buffers:
                        ready = False
                        break
                    input_arrs.append(buffers[nm])
                if not ready:
                    break
            if not ready:
                deferred.append(op)
                continue
            # Execute
            op_a = dict(op.get_attrs())
            out_t = op.get_output_tensor()
            t_a = dict(out_t.get_attrs()) if hasattr(out_t, 'get_attrs') else {}
            fn = CPU_OPS.get(op.get_type())
            if fn is None:
                raise NotImplementedError(
                    f"sg[{sg_idx}] '{sg.get_name()}': CPU op '{op.get_type()}' "
                    f"(name='{op.get_name()}') not implemented")
            try:
                buffers[out_t.name] = fn(input_arrs, op_a, t_a)
            except Exception as e:
                raise RuntimeError(
                    f"sg[{sg_idx}] '{sg.get_name()}': op '{op.get_type()}' "
                    f"(name='{op.get_name()}') execution failed: {e}") from e
            progress = True
        if not progress:
            # All deferred ops are blocked on missing inputs — diagnostic dump
            missing = []
            for op in deferred:
                for arg, parents in op.get_input_ops().items():
                    for p in parents:
                        nm = p.get_output_tensor().name
                        if nm not in buffers:
                            missing.append(
                                f"    op '{op.get_type()}' (name='{op.get_name()}') "
                                f"waiting on '{nm}' (from '{p.get_type()}')")
            raise RuntimeError(
                f"sg[{sg_idx}] '{sg.get_name()}': topo-execute deadlock — "
                f"{len(deferred)} ops cannot be scheduled. First 5 unmet deps:\n"
                + '\n'.join(missing[:5]))
        pending = deferred


def run_dpu_subgraph(sg, runner, buffers, sg_idx=None):
    in_ts  = runner.get_input_tensors()
    out_ts = runner.get_output_tensors()
    in_arrs = []
    for t in in_ts:
        if t.name not in buffers:
            raise KeyError(f"sg[{sg_idx}] DPU '{sg.get_name()}' "
                           f"input tensor '{t.name}' missing from buffers")
        a = buffers[t.name]
        ex = list(t.dims)
        # Diagnose two failure modes that surface as "tuple index out of range":
        #   - a is 0-dim (some upstream op leaked a Python scalar)
        #   - t.dims is empty (runner input has no shape — unusual)
        # Better to fail loudly with sg_idx + tensor name + shapes than crash
        # with an opaque IndexError 6 frames deep.
        if not ex:
            raise RuntimeError(
                f"sg[{sg_idx}] DPU '{sg.get_name()}': runner input tensor "
                f"'{t.name}' has empty dims; buffer shape={a.shape}")
        if not hasattr(a, 'shape') or a.ndim == 0:
            raise RuntimeError(
                f"sg[{sg_idx}] DPU '{sg.get_name()}': buffer for input "
                f"'{t.name}' is a 0-d scalar (value={a}); runner expects "
                f"dims={ex}. Some upstream CPU op produced a scalar — "
                f"check op_const / op_aten_mean outputs.")
        ex[0] = a.shape[0]
        try:
            a = a.reshape(tuple(ex)).astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"sg[{sg_idx}] DPU '{sg.get_name()}': cannot reshape "
                f"buffer '{t.name}' from {a.shape} to runner's expected "
                f"{tuple(ex)}: {e}") from e
        in_arrs.append(np.ascontiguousarray(a))
    out_arrs = []
    for t in out_ts:
        sh = list(t.dims)
        if sh:
            sh[0] = in_arrs[0].shape[0] if in_arrs else 1
        out_arrs.append(np.empty(tuple(sh), dtype=np.float32))
    job = runner.execute_async(in_arrs, out_arrs)
    runner.wait(job)
    for t, a in zip(out_ts, out_arrs):
        buffers[t.name] = a


# ============================================================
# Main
# ============================================================
def load_eval(npz, n):
    d = np.load(npz)
    sigs = d['signals'][:n].astype(np.float32)
    labs = d['labels'][:n].astype(np.int64)
    if sigs.ndim == 3:
        sigs = sigs[:, np.newaxis, :, :]
    return sigs, labs


def describe_subgraph(sg):
    """One-line human-readable summary of what the subgraph computes,
    derived from its op composition. For the per-subgraph timing print."""
    counts = {}
    for op in sg.get_ops():
        counts[op.get_type()] = counts.get(op.get_type(), 0) + 1
    n_conv = counts.get('conv2d-fix', 0)
    n_mm   = counts.get('matmul', 0)
    n_sm   = counts.get('softmax', 0)
    n_ss   = counts.get('strided_slice-fix', 0)
    n_cat  = counts.get('concat-fix', 0)
    n_fq   = counts.get('float2fix', 0) + counts.get('fix2float', 0)
    if n_mm and n_sm:
        return f'attention ({n_mm} matmul + {n_sm} softmax + {n_ss} slice)'
    if 'aten::mean' in counts:
        return 'GAP (aten::mean) + residual'
    if n_conv > 1:
        return f'{n_conv}× conv2d (q/k/v projections)'
    if n_conv == 1 and counts.get('eltwise-fix', 0) <= 2:
        return '1× conv2d (linear projection or patch_conv)'
    if n_cat:
        return f'head concat ({n_cat}) + {counts.get("reshape-fix", 0)} reshape'
    if 'data-fix' in counts:
        return 'input quantization (data-fix)'
    if counts.get('add', 0) and not n_fq:
        return 'BN counter increment (no data path)'
    if n_fq and n_conv == 0:
        return f'fake-quant chain ({n_fq} fq ops)'
    if 'eltwise-fix' in counts and n_fq == 0:
        return 'eltwise residual / scale'
    return f'mixed ({sum(counts.values())} ops)'


def find_graph_input(subs):
    outs = set()
    for sg in subs:
        for t in sg.get_output_tensors():
            outs.add(t.name)
    inputs = []
    for sg in subs:
        for t in sg.get_input_tensors():
            if t.name not in outs and t.name not in inputs:
                inputs.append(t.name)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmodel', default='/home/petalinux/models/dpu/transformer_radioml/transformer_radioml.xmodel')
    ap.add_argument('--data',   default='/home/petalinux/data/radioml2018_eval_snr_filtered.npz')
    ap.add_argument('--n',      type=int, default=100)
    ap.add_argument('--results', default='/home/petalinux/results')
    ap.add_argument('--debug',  action='store_true',
                    help='Print logits and top-3 predictions for first 3 samples')
    args = ap.parse_args()

    print(f"xmodel: {args.xmodel}")
    graph = xir.Graph.deserialize(args.xmodel)
    subs = graph.get_root_subgraph().toposort_child_subgraph()
    print(f"  {len(subs)} subgraphs")

    # Pre-create runners for DPU subgraphs (skip sg 0 = USER, sg 1 = BN counter)
    sg_dev = []
    runners = [None] * len(subs)
    for i, sg in enumerate(subs):
        dev = sg.get_attr('device') if sg.has_attr('device') else 'UNKNOWN'
        sg_dev.append(dev)
        if i in (0, 1): continue
        if dev == 'DPU':
            try:
                runners[i] = vart.Runner.create_runner(sg, 'run')
            except Exception as e:
                print(f"  sg[{i}] DPU runner FAILED: {e}")

    # Locate the data-fix op in sg[0] (USER) — we apply it manually before sg[2]
    df_op = None
    for op in subs[0].get_ops():
        if op.get_type() == 'data-fix':
            df_op = op; break
    if df_op is None:
        print("  WARN: no data-fix op in sg[0]; passing raw float input")
    else:
        df_t_a = dict(df_op.get_output_tensor().get_attrs())
        df_name = df_op.get_output_tensor().name
        df_fp = df_t_a.get('fix_point', 5)
        df_bw = df_t_a.get('bit_width', 8)
        df_sgn= df_t_a.get('if_signed', True)
        print(f"  sg[0] data-fix: '{df_name}' fp={df_fp} bw={df_bw} signed={df_sgn}")

    g_ins = find_graph_input(subs)
    print(f"  graph input(s): {g_ins}")

    # Load data
    print(f"Loading {args.n} samples from {args.data}")
    sigs, labs = load_eval(args.data, args.n)
    print(f"  shape={sigs.shape}")

    # Inference loop
    correct = 0
    timings = defaultdict(list)
    print(f"\nInference (1 warmup + {args.n - 1} measured)...")
    t_loop = time.perf_counter()
    for it in range(args.n):
        sample = sigs[it:it+1]                  # (1, 1, 1024, 2)
        buf = {}
        # Apply input data-fix manually, store at sg[0]'s output tensor name
        if df_op is not None:
            buf[df_name] = _quantize(sample, df_fp, df_bw, df_sgn).astype(np.float32)
        else:
            for n in g_ins:
                buf[n] = sample.astype(np.float32)

        # Walk subgraphs starting from sg[2]
        for i in range(2, len(subs)):
            t0 = time.perf_counter()
            if runners[i] is not None:
                run_dpu_subgraph(subs[i], runners[i], buf, sg_idx=i)
            elif sg_dev[i] == 'CPU':
                run_cpu_subgraph(subs[i], buf, sg_idx=i)
            t1 = time.perf_counter()
            if it > 0:
                timings[i].append(t1 - t0)

        # Output: last subgraph (sg[20]) produces (1, 24) logits.
        # get_output_tensors() returns a set in this xir build, not a list.
        out_t = next(iter(subs[-1].get_output_tensors()))
        if out_t.name in buf:
            arr = buf[out_t.name]
            pred = int(arr.flatten().argmax()) if arr.size == 24 else int(arr.argmax(axis=-1).flatten()[0])
            if pred == int(labs[it]):
                correct += 1
            if args.debug and it < 3:
                logits = arr.flatten()[:24]
                top3 = np.argsort(logits)[::-1][:3].tolist()
                print(f"\n  [debug] sample {it}: label={int(labs[it])}, "
                      f"pred={pred}, {'CORRECT' if pred == int(labs[it]) else 'WRONG'}")
                print(f"    top-3 (class, logit): "
                      f"{[(int(c), round(float(logits[c]), 3)) for c in top3]}")
                print(f"    full 24 logits: "
                      f"{[round(float(x), 3) for x in logits]}")

    t_loop = time.perf_counter() - t_loop

    # Report
    acc = 100.0 * correct / args.n
    fps = args.n / t_loop
    print(f"\n=== Results ===")
    print(f"  Accuracy:    {acc:.2f}% ({correct}/{args.n})")
    print(f"  Wall-clock:  {t_loop:.2f}s, {fps:.2f} FPS, {1000*t_loop/args.n:.2f} ms/inf")

    dpu_t = cpu_t = 0.0
    print(f"\n  Per-subgraph mean ms (after warmup):")
    print(f"    {'idx':>3s}  {'dev':>4s}  {'ms':>8s}  {'ops':>4s}  description")
    for i in range(len(subs)):
        if i not in timings: continue
        m = float(np.mean(timings[i]))
        if sg_dev[i] == 'DPU': dpu_t += m
        elif sg_dev[i] == 'CPU': cpu_t += m
        desc = describe_subgraph(subs[i])
        print(f"    [{i:2d}]  {sg_dev[i]:>4s}  {m*1000:>8.3f}  "
              f"{len(subs[i].get_ops()):>4d}  {desc}")
    grand = dpu_t + cpu_t
    if grand:
        print(f"\n  DPU: {dpu_t*1000:.3f} ms ({100*dpu_t/grand:.1f}%)")
        print(f"  CPU: {cpu_t*1000:.3f} ms ({100*cpu_t/grand:.1f}%)")

    # JSON output
    os.makedirs(args.results, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.results, f'dpu_transformer_radioml_{ts}.json')
    payload = {
        'config': {
            'toolchain': 'dpu', 'model_path': args.xmodel,
            'dataset': 'radioml2018', 'num_samples': args.n,
            'timestamp': datetime.now().isoformat(),
            'board': 'AUP-ZU3', 'dpu': 'DPUCZDX8G_ISA1_B512',
        },
        'summary': {
            'accuracy': acc, 'throughput_fps': fps,
            'latency_ms_mean': 1000 * t_loop / args.n,
            'wall_seconds': t_loop, 'correct': correct,
            'dpu_seconds_per_inf': dpu_t, 'cpu_seconds_per_inf': cpu_t,
            'dpu_fraction': dpu_t/grand if grand else None,
            'cpu_fraction': cpu_t/grand if grand else None,
        },
        'per_subgraph': [
            {'index': i, 'device': sg_dev[i], 'name': subs[i].get_name(),
             'n_ops': len(subs[i].get_ops()),
             'description': describe_subgraph(subs[i]),
             'mean_seconds': float(np.mean(timings[i])) if i in timings else None}
            for i in range(len(subs))
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
