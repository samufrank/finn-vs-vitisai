"""Per-subgraph timing for the multi-subgraph DPU transformer.

Implements option 3 from the Gate 1 plan (manual subgraph-by-subgraph
execution with time.perf_counter()) as the default path. Options 1 and 2
(xdputil profiling, VART_PROFILING env var) are checked at startup and
reported but not parsed — they're external tools whose output formats
vary by VART version.

Manual iteration approach:
  1. Topo-sort the 21 subgraphs.
  2. For each subgraph, call vart.Runner.create_runner(sg, 'run') —
     VART chooses DPU or CPU backend per subgraph's `device` attribute.
  3. Inputs come from a name-keyed buffer dict populated as upstream
     subgraphs produce outputs. The model's graph input is fed in once
     at the start.
  4. For each call: time the round-trip via time.perf_counter, store
     outputs in the buffer dict by xir tensor name.
  5. Sum per-device times; emit JSON.

This produces unambiguous DPU-vs-CPU runtime fractions and per-stage
breakdown. The trade-off: vart's CPU runners may not match the
performance of graph-runner orchestration (which can pipeline DPU+CPU),
so timing here is an UPPER bound on what graph runner achieves. The
DPU-vs-CPU FRACTIONS are the meaningful number, not absolute throughput.

Usage:
    python3 profile_dpu_subgraphs.py [xmodel] [npz] [n_iterations]

Output:
    /home/petalinux/results/profile_dpu_transformer.json
"""

import os
import sys
import time
import json
import subprocess
from collections import defaultdict

import numpy as np

import xir
import vart


XMODEL_DEFAULT = '/home/petalinux/models/dpu/transformer_radioml/transformer_radioml.xmodel'
NPZ_DEFAULT    = '/home/petalinux/data/radioml2018_eval_snr_filtered.npz'
RESULTS_DIR    = '/home/petalinux/results'
DEFAULT_ITERS  = 32


def check_external_profilers():
    """Check whether xdputil/vaitrace exist and what subcommands they expose.
    We don't actually parse their output — just print availability so the
    user can choose to use them outside this script."""
    print("\n===== External profiler availability =====")
    for tool in ('xdputil', 'vaitrace'):
        path = subprocess.run(['which', tool], capture_output=True,
                              text=True).stdout.strip()
        if not path:
            print(f"  {tool}: not found")
            continue
        print(f"  {tool}: {path}")
        # Run --help and grep for keywords that hint at profiling support
        try:
            help_out = subprocess.run([tool, '--help'], capture_output=True,
                                      text=True, timeout=5)
            txt = (help_out.stdout + help_out.stderr).lower()
            for kw in ('benchmark', 'profile', 'trace', 'run'):
                if kw in txt:
                    print(f"    --help mentions: {kw}")
        except Exception as e:
            print(f"    --help failed: {e}")
    for env in ('VART_PROFILING', 'DEEPHI_PROFILING', 'XLNX_VART_FIRMWARE'):
        print(f"  ${env} = {os.environ.get(env, '<unset>')}")


def load_input_samples(npz_path, n):
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        sigs = d['signals'][:n].astype(np.float32)
        labs = d['labels'][:n].astype(np.int64)
        if sigs.ndim == 3:
            sigs = sigs[:, np.newaxis, :, :]
        return sigs, labs
    print(f"  WARN: {npz_path} not found; using random input")
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 1, 1024, 2)).astype(np.float32), None


def runner_input_names(runner):
    return [t.name for t in runner.get_input_tensors()]


def runner_output_names(runner):
    return [t.name for t in runner.get_output_tensors()]


def alloc_buffer(tensor, batch_size):
    """Allocate a numpy buffer matching tensor's dims, with batch_size in dim 0."""
    shape = list(tensor.dims)
    if shape:
        shape[0] = batch_size
    # vart Python bindings expect float32 buffers regardless of internal int8 —
    # vart handles fix2float / float2fix internally.
    return np.empty(tuple(shape) if shape else (1,), dtype=np.float32)


def make_runners(subgraphs):
    """Try to create a runner for each subgraph. Returns (runners, statuses)
    where statuses[i] is 'ok' or an error message."""
    runners = [None] * len(subgraphs)
    statuses = [None] * len(subgraphs)
    for i, sg in enumerate(subgraphs):
        try:
            runners[i] = vart.Runner.create_runner(sg, 'run')
            statuses[i] = 'ok'
        except Exception as e:
            statuses[i] = f'create_failed: {e}'
    return runners, statuses


def run_one_iteration(subgraphs, runners, statuses, input_array,
                      single_input_name=None):
    """Drive the entire graph for ONE input batch. Times each subgraph call.
    Returns dict {sg_index: elapsed_seconds, ...} plus the final-output dict."""
    buffers = {}
    times = {}
    skipped = []

    # The model's graph input. We need to know what tensor name(s) the FIRST
    # subgraph reads from external — typically just one input that doesn't
    # match any upstream output. Detect it:
    all_outputs = set()
    for sg in subgraphs:
        for t in sg.get_output_tensors():
            all_outputs.add(t.name)
    graph_input_names = []
    for sg in subgraphs:
        for t in sg.get_input_tensors():
            if t.name not in all_outputs and t.name not in graph_input_names:
                graph_input_names.append(t.name)
    # Seed buffers with the model input.
    if single_input_name:
        buffers[single_input_name] = input_array
    elif len(graph_input_names) == 1:
        buffers[graph_input_names[0]] = input_array
    else:
        # Multiple graph inputs — try first one with input_array, others with
        # zeros (transformers don't have multiple external inputs in this build).
        for n in graph_input_names:
            if n not in buffers:
                buffers[n] = input_array if not buffers else np.zeros_like(input_array)

    for i, sg in enumerate(subgraphs):
        runner = runners[i]
        if runner is None:
            skipped.append((i, statuses[i]))
            continue
        # Collect input arrays in the order the runner expects.
        in_arrs = []
        in_tensors = runner.get_input_tensors()
        missing = False
        for t in in_tensors:
            if t.name not in buffers:
                # Not yet computed — possibly an external input we missed,
                # possibly out-of-order. Skip and record.
                skipped.append((i, f'missing_input_tensor: {t.name}'))
                missing = True
                break
            arr = buffers[t.name]
            # Reshape arr to whatever this runner wants (batch dim).
            expected = list(t.dims)
            expected[0] = arr.shape[0]
            try:
                arr = arr.reshape(tuple(expected)).astype(np.float32)
            except Exception:
                # Element count mismatch — leave as-is; vart will complain
                pass
            in_arrs.append(np.ascontiguousarray(arr))
        if missing:
            continue

        # Allocate output buffers.
        out_arrs = [alloc_buffer(t, in_arrs[0].shape[0]) for t in runner.get_output_tensors()]

        try:
            t0 = time.perf_counter()
            job = runner.execute_async(in_arrs, out_arrs)
            runner.wait(job)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            skipped.append((i, f'exec_failed: {e}'))
            continue

        times[i] = elapsed
        for t, arr in zip(runner.get_output_tensors(), out_arrs):
            buffers[t.name] = arr

    return times, buffers, skipped


def aggregate_iterations(per_iter_times, n_iters):
    """Average per-subgraph times over iterations (skip iter 0 as warmup).
    Returns dict {sg_index: mean_seconds}."""
    accumulator = defaultdict(list)
    for it_times in per_iter_times[1:]:        # skip warmup
        for i, t in it_times.items():
            accumulator[i].append(t)
    return {i: float(np.mean(ts)) for i, ts in accumulator.items()}


def main():
    xmodel_path = sys.argv[1] if len(sys.argv) > 1 else XMODEL_DEFAULT
    npz_path    = sys.argv[2] if len(sys.argv) > 2 else NPZ_DEFAULT
    n_iters     = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ITERS

    print(f"Per-subgraph DPU profile")
    print(f"  xmodel: {xmodel_path}")
    print(f"  data:   {npz_path}")
    print(f"  iterations: {n_iters} (1 warmup + {n_iters-1} measured)")

    if not os.path.exists(xmodel_path):
        print(f"FATAL: xmodel not found"); sys.exit(2)

    check_external_profilers()

    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    print(f"\n  Loaded graph: {len(subgraphs)} subgraphs")

    # Build subgraph metadata for the report
    sg_meta = []
    for i, sg in enumerate(subgraphs):
        device = sg.get_attr('device') if sg.has_attr('device') else 'UNKNOWN'
        op_counts = defaultdict(int)
        for op in sg.get_ops():
            op_counts[op.get_type()] += 1
        sg_meta.append({
            'index': i,
            'name': sg.get_name(),
            'device': device,
            'n_ops': len(sg.get_ops()),
            'op_counts': dict(op_counts),
            'input_tensors':  [t.name for t in sg.get_input_tensors()],
            'output_tensors': [t.name for t in sg.get_output_tensors()],
        })

    # Create runners
    print("\n===== Creating per-subgraph runners =====")
    runners, statuses = make_runners(subgraphs)
    n_ok = sum(1 for s in statuses if s == 'ok')
    print(f"  {n_ok}/{len(subgraphs)} runners created")
    for i, s in enumerate(statuses):
        if s != 'ok':
            print(f"    [{i:2d}] {sg_meta[i]['device']}: {s}")

    # Use one sample at a time for clean per-call timing.
    samples, labels = load_input_samples(npz_path, n_iters)
    print(f"  loaded {samples.shape[0]} samples for {n_iters} iterations")

    per_iter_times = []
    per_iter_skipped = []
    per_iter_outputs = []
    print(f"\n===== Running {n_iters} iterations =====")
    for it in range(n_iters):
        sample = samples[it:it+1]      # (1, 1, 1024, 2)
        times, buffers, skipped = run_one_iteration(
            subgraphs, runners, statuses, sample)
        per_iter_times.append(times)
        per_iter_skipped.append(skipped)
        # Record final output for sanity check
        last_outputs = {}
        for sg in subgraphs[::-1]:                      # find a reachable terminal
            outs = [t.name for t in sg.get_output_tensors()]
            if outs and outs[0] in buffers:
                last_outputs = {n: buffers[n] for n in outs}
                break
        per_iter_outputs.append(last_outputs)
        if it == 0:
            total_ms = sum(times.values()) * 1000
            print(f"  iter {it}: total {total_ms:.2f} ms ({len(times)} subgraphs ran, "
                  f"{len(skipped)} skipped)")

    # Aggregate
    mean_times = aggregate_iterations(per_iter_times, n_iters)

    # Per-device totals
    dpu_total = sum(t for i, t in mean_times.items() if sg_meta[i]['device'] == 'DPU')
    cpu_total = sum(t for i, t in mean_times.items() if sg_meta[i]['device'] == 'CPU')
    other_total = sum(t for i, t in mean_times.items()
                      if sg_meta[i]['device'] not in ('DPU', 'CPU'))
    grand_total = dpu_total + cpu_total + other_total

    print("\n===== Per-subgraph mean timing (ms, after warmup) =====")
    print(f"  {'idx':>3s}  {'device':>6s}  {'ops':>4s}  {'ms/inf':>8s}  {'%':>6s}  name")
    for i in range(len(subgraphs)):
        if i not in mean_times:
            print(f"  [{i:2d}]  {sg_meta[i]['device']:>6s}  {sg_meta[i]['n_ops']:>4d}  "
                  f"{'SKIP':>8s}  {'':>6s}  {sg_meta[i]['name'][:60]}")
            continue
        t = mean_times[i]
        pct = 100 * t / grand_total if grand_total else 0
        print(f"  [{i:2d}]  {sg_meta[i]['device']:>6s}  {sg_meta[i]['n_ops']:>4d}  "
              f"{t*1000:>8.3f}  {pct:>5.1f}%  {sg_meta[i]['name'][:60]}")

    print("\n===== Per-device totals =====")
    print(f"  DPU:    {dpu_total*1000:8.3f} ms  ({100*dpu_total/grand_total:.1f}% of total)")
    print(f"  CPU:    {cpu_total*1000:8.3f} ms  ({100*cpu_total/grand_total:.1f}% of total)")
    if other_total > 0:
        print(f"  Other:  {other_total*1000:8.3f} ms  ({100*other_total/grand_total:.1f}%)")
    print(f"  TOTAL:  {grand_total*1000:8.3f} ms  ({1.0/grand_total:.1f} FPS upper bound)")

    # Report sample predictions for sanity
    if labels is not None and per_iter_outputs:
        # Look for a (1, 24) output in the last iter
        for nm, arr in per_iter_outputs[-1].items():
            if arr.ndim == 2 and arr.shape[1] == 24:
                pred = int(arr.argmax(axis=1)[0])
                print(f"\n  Last-iter pred: {pred}, label: {labels[n_iters-1]}, "
                      f"output tensor: {nm}")
                break

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'profile_dpu_transformer.json')
    payload = {
        'xmodel': xmodel_path,
        'iterations_total': n_iters,
        'iterations_measured': n_iters - 1,
        'subgraphs': [
            {
                **sg_meta[i],
                'mean_seconds': mean_times.get(i),
                'status': statuses[i],
            }
            for i in range(len(subgraphs))
        ],
        'totals': {
            'dpu_seconds': dpu_total,
            'cpu_seconds': cpu_total,
            'other_seconds': other_total,
            'grand_total_seconds': grand_total,
            'dpu_fraction': dpu_total / grand_total if grand_total else 0,
            'cpu_fraction': cpu_total / grand_total if grand_total else 0,
        },
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
