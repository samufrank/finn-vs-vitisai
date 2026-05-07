"""VART API probe for the multi-DPU-subgraph transformer.

The xmodel has 8 DPU + 12 CPU subgraphs. The existing run_dpu_benchmark
pattern in benchmark.py picks the *first* DPU subgraph and creates a single
runner — that worked for the 1-DPU+2-CPU MLP/CNN xmodels because VART
auto-orchestrated the few CPU reshape ops. For 21 subgraphs with attention
matmuls, softmax, and 16 fake-quant ops on CPU, we need a graph runner
that knows the whole DAG.

This script tries three approaches in order and reports which works:

  A. Single-DPU-subgraph runner (existing pattern).
     Predicted to fail or produce garbage on this xmodel — checking anyway.
  B. Graph runner via vart.Runner.create_runner(root_subgraph, 'run').
     The 'expected' approach for multi-subgraph xmodels.
  C. RunnerExt explicit graph runner.
     Some Vitis AI versions expose this as a separate class.

For each working approach: report runtime, output shape, and the argmax
index of the first sample.

Usage:
    python3 probe_dpu_transformer.py [path_to_xmodel] [path_to_npz]
"""

import os
import sys
import time
import numpy as np

import xir
import vart


XMODEL_DEFAULT = '/home/petalinux/models/dpu/transformer_radioml/transformer_radioml.xmodel'
NPZ_DEFAULT    = '/home/petalinux/data/radioml2018_eval_snr_filtered.npz'

N_PROBE_SAMPLES = 4   # small batch for probe — we just want "does it run"


def list_subgraphs(graph):
    """Print every subgraph and its device assignment."""
    root = graph.get_root_subgraph()
    subs = root.toposort_child_subgraph()
    print(f"\n== {len(subs)} subgraphs (toposort) ==")
    for i, sg in enumerate(subs):
        device = sg.get_attr('device') if sg.has_attr('device') else 'UNKNOWN'
        nops = len(sg.get_ops())
        nm = sg.get_name()
        if len(nm) > 80:
            nm = nm[:77] + '...'
        print(f"  [{device:3s}] [{i:2d}] {nm} ({nops} ops)")
    return root, subs


def load_input(npz_path, n_samples):
    """Return (samples_float32, labels_int64) of the first n_samples or random
    if the npz is unavailable."""
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        sigs = d['signals'][:n_samples].astype(np.float32)
        labs = d['labels'][:n_samples].astype(np.int64)
        if sigs.ndim == 3:
            sigs = sigs[:, np.newaxis, :, :]
        print(f"  loaded {sigs.shape} from {npz_path}")
        print(f"  first {n_samples} labels: {labs.tolist()}")
        return sigs, labs
    print(f"  WARN: {npz_path} not found; using random input")
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_samples, 1, 1024, 2)).astype(np.float32), None


def reshape_for_runner(sample_batch, runner):
    """Reshape a (N, 1, 1024, 2) batch to whatever the runner expects."""
    in_t = runner.get_input_tensors()[0]
    expected = tuple(in_t.dims)
    print(f"  runner input tensor: name={in_t.name} dims={expected} dtype={in_t.dtype}")
    n = sample_batch.shape[0]
    flat_per_sample = int(np.prod(sample_batch.shape[1:]))
    flat_per_runner = int(np.prod(expected[1:]))
    if flat_per_sample != flat_per_runner:
        raise ValueError(
            f"input element-count mismatch: sample has {flat_per_sample}, "
            f"runner expects {flat_per_runner} per sample")
    return sample_batch.reshape((n,) + expected[1:]).astype(np.float32)


def alloc_outputs(runner, batch_size):
    outs = []
    for t in runner.get_output_tensors():
        shape = list(t.dims)
        shape[0] = batch_size
        outs.append(np.empty(tuple(shape), dtype=np.float32))
        print(f"  runner output tensor: name={t.name} dims={shape} dtype={t.dtype}")
    return outs


def run_once(runner, in_arr, out_arrs, label):
    """Run a single batch through `runner`, time it, summarize output."""
    print(f"  Running batch through {label}...")
    t0 = time.perf_counter()
    job = runner.execute_async([np.ascontiguousarray(in_arr)],
                               [np.ascontiguousarray(o) for o in out_arrs])
    runner.wait(job)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed*1000:.2f} ms for batch of {in_arr.shape[0]}")
    for i, o in enumerate(out_arrs):
        flat_per_sample = int(np.prod(o.shape[1:]))
        # If the model is the transformer, output is logits (B, 24).
        if o.ndim == 2 and o.shape[1] == 24:
            preds = o.argmax(axis=1).tolist()
            print(f"  output[{i}] shape={o.shape} argmax={preds}")
        else:
            print(f"  output[{i}] shape={o.shape} mean|.|={np.abs(o).mean():.4f}")
    return elapsed


def try_approach_A_single_dpu(graph, in_batch):
    """Pick first DPU subgraph, run with vart.Runner. Mirrors the existing
    benchmark.py pattern."""
    print("\n===== Approach A: single DPU subgraph (legacy pattern) =====")
    subs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subs = [s for s in subs if s.has_attr('device') and s.get_attr('device') == 'DPU']
    if not dpu_subs:
        print("  no DPU subgraphs found; skipping")
        return False
    print(f"  {len(dpu_subs)} DPU subgraphs total — picking first: {dpu_subs[0].get_name()[:80]}")
    try:
        runner = vart.Runner.create_runner(dpu_subs[0], 'run')
    except Exception as e:
        print(f"  create_runner FAILED: {e}")
        return False
    try:
        in_arr = reshape_for_runner(in_batch, runner)
        out_arrs = alloc_outputs(runner, in_batch.shape[0])
        run_once(runner, in_arr, out_arrs, 'A (single-DPU)')
        return True
    except Exception as e:
        print(f"  execute FAILED: {e}")
        return False
    finally:
        try:
            del runner
        except Exception:
            pass


def try_approach_B_graph_runner(graph, in_batch):
    """vart.Runner.create_runner on the ROOT subgraph (graph runner mode)."""
    print("\n===== Approach B: graph runner via Runner.create_runner(root) =====")
    root = graph.get_root_subgraph()
    try:
        runner = vart.Runner.create_runner(root, 'run')
    except Exception as e:
        print(f"  create_runner(root, 'run') FAILED: {e}")
        return False
    try:
        in_arr = reshape_for_runner(in_batch, runner)
        out_arrs = alloc_outputs(runner, in_batch.shape[0])
        run_once(runner, in_arr, out_arrs, 'B (graph-runner)')
        return True
    except Exception as e:
        print(f"  execute FAILED: {e}")
        return False
    finally:
        try:
            del runner
        except Exception:
            pass


def try_approach_C_runner_ext(graph, in_batch):
    """vart.RunnerExt — some Vitis AI versions expose a separate graph runner
    class that returns Tensor buffers rather than numpy arrays."""
    print("\n===== Approach C: vart.RunnerExt =====")
    if not hasattr(vart, 'RunnerExt'):
        print("  vart.RunnerExt not available in this VART build")
        return False
    root = graph.get_root_subgraph()
    try:
        runner = vart.RunnerExt.create_runner(root, 'run')
    except Exception as e:
        print(f"  RunnerExt.create_runner FAILED: {e}")
        return False
    try:
        # RunnerExt typically uses get_inputs() / get_outputs() returning
        # pre-allocated TensorBuffer objects.
        if hasattr(runner, 'get_inputs') and hasattr(runner, 'get_outputs'):
            inps = runner.get_inputs()
            outs = runner.get_outputs()
            print(f"  RunnerExt: {len(inps)} input buffers, {len(outs)} output buffers")
            # Copy data into the input buffer.
            in_arr = reshape_for_runner(in_batch, runner)
            np.copyto(np.asarray(inps[0]), in_arr)
            t0 = time.perf_counter()
            job = runner.execute_async(inps, outs)
            runner.wait(job)
            elapsed = time.perf_counter() - t0
            print(f"  C (RunnerExt): {elapsed*1000:.2f} ms")
            out_np = np.asarray(outs[0])
            if out_np.ndim == 2 and out_np.shape[1] == 24:
                print(f"  argmax: {out_np.argmax(axis=1).tolist()}")
            return True
        else:
            print("  RunnerExt API doesn't have get_inputs/get_outputs; fallback")
            in_arr = reshape_for_runner(in_batch, runner)
            out_arrs = alloc_outputs(runner, in_batch.shape[0])
            run_once(runner, in_arr, out_arrs, 'C (RunnerExt)')
            return True
    except Exception as e:
        print(f"  execute FAILED: {e}")
        return False
    finally:
        try:
            del runner
        except Exception:
            pass


def main():
    xmodel_path = sys.argv[1] if len(sys.argv) > 1 else XMODEL_DEFAULT
    npz_path    = sys.argv[2] if len(sys.argv) > 2 else NPZ_DEFAULT

    print(f"VART API probe — multi-DPU-subgraph transformer")
    print(f"  xmodel: {xmodel_path}")
    print(f"  data:   {npz_path}")
    if not os.path.exists(xmodel_path):
        print(f"FATAL: xmodel not found")
        sys.exit(2)

    print(f"\n  vart module: {vart.__file__ if hasattr(vart, '__file__') else '?'}")
    print(f"  vart attrs:  {[a for a in dir(vart) if not a.startswith('_')]}")

    graph = xir.Graph.deserialize(xmodel_path)
    list_subgraphs(graph)

    in_batch, labels = load_input(npz_path, N_PROBE_SAMPLES)

    results = {}
    results['A_single_dpu']   = try_approach_A_single_dpu(graph, in_batch)
    results['B_graph_runner'] = try_approach_B_graph_runner(graph, in_batch)
    results['C_runner_ext']   = try_approach_C_runner_ext(graph, in_batch)

    print("\n===== Summary =====")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    if labels is not None:
        print(f"  ground-truth labels for the {N_PROBE_SAMPLES} samples: {labels.tolist()}")

    # Useful follow-up info: any of {VART_PROFILING, DEEPHI_PROFILING, vaitrace}
    # that the user could enable for option-2 (env-var) timing.
    print("\n===== Profiling-tool availability =====")
    for tool in ('xdputil', 'vaitrace'):
        path = os.popen(f'which {tool} 2>/dev/null').read().strip()
        print(f"  {tool}: {path or 'not found'}")
    for env in ('VART_PROFILING', 'DEEPHI_PROFILING', 'XLNX_VART_FIRMWARE'):
        print(f"  ${env} = {os.environ.get(env, '<unset>')}")
    print("\n  To attempt env-based per-subgraph timing, rerun with:")
    print("    VART_PROFILING=1 python3 probe_dpu_transformer.py")
    print("  and inspect stdout for any per-subgraph tracing output.")


if __name__ == '__main__':
    main()
