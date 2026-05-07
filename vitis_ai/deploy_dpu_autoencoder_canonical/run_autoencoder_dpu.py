#!/usr/bin/env python3
"""run_autoencoder_dpu.py — board-side AUC + power-ready DPU runner for the
canonical FC autoencoder.

Designed to run on the AUP-ZU3 PetaLinux SD card. Emits the same top-level
JSON shape as `board/benchmark.py` (`config`, `idle`, `runs[]`, `summary`)
so that `board/merge_power.py` can ingest the FNB58 power CSV against this
output directly — no schema fork. AUC is reported in the `accuracy` field
(scaled to [0, 100]) so merge_power.py's existing summary fields populate.
Per-recording AUC + per-machine AUC + windows-per-recording aggregation
go into auxiliary fields under each run and the summary.

Steps per window: standardize → DPU forward → reconstruction MSE.
Per-recording aggregation: mean MSE → AUC-ROC overall and per machine ID.

Usage on board:
    cd ~/deploy_dpu_autoencoder_canonical
    sudo python3 run_autoencoder_dpu.py
        [--runs 5] [--idle 10] [--stabilize 10] [--output board_auc_run.json]
"""
import argparse, hashlib, json, os, time
from datetime import datetime
import numpy as np
import xir, vart
from sklearn.metrics import roc_auc_score


HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 16), b''):
            h.update(blk)
    return h.hexdigest()


def per_window_mse(runner, test_x_std, in_shape, out_shape):
    """Run all windows through the DPU; return float32 MSE per window."""
    n = len(test_x_std)
    mse = np.empty(n, dtype=np.float32)
    out = np.empty(out_shape, dtype=np.float32)
    for i in range(n):
        inp = np.ascontiguousarray(test_x_std[i].reshape(in_shape).astype(np.float32))
        jid = runner.execute_async([inp], [out])
        runner.wait(jid)
        recon = out.flatten()
        mse[i] = float(np.mean((recon - test_x_std[i]) ** 2))
    return mse


def aggregate_auc(per_window, test_rid, test_lbl, test_mid):
    """Per-recording mean MSE → AUC (overall + per-machine)."""
    n_rec = len(test_lbl)
    per_rec = np.zeros(n_rec, dtype=np.float64)
    counts  = np.zeros(n_rec, dtype=np.int64)
    np.add.at(per_rec, test_rid, per_window)
    np.add.at(counts,  test_rid, 1)
    per_rec = per_rec / np.maximum(counts, 1)
    auc_overall = float(roc_auc_score(test_lbl, per_rec))
    aucs_pm = {}
    for mid in sorted(np.unique(test_mid).tolist()):
        mask = test_mid == mid
        if (test_lbl[mask] == 0).sum() == 0 or (test_lbl[mask] == 1).sum() == 0:
            continue
        aucs_pm[int(mid)] = float(roc_auc_score(test_lbl[mask], per_rec[mask]))
    return auc_overall, aucs_pm, per_rec


def run_once(runner, run_num, test_x_std, test_rid, test_lbl, test_mid,
             in_shape, out_shape, n_windows, n_recordings):
    """One measured run. Returns a dict matching benchmark.py:build_run_result()."""
    t_start = time.time()
    mse = per_window_mse(runner, test_x_std, in_shape, out_shape)
    elapsed = time.time() - t_start
    t_end = time.time()

    auc_overall, aucs_pm, _ = aggregate_auc(mse, test_rid, test_lbl, test_mid)
    accuracy = 100.0 * auc_overall

    result = {
        'run':            run_num,
        't_start':        t_start,
        't_end':          t_end,
        'accuracy':       accuracy,                # AUC*100, fills merge_power's "accuracy"
        'time_s':         elapsed,
        'throughput_fps': n_windows / elapsed,     # windows/s
        'latency_ms':     1000.0 * elapsed / n_windows,
        # On-board onboard-power sampling is skipped (FNB58 is the primary
        # path on the AUP-ZU3 PetaLinux card). merge_power.py fills in
        # avg_power_w / energy_total_j / energy_per_image_mj from the FNB58
        # CSV based on t_start/t_end.
        'avg_power_w':         None,
        'energy_total_j':      None,
        'energy_per_image_mj': None,
        'power_samples':       0,
        'sysmon':              None,
        # Autoencoder-specific. merge_power.py ignores these; downstream
        # plotting + analysis read them.
        'auc_overall':     auc_overall,
        'auc_per_machine': aucs_pm,
        'n_windows':       int(n_windows),
        'n_recordings':    int(n_recordings),
    }
    print(f"  Run {run_num}: AUC={auc_overall:.4f} acc={accuracy:5.2f}%  "
          f"{n_windows/elapsed:7.1f} win/s  ({elapsed:.2f}s)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs',      type=int, default=5)
    ap.add_argument('--idle',      type=int, default=10,
                    help='Idle window seconds before measured runs.')
    ap.add_argument('--stabilize', type=int, default=10,
                    help='Sleep seconds before idle window (thermal stabilization).')
    ap.add_argument('--warmup',    type=int, default=200,
                    help='Warmup window count before measured runs.')
    ap.add_argument('--output',    default=None,
                    help='Output JSON path (default: board_auc_run_<timestamp>.json).')
    args = ap.parse_args()

    xmodel = os.path.join(HERE, 'autoencoder_canonical_toycar_int8.xmodel')
    print(f"xmodel: {xmodel}")
    print(f"  sha256: {sha256(xmodel)}")

    mean = np.load(os.path.join(HERE, 'input_mean.npy')).astype(np.float32)
    std  = np.load(os.path.join(HERE, 'input_std.npy')).astype(np.float32)

    sub = os.path.join(HERE, 'eval_subset')
    test_x   = np.load(os.path.join(sub, 'test_features.npy')).astype(np.float32)
    test_rid = np.load(os.path.join(sub, 'test_recording_ids.npy'))
    test_lbl = np.load(os.path.join(sub, 'test_recording_labels.npy'))
    test_mid = np.load(os.path.join(sub, 'test_recording_machine_ids.npy'))
    n_windows    = len(test_x)
    n_recordings = len(test_lbl)
    print(f"eval subset: {n_windows} windows / {n_recordings} recordings "
          f"({int((test_lbl==0).sum())} normal, {int((test_lbl==1).sum())} anomaly)")

    test_x_std = ((test_x - mean) / np.maximum(std, 1e-6)).astype(np.float32)

    g = xir.Graph.deserialize(xmodel)
    sg = [s for s in g.get_root_subgraph().toposort_child_subgraph()
          if s.has_attr('device') and s.get_attr('device') == 'DPU'][0]
    runner = vart.Runner.create_runner(sg, 'run')
    in_t  = runner.get_input_tensors()[0]
    out_t = runner.get_output_tensors()[0]
    in_shape  = tuple(in_t.dims)
    out_shape = tuple(out_t.dims)
    print(f"DPU input:  {in_t.name}  shape={in_shape}  dtype={in_t.dtype}")
    print(f"DPU output: {out_t.name}  shape={out_shape}  dtype={out_t.dtype}")

    # Config — same field names as benchmark.py:run_dpu_benchmark for
    # tooling compatibility.
    config = {
        'toolchain':         'dpu',
        'task':              'autoencoder',
        'model_path':        xmodel,
        'dataset':           'toycar_dcase2020',
        'batch_size':        1,
        'num_runs':          args.runs,
        'num_images':        n_windows,                # one inference per window
        'image_shape':       [int(test_x_std.shape[1])],
        'dpu_input_shape':   list(in_shape),
        'dpu_output_shape':  list(out_shape),
        'timestamp':         datetime.now().isoformat(),
        'board':             'AUP-ZU3',
        'dpu':               'DPUCZDX8G_ISA1_B512',
        'power_method':      'fnb58_external',
        'eval_subset': {
            'n_windows':    n_windows,
            'n_recordings': n_recordings,
            'n_normal':     int((test_lbl == 0).sum()),
            'n_anomaly':    int((test_lbl == 1).sum()),
            'machine_ids':  sorted(np.unique(test_mid).tolist()),
        },
        'standardization': {
            'mean_npy': 'input_mean.npy',
            'std_npy':  'input_std.npy',
        },
    }

    print(f"Thermal stabilization ({args.stabilize}s)...")
    time.sleep(args.stabilize)

    print(f"Measuring idle ({args.idle}s)...")
    idle_t_start = time.time()
    time.sleep(args.idle)
    idle_t_end = time.time()
    idle = {
        't_start': idle_t_start,
        't_end':   idle_t_end,
        # No on-board INA260/sysmon sampling. merge_power.py fills in
        # idle_power from the FNB58 CSV.
        'power':  {'mean': None, 'std': None, 'n_samples': 0},
        'sysmon': {'temp_ps_c': None, 'temp_pl_c': None, 'vccint_v': None,
                   'n_samples': 0},
    }

    print(f"Warmup ({args.warmup} windows)...")
    out = np.empty(out_shape, dtype=np.float32)
    for i in range(min(args.warmup, n_windows)):
        inp = np.ascontiguousarray(test_x_std[i].reshape(in_shape).astype(np.float32))
        jid = runner.execute_async([inp], [out])
        runner.wait(jid)

    print(f"Running {args.runs} measured runs...")
    all_runs = []
    for run in range(args.runs):
        all_runs.append(run_once(
            runner, run + 1, test_x_std, test_rid, test_lbl, test_mid,
            in_shape, out_shape, n_windows, n_recordings))
    del runner

    # Summary — same field names as benchmark.py:save_results so merge_power
    # populates the same keys. Power-related fields stay None until
    # merge_power.py fills them in from FNB58.
    has_power = False  # always False here; merge_power.py owns power numbers
    summary = {
        'accuracy':            float(np.mean([r['accuracy']       for r in all_runs])),
        'throughput_fps_mean': float(np.mean([r['throughput_fps'] for r in all_runs])),
        'throughput_fps_std':  float(np.std( [r['throughput_fps'] for r in all_runs])),
        'latency_ms_mean':     float(np.mean([r['latency_ms']     for r in all_runs])),
        'latency_ms_std':      float(np.std( [r['latency_ms']     for r in all_runs])),
        'idle_power_w':        idle['power']['mean'],
        'idle_power_std':      idle['power']['std'],
        'idle_temp_pl_c':      idle['sysmon']['temp_pl_c'],
        'avg_power_w_mean':    None,
        'avg_power_w_std':     None,
        'dynamic_power_w':     None,
        'energy_per_image_mj_mean': None,
        'energy_per_image_mj_std':  None,
        # AUC summary — auxiliary, merge_power ignores.
        'auc_overall_mean': float(np.mean([r['auc_overall'] for r in all_runs])),
        'auc_overall_std':  float(np.std( [r['auc_overall'] for r in all_runs])),
        'auc_per_machine_mean': {
            int(mid): float(np.mean([r['auc_per_machine'].get(mid, np.nan)
                                     for r in all_runs
                                     if mid in r['auc_per_machine']]))
            for mid in sorted({mid for r in all_runs for mid in r['auc_per_machine']})
        },
    }

    output = {'config': config, 'idle': idle, 'runs': all_runs, 'summary': summary}

    out_path = args.output or os.path.join(
        HERE, f"board_auc_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  AUC overall:    {summary['auc_overall_mean']:.4f} +/- {summary['auc_overall_std']:.4f}")
    print(f"  Throughput:     {summary['throughput_fps_mean']:.1f} +/- {summary['throughput_fps_std']:.1f} win/s")
    print(f"  Latency/window: {summary['latency_ms_mean']:.3f} +/- {summary['latency_ms_std']:.3f} ms")
    for mid, a in sorted(summary['auc_per_machine_mean'].items()):
        print(f"  machine {mid}:    {a:.4f}")
    print(f"  Power: collected host-side (FNB58); run merge_power.py to populate")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
