#!/usr/bin/env python3
"""sweep_driver.py — target_fps sweep across (model, precision) on ZU3EG.

See results/finn/target_fps_sweep/README.md for methodology and resume procedure.

Phase 1: try target_fps in {1000, 10000, 100000, 500000} sequentially. Stop on
the first resource_fail or timing_fail (Phase 1 ends, ceiling bracketed). Stop
on tool_fail too (sweep marked partial; no Phase 2 run).

Phase 2: 3 log-spaced target_fps values between lo_pass and hi_fail.

Usage:
  python finn/sweep_driver.py                       # full sweep, all four combos
  python finn/sweep_driver.py --only mlp_int8       # one sweep
  python finn/sweep_driver.py --only mlp_int8 --target 1000   # one build (dry-run / fill-in)
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse resource_summary.py from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resource_summary

# ---- Paths -------------------------------------------------------------------

REPO_ROOT       = Path(__file__).resolve().parent.parent           # finn-vs-vitisai/
FINN_DIR        = REPO_ROOT / 'finn'
SWEEP_RUNS_DIR  = FINN_DIR / 'target_fps_sweep_runs'
RESULTS_DIR     = REPO_ROOT / 'results' / 'finn' / 'target_fps_sweep'
STATE_PATH      = RESULTS_DIR / 'sweep_state.json'
CSV_PATH        = RESULTS_DIR / 'resource_summary.csv'
DRIVER_LOG      = RESULTS_DIR / 'sweep_driver.log'
TOOL_FAILS_TXT  = RESULTS_DIR / 'tool_fails.txt'

# ---- Docker / FINN env -------------------------------------------------------

DOCKER_TAG     = 'xilinx/finn:v0.10.1-215-gd90c0878.xrt_202220.2.14.354_22.04-amd64-xrt'
FINN_BUILD_DIR = '/tmp/finn_dev_samu'
FINN_REPO_HOST = str(REPO_ROOT / 'finn-repo')
PROJECT_ROOT   = str(REPO_ROOT)

# ---- Sweep definitions -------------------------------------------------------

SWEEPS = {
    'mlp_int8': {'onnx': 'mlp_mnist_tiny.onnx'},
    'mlp_int4': {'onnx': 'mlp_mnist_tiny_int4.onnx'},
    'cnn_int8': {'onnx': 'cnn_mnist_tiny.onnx'},
    'cnn_int4': {'onnx': 'cnn_mnist_tiny_int4.onnx'},
}
SWEEP_ORDER     = ['mlp_int8', 'mlp_int4', 'cnn_int8', 'cnn_int4']
PHASE1_TARGETS  = [1000, 10000, 100000, 500000]
BUILD_TIMEOUT_S = 10800                          # 3 h hard cap per build

# Vivado place/route resource error patterns (case-insensitive substring match).
RESOURCE_ERROR_PATTERNS = [
    'unable to place',
    'utilization exceeded',
    'route_design failed',
    '[place 30-',
    '[route 35-',
    'insufficient resources',
    'overlap of placement',
    'placement is impossible',
    'too many lut',
    'too many bram',
    'too many dsp',
]

CSV_HEADER = [
    'sweep', 'target_fps', 'phase', 'status', 'started_iso', 'elapsed_s', 'last_step',
    'wns_ns', 'fmax_mhz',
    'lut', 'lut_pct', 'ff', 'bram18', 'bram18_pct', 'dsp', 'dsp_pct',
    'est_fps_fpga', 'est_latency_us', 'bitstream_mb',
    'folding_json', 'cpu_layers_json',
    'log_path', 'error_excerpt',
]

# =============================================================================
# State + CSV + log helpers (atomic writes, idempotent init)
# =============================================================================

def now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def driver_log(msg):
    line = f'[{now_iso()}] {msg}'
    print(line, flush=True)
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIVER_LOG, 'a') as f:
        f.write(line + '\n')


def init_state():
    return {
        'schema_version': 1,
        'started':        now_iso(),
        'last_update':    now_iso(),
        'current_sweep':  None,
        'current_target': None,
        'sweeps': {
            sweep_name: {
                'status':            'not_started',
                'phase':             'phase1',
                'model_onnx':        SWEEPS[sweep_name]['onnx'],
                'phase1_targets':    list(PHASE1_TARGETS),
                'ceiling_lo':        None,
                'ceiling_hi':        None,
                'phase2_targets':    [],
                'completed_builds':  {},
                'tool_fails':        [],
            }
            for sweep_name in SWEEP_ORDER
        },
    }


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return init_state()


def save_state(state):
    state['last_update'] = now_iso()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)


def init_csv():
    if CSV_PATH.exists():
        return
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def append_csv_row(row_dict):
    full_row = {k: row_dict.get(k, '') for k in CSV_HEADER}
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerow(full_row)


# =============================================================================
# Docker invocation
# =============================================================================

def docker_cmd(onnx, target_fps, output_dir_in_container):
    """Build a docker run argv list. Mirrors the Gate-4 working invocation."""
    return [
        'docker', 'run', '--rm', '--init', '--hostname', 'finn_dev_samu',
        '-e', 'SHELL=/bin/bash',
        '-w', FINN_REPO_HOST,
        '-v', f'{FINN_REPO_HOST}:{FINN_REPO_HOST}',
        '-v', f'{FINN_BUILD_DIR}:{FINN_BUILD_DIR}',
        '-e', f'FINN_BUILD_DIR={FINN_BUILD_DIR}',
        '-e', f'FINN_ROOT={FINN_REPO_HOST}',
        '-e', f'VIVADO_IP_CACHE={FINN_BUILD_DIR}/vivado_ip_cache',
        '-e', 'NUM_DEFAULT_WORKERS=4',
        '-e', 'LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1',
        '-v', '/etc/group:/etc/group:ro',
        '-v', '/etc/passwd:/etc/passwd:ro',
        '-v', '/etc/shadow:/etc/shadow:ro',
        '--user', '1000:1000',
        '-v', '/tools/Xilinx:/tools/Xilinx',
        '-e', 'XILINX_VIVADO=/tools/Xilinx/Vivado/2022.2',
        '-e', 'VIVADO_PATH=/tools/Xilinx/Vivado/2022.2',
        '-e', 'HLS_PATH=/tools/Xilinx/Vitis_HLS/2022.2',
        '-v', f'{PROJECT_ROOT}:/workspace/project',
        DOCKER_TAG,
        'bash', '-c',
        f'cd /workspace/project/finn && '
        f'python compile.py --model {onnx} --fps {target_fps} '
        f'--board Ultra96 --output {output_dir_in_container}',
    ]


def clean_finn_cache():
    """Stomp HLS-IP and stitched-IP caches between builds.
    Keep vivado_ip_cache (legitimately speeds up identical IP regen)."""
    if not os.path.isdir(FINN_BUILD_DIR):
        return
    for entry in os.listdir(FINN_BUILD_DIR):
        if entry.startswith('code_gen_ipgen_') or entry.startswith('vivado_stitch_proj_'):
            full = os.path.join(FINN_BUILD_DIR, entry)
            try:
                shutil.rmtree(full)
            except Exception as e:
                driver_log(f'WARN clean_finn_cache: could not remove {full}: {e}')


# =============================================================================
# Build classification — exit code + stdout markers + artifact checks
# =============================================================================

def parse_last_step(log_text):
    last = None
    for m in re.finditer(r'Running step: (step_\w+) \[(\d+)/19\]', log_text):
        last = m.group(1)
    return last


def find_first_error_excerpt(log_text):
    m = re.search(r'(ERROR: \[[^\n]{0,200}\][^\n]{0,300})', log_text)
    if m:
        return m.group(1)[:500]
    idx = log_text.find('Build failed')
    if idx >= 0:
        start = max(0, idx - 600)
        return log_text[start:idx + 50][-500:]
    idx = log_text.find('Traceback (most recent call last):')
    if idx >= 0:
        return log_text[idx:idx + 500]
    return ''


def is_resource_error(log_text):
    low = log_text.lower()
    return any(p in low for p in RESOURCE_ERROR_PATTERNS)


def classify(log_text, output_dir, returncode):
    """Return (status, last_step, error_excerpt).

    Cross-validates exit code, stdout markers, and artifact presence.
    Disagreements fall through to tool_fail.
    """
    last_step = parse_last_step(log_text)
    has_complete = 'Completed successfully' in log_text
    has_failed = 'Build failed' in log_text
    bit_dir = os.path.join(output_dir, 'deploy', 'bitfile')
    bitfile_present = (os.path.isdir(bit_dir) and
                       any(f.endswith('.bit') for f in os.listdir(bit_dir)))

    # Happy path: rc=0, marker present, artifact present.
    if returncode == 0 and has_complete and bitfile_present:
        wns = resource_summary.parse_wns(
            os.path.join(output_dir, 'report', 'post_route_timing.rpt'))
        if wns is None:
            return 'tool_fail', last_step, 'completed but no timing report'
        if wns < 0:
            return 'timing_fail', last_step, f'WNS={wns}ns'
        return 'success', last_step, ''

    # Failure path.
    if has_failed or returncode != 0:
        excerpt = find_first_error_excerpt(log_text)
        in_synth_step = last_step in (
            'step_synthesize_bitfile', 'step_out_of_context_synthesis')
        if in_synth_step and is_resource_error(log_text):
            return 'resource_fail', last_step, excerpt
        return 'tool_fail', last_step, excerpt

    # No marker, rc=0 but no artifact: confused state.
    return 'tool_fail', last_step, 'no markers found; rc=0 but bitfile missing'


def collect_metrics(output_dir):
    """Summarize a build, write per-build resource_report.json + .md, return flat dict.
    Returns None on summarize error."""
    try:
        rep = resource_summary.summarize(output_dir)
    except Exception as e:
        driver_log(f'WARN collect_metrics({output_dir}): {e}')
        return None
    # Mirror resource_summary.main()'s archive: JSON + markdown alongside the build.
    try:
        with open(os.path.join(output_dir, 'resource_report.json'), 'w') as f:
            json.dump(rep, f, indent=2)
        with open(os.path.join(output_dir, 'resource_summary.md'), 'w') as f:
            f.write(resource_summary.render_markdown(rep))
    except Exception as e:
        driver_log(f'WARN collect_metrics({output_dir}) write artifacts: {e}')
    util   = rep.get('utilization') or {}
    timing = rep.get('timing') or {}
    perf   = rep.get('estimate_performance') or {}
    bit    = rep.get('bitfile') or {}
    lat_ns = perf.get('estimated_latency_ns')
    return {
        'wns_ns':         timing.get('wns_ns'),
        'fmax_mhz':       timing.get('fmax_mhz'),
        'lut':            util.get('LUT'),
        'lut_pct':        util.get('LUT_pct'),
        'ff':             util.get('FF'),
        'bram18':         util.get('BRAM_18K_equiv'),
        'bram18_pct':     util.get('BRAM_18K_pct'),
        'dsp':            util.get('DSP'),
        'dsp_pct':        util.get('DSP_pct'),
        'est_fps_fpga':   perf.get('estimated_throughput_fps'),
        'est_latency_us': (lat_ns / 1000.0) if lat_ns else None,
        'bitstream_mb':   bit.get('mb'),
        'folding':        rep.get('folding') or {},
        'cpu_layers':     rep.get('cpu_partition_layers') or [],
    }


# =============================================================================
# Phase logic
# =============================================================================

def highest_pass(sweep_state):
    """Highest target_fps that compiled (success preferred; timing_fail fallback)."""
    builds = sweep_state['completed_builds']
    successes = [int(t) for t, b in builds.items() if b.get('status') == 'success']
    if successes:
        return max(successes)
    timing = [int(t) for t, b in builds.items() if b.get('status') == 'timing_fail']
    if timing:
        return max(timing)
    return None


def compute_phase2_targets(lo, hi):
    """Three log-spaced points strictly between lo and hi, rounded to nice numbers.

    Geometric: lo * (hi/lo)^(k/4) for k in {1, 2, 3}.
    Rounding: <5000 → nearest 100; 5000-50000 → 500; >=50000 → 1000.
    Drops duplicates and any point that rounds onto lo or hi.
    """
    if lo is None or hi is None or hi <= lo:
        return []
    raw = [lo * (hi / lo) ** (k / 4.0) for k in (1, 2, 3)]

    def nice_round(t):
        if t < 5000:
            return int(round(t / 100) * 100)
        if t < 50000:
            return int(round(t / 500) * 500)
        return int(round(t / 1000) * 1000)

    seen = {lo, hi}
    out = []
    for t in raw:
        nt = nice_round(t)
        if nt in seen or nt <= lo or nt >= hi:
            continue
        seen.add(nt)
        out.append(nt)
    return sorted(out)


# =============================================================================
# Build runner — single target
# =============================================================================

def run_one_build(state, sweep, target, phase):
    """Compile + classify + persist one (sweep, target_fps) build.
    Returns the resulting status string."""
    sweep_state = state['sweeps'][sweep]
    if sweep_state['status'] == 'not_started':
        sweep_state['status'] = 'in_progress'

    onnx = sweep_state['model_onnx']
    output_dir_host = SWEEP_RUNS_DIR / f'{sweep}_fps{target}'
    output_dir_container = (
        f'/workspace/project/finn/target_fps_sweep_runs/{sweep}_fps{target}'
    )
    log_path = output_dir_host / 'build.log'

    # Clean any prior partial output that wasn't a recorded completion.
    prior = sweep_state['completed_builds'].get(str(target))
    if (prior is None) or prior.get('status') == 'in_progress':
        if output_dir_host.exists():
            driver_log(f'cleaning prior partial output: {output_dir_host}')
            shutil.rmtree(output_dir_host)

    output_dir_host.mkdir(parents=True, exist_ok=True)
    clean_finn_cache()

    # Mark in-progress before launching docker.
    started_iso = now_iso()
    sweep_state['completed_builds'][str(target)] = {
        'status':  'in_progress',
        'started': started_iso,
    }
    state['current_sweep']  = sweep
    state['current_target'] = target
    save_state(state)

    driver_log(f'BEGIN {sweep} fps={target} phase={phase} onnx={onnx}')

    cmd = docker_cmd(onnx, target, output_dir_container)
    start = time.time()
    timed_out = False
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=BUILD_TIMEOUT_S, text=True)
        rc  = result.returncode
        out = (result.stdout or '') + '\n' + (result.stderr or '')
    except subprocess.TimeoutExpired as e:
        rc        = -9
        out       = (e.stdout or '') + '\n' + (e.stderr or '') + '\n[TIMEOUT after 3h]'
        timed_out = True
    elapsed_s = time.time() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(out)

    if timed_out:
        status, last_step, error_excerpt = (
            'tool_fail', parse_last_step(out), 'TIMEOUT after 3h')
    else:
        status, last_step, error_excerpt = classify(out, str(output_dir_host), rc)

    metrics = None
    if status in ('success', 'timing_fail'):
        metrics = collect_metrics(str(output_dir_host))

    completed_iso = now_iso()
    build_record = {
        'status':        status,
        'started':       started_iso,
        'completed':     completed_iso,
        'elapsed_s':     round(elapsed_s, 1),
        'last_step':     last_step,
        'error_excerpt': error_excerpt,
        'returncode':    rc,
        'log_path':      str(log_path.relative_to(REPO_ROOT)),
    }
    if metrics:
        build_record.update(metrics)

    sweep_state['completed_builds'][str(target)] = build_record
    if status == 'tool_fail':
        if str(target) not in sweep_state['tool_fails']:
            sweep_state['tool_fails'].append(str(target))
        TOOL_FAILS_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(TOOL_FAILS_TXT, 'a') as f:
            f.write(f'[{completed_iso}] {sweep} fps={target} '
                    f'last_step={last_step} excerpt={(error_excerpt or "")[:200]}\n')
    save_state(state)

    csv_row = {
        'sweep':         sweep,
        'target_fps':    target,
        'phase':         phase,
        'status':        status,
        'started_iso':   started_iso,
        'elapsed_s':     round(elapsed_s, 1),
        'last_step':     last_step or '',
        'log_path':      str(log_path.relative_to(REPO_ROOT)),
        'error_excerpt': (error_excerpt or '')[:500],
    }
    if metrics:
        csv_row.update({
            'wns_ns':         metrics['wns_ns'],
            'fmax_mhz':       metrics['fmax_mhz'],
            'lut':            metrics['lut'],
            'lut_pct':        metrics['lut_pct'],
            'ff':             metrics['ff'],
            'bram18':         metrics['bram18'],
            'bram18_pct':     metrics['bram18_pct'],
            'dsp':            metrics['dsp'],
            'dsp_pct':        metrics['dsp_pct'],
            'est_fps_fpga':   metrics['est_fps_fpga'],
            'est_latency_us': metrics['est_latency_us'],
            'bitstream_mb':   metrics['bitstream_mb'],
            'folding_json':   json.dumps(metrics['folding']),
            'cpu_layers_json': json.dumps(metrics['cpu_layers']),
        })
    append_csv_row(csv_row)

    driver_log(f'END   {sweep} fps={target} phase={phase} '
               f'status={status} elapsed={elapsed_s:.0f}s last_step={last_step}')
    return status


# =============================================================================
# Phase 1 / Phase 2 / sweep orchestration
# =============================================================================

def run_phase1(state, sweep):
    sweep_state = state['sweeps'][sweep]
    sweep_state['status'] = 'in_progress'
    save_state(state)

    for target in PHASE1_TARGETS:
        existing = sweep_state['completed_builds'].get(str(target))
        if existing and existing.get('status') not in (None, 'in_progress'):
            # Already attempted in a previous run — reuse the result.
            est = existing['status']
            if est in ('resource_fail', 'timing_fail'):
                lo = highest_pass(sweep_state)
                sweep_state['ceiling_lo']     = lo
                sweep_state['ceiling_hi']     = target
                sweep_state['phase2_targets'] = (
                    compute_phase2_targets(lo, target) if lo is not None else [])
                sweep_state['phase'] = 'phase2'
                save_state(state)
                return
            if est == 'tool_fail':
                sweep_state['phase']  = 'tool_failed'
                sweep_state['status'] = 'partial'
                save_state(state)
                return
            # success → continue
            continue

        status = run_one_build(state, sweep, target, phase='phase1')

        if status in ('resource_fail', 'timing_fail'):
            lo = highest_pass(sweep_state)
            sweep_state['ceiling_lo']     = lo
            sweep_state['ceiling_hi']     = target
            sweep_state['phase2_targets'] = (
                compute_phase2_targets(lo, target) if lo is not None else [])
            sweep_state['phase'] = 'phase2'
            save_state(state)
            return

        if status == 'tool_fail':
            sweep_state['phase']  = 'tool_failed'
            sweep_state['status'] = 'partial'
            save_state(state)
            return

    # All four passed.
    sweep_state['ceiling_lo']     = PHASE1_TARGETS[-1]
    sweep_state['ceiling_hi']     = None
    sweep_state['phase2_targets'] = []
    sweep_state['phase']          = 'done'
    sweep_state['status']         = 'completed'
    save_state(state)


def run_phase2(state, sweep):
    sweep_state = state['sweeps'][sweep]
    if not sweep_state['phase2_targets']:
        sweep_state['phase']  = 'done'
        sweep_state['status'] = 'completed'
        save_state(state)
        return

    for target in sweep_state['phase2_targets']:
        existing = sweep_state['completed_builds'].get(str(target))
        if existing and existing.get('status') not in (None, 'in_progress'):
            continue
        run_one_build(state, sweep, target, phase='phase2')

    sweep_state['phase']  = 'done'
    sweep_state['status'] = 'completed'
    save_state(state)


def run_sweep(state, sweep):
    sweep_state = state['sweeps'][sweep]
    if sweep_state['status'] == 'completed':
        driver_log(f'skip {sweep}: already completed')
        return
    if sweep_state['phase'] == 'tool_failed':
        driver_log(f'skip {sweep}: tool_failed (manual review required)')
        return

    if sweep_state['phase'] in ('phase1', None):
        run_phase1(state, sweep)

    sweep_state = state['sweeps'][sweep]
    if sweep_state['phase'] == 'phase2':
        run_phase2(state, sweep)


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=SWEEP_ORDER,
                    help='Run just one sweep instead of all four.')
    ap.add_argument('--target', type=int,
                    help='Run a single build at this target_fps for the --only sweep. '
                         'Use for dry-run or back-fill.')
    ap.add_argument('--phase', choices=['phase1', 'phase2', 'dryrun'],
                    default='phase1',
                    help='When using --target, label the phase in CSV/state. '
                         'Default phase1.')
    args = ap.parse_args()

    init_csv()
    state = load_state()
    save_state(state)

    driver_log(f'sweep_driver start (only={args.only}, target={args.target}, phase={args.phase})')

    if args.target is not None:
        if not args.only:
            print('error: --target requires --only', file=sys.stderr)
            return 2
        run_one_build(state, args.only, args.target, phase=args.phase)
        driver_log('sweep_driver end (single-target run)')
        return 0

    sweeps_to_run = [args.only] if args.only else SWEEP_ORDER
    for sweep in sweeps_to_run:
        run_sweep(state, sweep)

    driver_log('sweep_driver end (all sweeps complete)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
