#!/usr/bin/env python3
"""run_cnn_int8_qi_sweep.py — 10 CNN INT8 QuantIdentity synth/impl builds at new
target_fps points, captured per-build as primary records.

Each build is run via finn/capture_build.py (the exact path the Jun-1 recompiles
used) in an ISOLATED FINN scratch dir (--finn-build-dir /tmp/finn_qi_int8_sweep) so
it cannot collide with a concurrent FINN build using the default /tmp/finn_dev_samu.

Per-build capture is IMMEDIATE — before the next build starts, this writes into
results/finn/cnn_int8_qi_sweep/<label>/:
    impl_runme.log                      (Vivado place_design log; bust DRC verdict)
    top_wrapper_utilization_placed.rpt  (FIT only; LUT/FF/BRAM/DSP/CARRY8 actuals)
    top_wrapper_clock_utilization_routed.rpt  (FIT only)
    report/<all FINN report JSONs>      (estimate_*, post_synth_resources.json, ...)
    capture_console.log                 (capture_build.py stdout)
    verdict.txt                         (FIT/BUST, %s, binding resource, elapsed)
so a mid-batch death loses nothing already built. The FINN report/ JSONs are
copied OUT of the gitignored finn/size_sweep_runs/ into this committed path.

Build order: tiny first (cheapest, highest value).
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINN_DIR = REPO / 'finn'
CAPTURE = FINN_DIR / 'capture_build.py'
BUILD_DIR = '/tmp/finn_qi_int8_sweep'                 # isolated scratch (not _samu)
RESULTS_ROOT = REPO / 'results' / 'finn' / 'cnn_int8_qi_sweep'
DRIVER_LOG = RESULTS_ROOT / 'driver.log'
PROGRESS = RESULTS_ROOT / 'progress.txt'
SUMMARY = RESULTS_ROOT / 'summary.md'

# (onnx, target_fps, label) in build order. INT8 QI = cnn_mnist_<size>_qi.onnx.
BUILDS = [
    ('cnn_mnist_tiny_qi.onnx',   200,  'cnn_int8_tiny_qi_fps200'),
    ('cnn_mnist_tiny_qi.onnx',   500,  'cnn_int8_tiny_qi_fps500'),
    ('cnn_mnist_tiny_qi.onnx',   3000, 'cnn_int8_tiny_qi_fps3000'),
    ('cnn_mnist_tiny_qi.onnx',   5000, 'cnn_int8_tiny_qi_fps5000'),
    ('cnn_mnist_small_qi.onnx',  200,  'cnn_int8_small_qi_fps200'),
    ('cnn_mnist_small_qi.onnx',  500,  'cnn_int8_small_qi_fps500'),
    ('cnn_mnist_small_qi.onnx',  3000, 'cnn_int8_small_qi_fps3000'),
    ('cnn_mnist_medium_qi.onnx', 500,  'cnn_int8_medium_qi_fps500'),
    ('cnn_mnist_deep_3_qi.onnx', 500,  'cnn_int8_deep_3_qi_fps500'),
    ('cnn_mnist_deep_3_qi.onnx', 1000, 'cnn_int8_deep_3_qi_fps1000'),
]

# FIT utilization report: map Vivado "Site Type" row -> headline column.
FIT_ROWS = {'CLB LUTs': 'LUT', 'CLB Registers': 'FF', 'CARRY8': 'CARRY8',
            'Block RAM Tile': 'BRAM', 'DSPs': 'DSP'}
# BUST DRC over-utilization line.
DRC_RE = re.compile(
    r'Resource utilization:\s*(.+?)\s+over-utilized.*?requires\s+([\d,]+)\s+of\s+'
    r'such.*?only\s+([\d,]+)\s+compatible', re.IGNORECASE)


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with open(DRIVER_LOG, 'a') as f:
        f.write(line + '\n')


def clean_scratch():
    """Drop transient build dirs from our ISOLATED scratch (runme.log already
    copied out). Keeps vivado_ip_cache to speed identical-IP regen. Safe: no other
    build writes to BUILD_DIR."""
    for pat in ('code_gen_ipgen_*', 'vivado_stitch_proj_*', 'vivado_zynq_proj_*',
                'synth_out_of_context_*', 'vitis_floorplan_*'):
        for p in glob.glob(os.path.join(BUILD_DIR, pat)):
            shutil.rmtree(p, ignore_errors=True)


def strip_prefix(d, label):
    """capture_build.py saves <label>_FILE; rename to FILE to match the per-build
    layout (impl_runme.log, top_wrapper_*.rpt)."""
    pre = f'{label}_'
    for p in list(d.iterdir()):
        if p.is_file() and p.name.startswith(pre):
            p.replace(d / p.name[len(pre):])


def parse_console(console_path):
    """Pull elapsed_s / last_step / capture-verdict from capture_build.py stdout."""
    t = console_path.read_text(errors='replace') if console_path.exists() else ''
    el = re.search(r'elapsed_s\s*:\s*(\d+)', t)
    ls = re.search(r'^last_step\s*:\s*(\S+)', t, re.M) or re.search(r'last_step=(\S+)', t)
    vd = re.search(r'^verdict\s*:\s*(FIT|BUST)', t, re.M)
    return (int(el.group(1)) if el else None,
            ls.group(1) if ls else None,
            vd.group(1) if vd else None)


def parse_fit_util(rpt_path):
    """LUT/FF/CARRY8/BRAM/DSP {used, avail, pct} from a Vivado util_placed.rpt.
    Takes the Util% (last col) directly from the report — no device totals hardcoded."""
    out = {}
    for line in rpt_path.read_text(errors='replace').splitlines():
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 6:
            continue
        key = FIT_ROWS.get(cells[0])
        if key and key not in out:
            try:
                out[key] = {'used': float(cells[1]), 'avail': float(cells[-2]),
                            'pct': float(cells[-1])}
            except ValueError:
                continue
    return out


def parse_bust(runme_path):
    """{resource: {required, available, pct}} for every DRC over-utilized cell type."""
    seen = {}
    for m in DRC_RE.finditer(runme_path.read_text(errors='replace')):
        x = int(m.group(2).replace(',', ''))
        y = int(m.group(3).replace(',', ''))
        seen[m.group(1).strip()] = {'required': x, 'available': y,
                                    'pct': 100.0 * x / y if y else 0.0}
    return seen


def bust_headline(seen):
    """Best-effort LUT/BRAM/DSP/CARRY8 % from the DRC resources (raw lines kept too)."""
    cols = {'LUT': None, 'BRAM': None, 'DSP': None, 'CARRY8': None}
    for res, d in seen.items():
        r = res.lower()
        if 'carry8' in r:
            cols['CARRY8'] = d['pct']
        elif 'slice lut' in r or 'lut as logic' in r:
            cols['LUT'] = max(cols['LUT'] or 0.0, d['pct'])
        elif 'block ram' in r or 'bram' in r:
            cols['BRAM'] = d['pct']
        elif r.startswith('dsp'):
            cols['DSP'] = d['pct']
    return cols


def write_verdict(perbuild, onnx, fps, label, elapsed, last_step):
    """Inspect captured artifacts, classify FIT/BUST/INCOMPLETE, write verdict.txt.
    Returns a summary dict for the final table."""
    runme = perbuild / 'impl_runme.log'
    placed = perbuild / 'top_wrapper_utilization_placed.rpt'
    s = {'label': label, 'onnx': onnx, 'fps': fps, 'elapsed_s': elapsed,
         'last_step': last_step, 'verdict': 'INCOMPLETE',
         'LUT': None, 'BRAM': None, 'DSP': None, 'CARRY8': None, 'binding': None}
    lines = [f'label: {label}', f'onnx: {onnx}', f'target_fps: {fps}',
             f'elapsed_s: {elapsed if elapsed is not None else "unknown"}',
             f'last_finn_step: {last_step or "unknown"}']

    if placed.exists():                                   # placement succeeded -> FIT
        s['verdict'] = 'FIT'
        u = parse_fit_util(placed)
        for k in ('LUT', 'BRAM', 'DSP', 'CARRY8'):
            s[k] = round(u[k]['pct'], 2) if k in u else None
        s['binding'] = max((k for k in u), key=lambda k: u[k]['pct'], default=None)
        lines.append('verdict: FIT')
        for k in ('LUT', 'FF', 'CARRY8', 'BRAM', 'DSP'):
            if k in u:
                lines.append(f'{k}%: {u[k]["pct"]:.2f}  (used {u[k]["used"]:g} of {u[k]["avail"]:g})')
        lines.append(f'max_utilized_resource: {s["binding"]}')
    elif runme.exists() and parse_bust(runme):            # DRC over-util -> BUST
        s['verdict'] = 'BUST'
        seen = parse_bust(runme)
        cols = bust_headline(seen)
        for k in ('LUT', 'BRAM', 'DSP', 'CARRY8'):
            s[k] = round(cols[k], 2) if cols[k] is not None else None
        binding = max(seen, key=lambda r: seen[r]['pct'])
        s['binding'] = f'{binding} ({seen[binding]["pct"]:.1f}%)'
        lines.append('verdict: BUST')
        lines.append(f'binding_resource: {binding}')
        lines.append(f'binding_pct: {seen[binding]["pct"]:.1f}%  '
                     f'(requires {seen[binding]["required"]:,} of {seen[binding]["available"]:,})')
        for k in ('LUT', 'BRAM', 'DSP', 'CARRY8'):
            lines.append(f'{k}%: {cols[k]:.1f}' if cols[k] is not None
                         else f'{k}%: n/a (not over-utilized or no placed.rpt)')
        lines.append('all_over_utilized_resources:')
        for res, d in sorted(seen.items(), key=lambda kv: -kv[1]['pct']):
            lines.append(f'  {res}: {d["pct"]:.1f}%  (requires {d["required"]:,} of {d["available"]:,})')
    else:                                                 # never reached place/DRC
        lines.append('verdict: INCOMPLETE (no placed.rpt and no DRC over-util line; '
                     'see capture_console.log / impl_runme.log)')

    have = sorted(p.name for p in perbuild.iterdir() if p.name != 'verdict.txt')
    lines.append('captured_files: ' + ', '.join(have))
    (perbuild / 'verdict.txt').write_text('\n'.join(lines) + '\n')
    return s


def run_one(onnx, fps, label):
    perbuild = RESULTS_ROOT / label
    perbuild.mkdir(parents=True, exist_ok=True)
    console = perbuild / 'capture_console.log'
    out_subdir = f'size_sweep_runs/{label}'
    cmd = [sys.executable, str(CAPTURE), '--onnx', onnx, '--fps', str(fps),
           '--out-subdir', out_subdir,
           '--results-subdir', f'results/finn/cnn_int8_qi_sweep/{label}',
           '--label', label, '--finn-build-dir', BUILD_DIR]
    log(f'BUILD START {label}  onnx={onnx} fps={fps}')
    t0 = time.time()
    with open(console, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    log(f'capture_build.py rc={rc} wall={wall:.0f}s for {label}')

    strip_prefix(perbuild, label)                          # <label>_FILE -> FILE
    src_report = FINN_DIR / out_subdir / 'report'          # copy FINN JSONs out of
    if src_report.is_dir():                                # the gitignored build dir
        shutil.copytree(src_report, perbuild / 'report', dirs_exist_ok=True)
    src_buildlog = FINN_DIR / out_subdir / 'build.log'     # FINN flow log (also only
    if src_buildlog.is_file():                             # in the gitignored dir)
        shutil.copy2(src_buildlog, perbuild / 'finn_build.log')

    elapsed, last_step, _ = parse_console(console)
    if elapsed is None:
        elapsed = int(wall)
    s = write_verdict(perbuild, onnx, fps, label, elapsed, last_step)
    s['capture_rc'] = rc                                   # 4 = capture+verify FAILED

    if rc == 0:                                            # runme.log safely copied
        clean_scratch()
    else:
        log(f'WARN capture rc={rc} for {label}: leaving scratch for inspection')

    bind = s['binding'] or '-'
    def pc(v):
        return f'{v:.1f}' if isinstance(v, (int, float)) else 'n/a'
    log(f'BUILD DONE  {label}  {s["verdict"]}  '
        f'LUT={pc(s["LUT"])} BRAM={pc(s["BRAM"])} DSP={pc(s["DSP"])} '
        f'CARRY8={pc(s["CARRY8"])}  binding={bind}  elapsed={elapsed}s')
    with open(PROGRESS, 'a') as f:
        f.write(f'{label}\tfps={fps}\t{s["verdict"]}\tLUT={pc(s["LUT"])}\t'
                f'BRAM={pc(s["BRAM"])}\tDSP={pc(s["DSP"])}\tCARRY8={pc(s["CARRY8"])}\t'
                f'binding={bind}\telapsed={elapsed}s\n')
    return s


def write_summary(rows):
    def pc(v):
        return f'{v:.1f}' if isinstance(v, (int, float)) else 'n/a'
    out = ['# CNN INT8 QI target_fps sweep — synth/impl resource + fit verdict',
           '',
           f'Generated {now()}. Isolated scratch `{BUILD_DIR}`. Per-build primary '
           'records (runme.log, utilization rpt, FINN report JSONs) in '
           '`results/finn/cnn_int8_qi_sweep/<label>/`.',
           '',
           '| model | fps | verdict | LUT% | BRAM% | DSP% | CARRY8% | binding | elapsed |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        model = r['onnx'].replace('cnn_mnist_', '').replace('_qi.onnx', '')
        out.append(f'| {model} | {r["fps"]} | {r["verdict"]} | {pc(r["LUT"])} | '
                   f'{pc(r["BRAM"])} | {pc(r["DSP"])} | {pc(r["CARRY8"])} | '
                   f'{r["binding"] or "-"} | {r["elapsed_s"]}s |')
    SUMMARY.write_text('\n'.join(out) + '\n')
    log(f'wrote {SUMMARY}')


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    clean_scratch()
    log(f'=== SWEEP START: {len(BUILDS)} builds, scratch={BUILD_DIR} ===')
    rows = []
    for onnx, fps, label in BUILDS:
        try:
            rows.append(run_one(onnx, fps, label))
        except Exception as e:                             # never let one build kill the batch
            log(f'ERROR build {label} raised {type(e).__name__}: {e}')
            rows.append({'label': label, 'onnx': onnx, 'fps': fps, 'elapsed_s': '?',
                         'verdict': 'DRIVER_ERROR', 'LUT': None, 'BRAM': None,
                         'DSP': None, 'CARRY8': None, 'binding': str(e)[:60]})
        write_summary(rows)                                # rewrite after every build
        if rows[-1].get('capture_rc') == 4:                # capture+verify integrity failure
            log(f'HALT: capture VERIFICATION FAILED for {label} (rc=4). Sweep '
                f'stopped so a missing/empty artifact is not silently masked. '
                f'Inspect results/finn/cnn_int8_qi_sweep/{label}/ then re-run.')
            SUMMARY.with_name('SWEEP_HALTED').write_text(now() + f'  verify-fail: {label}\n')
            sys.exit(2)
    log('=== SWEEP COMPLETE ===')
    SUMMARY.with_name('SWEEP_DONE').write_text(now() + '\n')


if __name__ == '__main__':
    main()
