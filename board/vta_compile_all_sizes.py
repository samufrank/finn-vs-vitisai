#!/usr/bin/env python3
"""VTA export compile sweep driver.

Iterates (model, size, precision) tuples, runs extract_brevitas_weights.py
where applicable, then the matching export_vta_*.py per (model, precision):
  MLP INT8 / INT4: export_vta_model.py (consumes extracted W{i}.npy +
                   architecture string)
  CNN INT8       : export_vta_cnn.py (loads .pth directly, BN-fold + INT8
                   quant inline)
  CNN INT4       : export_vta_cnn_int4_o8.py (loads .pth directly, INT4
                   weight quant + zero-point Mode G pipeline)

Captures stdout/stderr per job and emits a JSON results table. The compile
step needs the TVM env vars (PYTHONPATH/TVM_HOME) — driver sets them via
subprocess env. Each job is independent; failures are logged and the sweep
continues.

Calls `vta/configs/switch_vta_config.sh` automatically per-precision before
each compile (int8 for precision=8, int4_o8 for precision=4). Idempotent —
no-op when the active config already matches. This avoids a class of bugs
where modules get compiled against an int4_o8 active config but deployed
against an int8 bitstream (or vice versa), producing instructions the
runtime silently misinterprets.

Run from anywhere; paths are derived from __file__.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO          = Path(__file__).resolve().parent.parent          # finn-vs-vitisai/
TVM_DIR       = REPO.parent / 'tvm-v0.12.0'
EXTRACT       = REPO / 'board' / 'extract_brevitas_weights.py'
EXPORT_MLP    = REPO / 'board' / 'export_vta_model.py'
EXPORT_CNN    = REPO / 'board' / 'export_vta_cnn.py'
EXPORT_CNN_I4 = REPO / 'board' / 'export_vta_cnn_int4_o8.py'
SWITCH_VTA    = REPO / 'vta' / 'configs' / 'switch_vta_config.sh'
VTA_CONFIG    = TVM_DIR / '3rdparty' / 'vta-hw' / 'config' / 'vta_config.json'
VTA_EXPORTS   = REPO / 'vta_exports'
LOGS          = REPO / 'logs' / 'vta_sweep'
RESULTS_PATH  = LOGS / 'results.json'

# (precision, model_kind) → vta/configs/<name>/ to make active before compile.
# The active config drives env.INP_WIDTH / WGT_WIDTH / OUT_WIDTH /
# LOG_*_BUFF_SIZE — modules built against the wrong config produce instructions
# the deployed bitstream silently misinterprets (correct-looking but wrong
# values). Caught the hard way in a debug session: 14% accuracy on small INT8
# CNN compiled while the active config was int4_o8.
#
# CNN INT4-perchan is Mode G (zp-offset INT4 input, INT8 output) → int4_o8 config.
# MLP INT4 historically used a plain INT4 (output also INT4) config, but no
# 'int4' subdir exists in vta/configs/ — keep it mapped to int4_o8 for now;
# if MLP INT4 results look wrong, that's the smell. (Out of scope for the
# current size-sweep deploy, which is CNN-only at INT4.)
PRECISION_TO_CONFIG = {
    8: 'int8',
    4: 'int4_o8',
}

# Hidden-layer architectures from models/mlp.py:get_mlp_config
MLP_ARCH = {
    'tiny_plus':  '784,96,48,10',
    'small':      '784,128,64,10',
    'small_plus': '784,192,96,10',
    'medium':     '784,256,128,10',
    'large':      '784,512,256,10',
    'original':   '784,256,256,128,10',
}
MLP_SIZES = list(MLP_ARCH.keys())
CNN_SIZES = ['small', 'medium', 'deep_3', 'large']

JOBS = []
for s in MLP_SIZES:
    for p in (8, 4):
        JOBS.append(('mlp', s, p))
for s in CNN_SIZES:
    for p in (8, 4):
        JOBS.append(('cnn', s, p))


def ckpt_path(model, size, precision):
    suffix = '_int4' if precision == 4 else ''
    return REPO / 'finn' / f'{model}_mnist_{size}{suffix}.pth'


def export_dir(model, size, precision):
    """Output directory naming. CNN INT4 gets the _perchan suffix to flag
    that it's the Mode G (per-channel weights, zp=8) pipeline — distinct
    from the (defunct, never produced here) plain INT4 path."""
    if model == 'cnn' and precision == 4:
        return VTA_EXPORTS / f'cnn_{size}_int4_perchan'
    return VTA_EXPORTS / f'{model}_{size}_int{precision}'


def vta_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = f'{TVM_DIR}/python:{TVM_DIR}/vta/python'
    env['TVM_HOME']   = str(TVM_DIR)
    return env


_active_config = None


def _detect_active_config():
    """Read vta_config.json and infer the saved-config name (int8 / int4_o8)."""
    if not VTA_CONFIG.exists():
        return None
    try:
        c = json.load(open(VTA_CONFIG))
    except Exception:
        return None
    inp = c.get('LOG_INP_WIDTH', -1)
    out = c.get('LOG_OUT_WIDTH', inp)   # mirrors switch_vta_config.sh:current_mode
    if (inp, out) == (3, 3): return 'int8'
    if (inp, out) == (2, 3): return 'int4_o8'
    return None


def ensure_config(name):
    """Switch the active TVM/VTA config if it's not already at `name`.

    `name` matches a subdir of vta/configs/ (currently 'int8' or 'int4_o8').
    Idempotent — no-op if already active. Aborts the sweep on switch failure
    so we don't compile the wrong thing silently.
    """
    global _active_config
    if _active_config is None:
        _active_config = _detect_active_config()
    if _active_config == name:
        return
    print(f'  [config] switching active VTA config: {_active_config} → {name}')
    if not SWITCH_VTA.exists():
        raise RuntimeError(f'switch script not found: {SWITCH_VTA}')
    res = subprocess.run([str(SWITCH_VTA), name], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f'switch_vta_config.sh {name!r} failed (rc={res.returncode}):\n'
            f'{res.stdout}\n{res.stderr}')
    _active_config = name


def run(cmd, log_path, timeout, env=None, cwd=None):
    """Run cmd, tee to log_path. Returns (rc, last_lines, elapsed_sec)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, env=env, cwd=cwd,
            capture_output=True, text=True, timeout=timeout)
        out = (result.stdout or '') + (result.stderr or '')
        rc = result.returncode
    except subprocess.TimeoutExpired as te:
        out = (te.stdout or '') + (te.stderr or '') + f'\n[TIMEOUT after {timeout}s]\n'
        rc = -1
    log_path.write_text(out)
    elapsed = time.monotonic() - t0
    last = '\n'.join([ln for ln in out.strip().splitlines() if ln][-12:])
    return rc, last, elapsed


def extract_one(model, size, precision):
    # CNN INT4 doesn't need a separate extract step — the Mode G export
    # script does its own inline Brevitas walk + BN fold + per-channel
    # quantization. Skip cleanly.
    if model == 'cnn' and precision == 4:
        return {'rc': 0, 'err': '', 'elapsed': 0.0,
                'note': 'skipped — CNN INT4 extracts inline in the export script'}
    out_dir = export_dir(model, size, precision)
    log = LOGS / f'extract_{model}_{size}_int{precision}.log'
    cp = ckpt_path(model, size, precision)
    if not cp.exists():
        return {'rc': -2, 'err': f'checkpoint not found: {cp}', 'elapsed': 0.0}
    cmd = [
        sys.executable, str(EXTRACT),
        '--model', model, '--size', size, '--precision', str(precision),
        '--checkpoint', str(cp),
        '--output-dir', str(out_dir),
    ]
    rc, last, elapsed = run(cmd, log, timeout=180)
    return {'rc': rc, 'err': last if rc != 0 else '', 'elapsed': elapsed,
            'log': str(log)}


def compile_mlp(size, precision):
    out_dir = export_dir('mlp', size, precision)
    log = LOGS / f'compile_mlp_{size}_int{precision}.log'
    if not (out_dir / 'meta.json').exists():
        return {'rc': -2, 'err': f'no extracted weights at {out_dir}',
                'elapsed': 0.0}
    try:
        ensure_config(PRECISION_TO_CONFIG[precision])
    except Exception as e:
        return {'rc': -3, 'err': f'config switch failed: {e}', 'elapsed': 0.0}
    cmd = [
        sys.executable, str(EXPORT_MLP),
        '--weights-dir', str(out_dir),
        '--output-dir',  str(out_dir),
        '--architecture', MLP_ARCH[size],
        '--cal-samples', '100',
    ]
    rc, last, elapsed = run(cmd, log, timeout=900,
                            env=vta_env(), cwd=str(TVM_DIR))
    return {'rc': rc, 'err': last if rc != 0 else '', 'elapsed': elapsed,
            'log': str(log)}


def compile_cnn(size, precision):
    out_dir = export_dir('cnn', size, precision)
    log = LOGS / f'compile_cnn_{size}_int{precision}.log'
    cp = ckpt_path('cnn', size, precision)
    if not cp.exists():
        return {'rc': -2, 'err': f'checkpoint not found: {cp}', 'elapsed': 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ensure_config(PRECISION_TO_CONFIG[precision])
    except Exception as e:
        return {'rc': -3, 'err': f'config switch failed: {e}', 'elapsed': 0.0}
    if precision == 8:
        cmd = [
            sys.executable, str(EXPORT_CNN),
            '--checkpoint', str(cp),
            '--size', size,
            '--output-dir', str(out_dir),
            '--cal-samples', '100',
            '--verify-samples', '0',
        ]
    else:
        # INT4 routes through the Mode G (per-channel + zp=8) script. It
        # does its own inline weight extraction + BN fold; no upstream
        # extract step is needed (and extract_brevitas_weights.py refuses
        # CNN INT4 anyway).
        cmd = [
            sys.executable, str(EXPORT_CNN_I4),
            '--checkpoint', str(cp),
            '--size', size,
            '--output-dir', str(out_dir),
        ]
    rc, last, elapsed = run(cmd, log, timeout=1500,
                            env=vta_env(), cwd=str(TVM_DIR))
    return {'rc': rc, 'err': last if rc != 0 else '', 'elapsed': elapsed,
            'log': str(log)}


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    results = {}

    print('=' * 72)
    print('Phase 1: extractions')
    print('=' * 72)
    for model, size, prec in JOBS:
        key = f'{model}_{size}_int{prec}'
        print(f'  [extract] {key} ... ', end='', flush=True)
        r = extract_one(model, size, prec)
        results[key] = {'extract': r}
        status = 'OK' if r['rc'] == 0 else f'FAIL rc={r["rc"]}'
        print(f'{status} ({r["elapsed"]:.1f}s)')
        if r['rc'] != 0 and r['err']:
            for ln in r['err'].splitlines()[-3:]:
                print(f'      {ln}')

    print()
    print('=' * 72)
    print('Phase 2: VTA compiles')
    print('=' * 72)
    for model, size, prec in JOBS:
        key = f'{model}_{size}_int{prec}'
        print(f'  [compile] {key} ... ', end='', flush=True)
        if model == 'mlp':
            r = compile_mlp(size, prec)
        else:
            r = compile_cnn(size, prec)
        results[key]['compile'] = r
        status = 'OK' if r['rc'] == 0 else f'FAIL rc={r["rc"]}'
        print(f'{status} ({r["elapsed"]:.1f}s)')
        if r['rc'] != 0 and r['err']:
            for ln in r['err'].splitlines()[-3:]:
                print(f'      {ln}')

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f'\nResults written: {RESULTS_PATH}')

    # Summary table
    print()
    print('=' * 72)
    print('Summary')
    print('=' * 72)
    print(f"{'job':28s} {'extract':10s} {'compile':10s} {'compile_t':>9s}")
    for model, size, prec in JOBS:
        key = f'{model}_{size}_int{prec}'
        r = results[key]
        ext = 'OK' if r['extract']['rc'] == 0 else f'FAIL'
        cmp_status = 'OK' if r['compile']['rc'] == 0 else f'FAIL'
        cmp_t = r['compile']['elapsed']
        print(f'{key:28s} {ext:10s} {cmp_status:10s} {cmp_t:>8.1f}s')


if __name__ == '__main__':
    sys.exit(main() or 0)
