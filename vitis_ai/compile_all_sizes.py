#!/usr/bin/env python3
"""compile_all_sizes.py — DPU compile sweep across (model, size) combos.

For each (model, size) pair, runs the Vitis AI flow inside the
xilinx/vitis-ai-pytorch-cpu Docker container:
  1. python train_and_quantize.py --model <m> --size <s>      (PTQ)
  2. vai_c_xir -x quantize_result/<MODEL>_int.xmodel \\
       -a arch_zu3_b512.json -o compiled/<m>_<s>/ -n <m>_mnist_<s>

Mirrors the docker invocation pattern from finn/sweep_driver.py:docker_cmd
but uses the Vitis AI image and conda env.

Output:
  vitis_ai/compiled/<m>_<s>/<m>_mnist_<s>.xmodel + meta.json + md5sum.txt
  vitis_ai/compile_summary.csv  (one row per build)
  vitis_ai/compile_logs/<m>_<s>.log

Existing tiny xmodel at vitis_ai/zu3_b512/compiled/ is preserved (different
output dir).

Models compiled: all 7 MLP sizes + all 5 CNN sizes = 12 builds. CNN medium
and large are included even though FINN couldn't fit them — DPU's overlay
should handle them ('overlay deploys what dataflow can't').

Usage:
  python3 vitis_ai/compile_all_sizes.py                 # full sweep
  python3 vitis_ai/compile_all_sizes.py --only mlp_tiny # one combo
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

REPO_ROOT     = Path(__file__).resolve().parent.parent           # finn-vs-vitisai/
VITIS_DIR     = REPO_ROOT / 'vitis_ai'
COMPILED_DIR  = VITIS_DIR / 'compiled'
LOGS_DIR      = VITIS_DIR / 'compile_logs'
SUMMARY_CSV   = VITIS_DIR / 'compile_summary.csv'

DOCKER_IMAGE = 'xilinx/vitis-ai-pytorch-cpu:latest'
ARCH_FILE_INSIDE_DOCKER = '/workspace/project/vitis_ai/arch_zu3_b512.json'

SIZES_MLP = ['tiny', 'tiny_plus', 'small', 'small_plus',
             'medium', 'large', 'original']
SIZES_CNN = ['tiny', 'small', 'medium', 'deep_3', 'large']

COMBOS = (
    [('mlp', s) for s in SIZES_MLP] +
    [('cnn', s) for s in SIZES_CNN]
)

CSV_HEADER = [
    'model', 'size', 'started_iso', 'elapsed_s',
    'status',                        # success | quant_fail | compile_fail | tool_fail
    'float_acc', 'quant_acc',
    'xmodel_path', 'xmodel_bytes',
    'log_path', 'error_excerpt',
]


# =============================================================================
# Helpers
# =============================================================================

def now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def driver_log(msg):
    line = f'[{now_iso()}] {msg}'
    print(line, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / 'driver.log', 'a') as f:
        f.write(line + '\n')


def init_csv():
    if SUMMARY_CSV.exists():
        return
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_csv(row):
    full = {k: row.get(k, '') for k in CSV_HEADER}
    with open(SUMMARY_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(full)


def existing_done():
    """Return set of (model, size) tuples already in the summary CSV."""
    if not SUMMARY_CSV.exists():
        return set()
    seen = set()
    with open(SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            if r.get('status') == 'success':
                seen.add((r['model'], r['size']))
    return seen


# =============================================================================
# Docker invocation
# =============================================================================

def docker_cmd(model, size, output_dir_in_docker):
    """Build docker run argv. Mirrors sweep_driver.docker_cmd structure."""
    repo = str(REPO_ROOT)
    # Class name in quantize_result/ is uppercase (MLP_int.xmodel, CNN_int.xmodel)
    cls_name = model.upper()
    # Inner shell command. Source conda, activate the vitis-ai-pytorch env,
    # cd to vitis_ai/, run train_and_quantize.py, then vai_c_xir.
    inner = (
        'source /opt/vitis_ai/conda/etc/profile.d/conda.sh && '
        'conda activate vitis-ai-pytorch && '
        'cd /workspace/project/vitis_ai && '
        'mkdir -p compile_logs && '
        'rm -rf quantize_result && '
        f'python train_and_quantize.py --model {model} --dataset mnist '
        f'--size {size} --epochs 10 --batch_size 1 && '
        f'mkdir -p {output_dir_in_docker} && '
        f'vai_c_xir -x quantize_result/{cls_name}_int.xmodel '
        f'-a {ARCH_FILE_INSIDE_DOCKER} '
        f'-o {output_dir_in_docker} '
        f'-n {model}_mnist_{size}'
    )
    return [
        'docker', 'run', '--rm', '--init',
        '-w', '/workspace/project/vitis_ai',
        '-v', f'{repo}:/workspace/project',
        '-v', '/etc/group:/etc/group:ro',
        '-v', '/etc/passwd:/etc/passwd:ro',
        '-v', '/etc/shadow:/etc/shadow:ro',
        '--user', '1000:1000',
        DOCKER_IMAGE,
        'bash', '-c', inner,
    ]


# =============================================================================
# Output classification
# =============================================================================

# train_and_quantize.py prints these.
RE_FLOAT_ACC = re.compile(r'Float accuracy:\s*([0-9.]+)%')
RE_QUANT_ACC = re.compile(r'Quantized accuracy:\s*([0-9.]+)%')


def classify(log_text, returncode, output_dir):
    """Return (status, float_acc, quant_acc, error_excerpt)."""
    fa = RE_FLOAT_ACC.search(log_text)
    qa = RE_QUANT_ACC.search(log_text)
    float_acc = float(fa.group(1)) if fa else None
    quant_acc = float(qa.group(1)) if qa else None

    xmodels = list(Path(output_dir).glob('*.xmodel')) if Path(output_dir).exists() else []

    # Quantization-stage failure (no xmodel emitted from quantize_result/).
    quant_failed = ('Exporting...' in log_text and
                    'Saved to:' not in log_text and
                    qa is None)

    if returncode == 0 and xmodels:
        return 'success', float_acc, quant_acc, ''

    excerpt = ''
    for needle in ('[VAIC ', 'ERROR:', 'RuntimeError', 'Traceback'):
        i = log_text.find(needle)
        if i >= 0:
            excerpt = log_text[i:i + 600]
            break

    if returncode != 0 and qa is not None and not xmodels:
        # vai_c_xir failed after a successful quantize.
        return 'compile_fail', float_acc, quant_acc, excerpt
    if returncode != 0 and qa is None:
        # Couldn't get past quantization.
        return 'quant_fail', float_acc, quant_acc, excerpt
    return 'tool_fail', float_acc, quant_acc, excerpt


# =============================================================================
# Per-build runner
# =============================================================================

def run_one(model, size):
    output_dir_host    = COMPILED_DIR / f'{model}_{size}'
    output_dir_docker  = f'/workspace/project/vitis_ai/compiled/{model}_{size}'
    log_path           = LOGS_DIR / f'{model}_{size}.log'

    if output_dir_host.exists():
        # Stale dir from a partial prior run. Remove (the user said new
        # outputs go to compiled/<m>_<s>/ which is its own tree).
        shutil.rmtree(output_dir_host)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    started_iso = now_iso()
    driver_log(f'BEGIN {model} {size}')

    cmd = docker_cmd(model, size, output_dir_docker)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=3600, text=True)
        rc  = result.returncode
        out = (result.stdout or '') + '\n' + (result.stderr or '')
    except subprocess.TimeoutExpired as e:
        rc  = -9
        out = (e.stdout or '') + '\n' + (e.stderr or '') + '\n[TIMEOUT 1h]'
    elapsed = time.time() - t0

    with open(log_path, 'w') as f:
        f.write(out)

    status, float_acc, quant_acc, excerpt = classify(
        out, rc, output_dir_host)

    xmodel_path = ''
    xmodel_bytes = ''
    if status == 'success':
        cands = list(output_dir_host.glob('*.xmodel'))
        if cands:
            xmodel_path  = str(cands[0].relative_to(REPO_ROOT))
            xmodel_bytes = cands[0].stat().st_size

    driver_log(
        f'END   {model} {size}  status={status}  elapsed={elapsed:.0f}s  '
        f'float={float_acc}%  quant={quant_acc}%  xmodel={xmodel_bytes}B')

    return {
        'model': model, 'size': size,
        'started_iso': started_iso,
        'elapsed_s': round(elapsed, 1),
        'status': status,
        'float_acc': float_acc if float_acc is not None else '',
        'quant_acc': quant_acc if quant_acc is not None else '',
        'xmodel_path': xmodel_path,
        'xmodel_bytes': xmodel_bytes,
        'log_path': str(log_path.relative_to(REPO_ROOT)),
        'error_excerpt': (excerpt or '')[:500],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    help='e.g. mlp_tiny — run just this combo')
    args = ap.parse_args()

    init_csv()
    seen = existing_done()

    combos = COMBOS
    if args.only:
        m, s = args.only.split('_', 1)
        combos = [(m, s)]

    driver_log(f'compile_all_sizes start ({len(combos)} combos, {len(seen)} already done)')
    for model, size in combos:
        if (model, size) in seen:
            driver_log(f'skip {model} {size}: already in summary')
            continue
        row = run_one(model, size)
        append_csv(row)
        seen.add((model, size))

    driver_log('compile_all_sizes end')
    return 0


if __name__ == '__main__':
    sys.exit(main())
