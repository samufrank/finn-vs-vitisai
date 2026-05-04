#!/usr/bin/env python3
"""Render accuracy_table.md from training_summary.csv + the four existing
tiny baselines (which weren't retrained as part of Phase A).

Usage:
  python3 render_accuracy_table.py
"""
import csv
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CSV_PATH = THIS_DIR / 'training_summary.csv'
OUT_PATH = THIS_DIR / 'accuracy_table.md'

# Tiny baselines were not retrained; values are from project history (the
# checkpoints used by the target_fps sweep + earlier work).
TINY_BASELINES = {
    ('mlp', 'tiny', 8): {'best_val_acc': 96.58, 'best_val_epoch': None,
                          'n_params': 52652,
                          'source': 'STATUS.md baseline (existing checkpoint)'},
    ('mlp', 'tiny', 4): {'best_val_acc': 97.29, 'best_val_epoch': None,
                          'n_params': 52652,
                          'source': 'STATUS_18.md (Alex MLP INT4 deploy)'},
    ('cnn', 'tiny', 8): {'best_val_acc': 91.99, 'best_val_epoch': None,
                          'n_params': 1444,
                          'source': 'STATUS.md baseline (existing checkpoint)'},
    ('cnn', 'tiny', 4): {'best_val_acc': 88.27, 'best_val_epoch': 25,
                          'n_params': 1444,
                          'source': 'session 22 Gate 3 warm-start'},
}

# Display order — matches the model definitions in models/{mlp,cnn}.py.
SIZE_ORDER_MLP = ['tiny', 'tiny_plus', 'small', 'small_plus',
                  'medium', 'large', 'original']
SIZE_ORDER_CNN = ['tiny', 'small', 'medium', 'deep_3', 'large']

# Channel/hidden config strings (for display; matches the model definition).
CONFIG_DISPLAY = {
    'mlp': {
        'tiny':       '[64,32]',
        'tiny_plus':  '[96,48]',
        'small':      '[128,64]',
        'small_plus': '[192,96]',
        'medium':     '[256,128]',
        'large':      '[512,256]',
        'original':   '[256,256,128]',
    },
    'cnn': {
        'tiny':   '[8,16]',
        'small':  '[16,32]',
        'medium': '[32,64]',
        'deep_3': '[16,32,64]',
        'large':  '[32,64,128]',
    },
}


def load_phase_a():
    """Return dict {(model, size, precision): row}."""
    rows = {}
    with open(CSV_PATH, newline='') as f:
        for r in csv.DictReader(f):
            if r['status'] != 'success':
                continue
            key = (r['model'], r['size'], int(r['precision']))
            rows[key] = {
                'best_val_acc':   float(r['best_val_acc']) if r['best_val_acc'] else None,
                'best_val_epoch': int(r['best_val_epoch']) if r['best_val_epoch'] else None,
                'n_params':       int(r['n_params']) if r['n_params'] else None,
                'elapsed_s':      float(r['elapsed_s']) if r['elapsed_s'] else None,
                'source':         'Phase A training (this experiment)',
            }
    return rows


def fmt_acc(v):
    return f'{v:.2f}' if v is not None else '—'


def fmt_epoch(v):
    return str(v) if v is not None else '—'


def fmt_int(v):
    return f'{v:,}' if v is not None else '—'


def render_model_table(model, sizes, results):
    lines = [f'## {model.upper()} — capacity-precision (MNIST)', '']
    lines.append('| Size | Channels/Hidden | Params | INT8 best-val | INT8 ep | INT4 best-val | INT4 ep | Δ (INT4−INT8) | Notes |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---|')
    for size in sizes:
        cfg = CONFIG_DISPLAY[model].get(size, '?')
        r8 = results.get((model, size, 8))
        r4 = results.get((model, size, 4))
        a8 = r8['best_val_acc'] if r8 else None
        a4 = r4['best_val_acc'] if r4 else None
        e8 = r8['best_val_epoch'] if r8 else None
        e4 = r4['best_val_epoch'] if r4 else None
        params = (r8 or r4 or {}).get('n_params')
        delta = a4 - a8 if (a4 is not None and a8 is not None) else None
        delta_str = f'{delta:+.2f}' if delta is not None else '—'
        notes = ''
        if size == 'tiny':
            notes = 'tiny existing baselines (not retrained)'
        elif model == 'mlp' and size == 'original':
            notes = '3-hidden-layer MLP'
        elif model == 'cnn' and size in ('deep_3', 'large'):
            notes = '3-conv-layer'
        lines.append(
            f'| {size} | {cfg} | {fmt_int(params)} | '
            f'{fmt_acc(a8)} | {fmt_epoch(e8)} | '
            f'{fmt_acc(a4)} | {fmt_epoch(e4)} | '
            f'{delta_str} | {notes} |'
        )
    lines.append('')
    return '\n'.join(lines)


def main():
    results = load_phase_a()
    results.update(TINY_BASELINES)

    out = ['# Phase A — accuracy table (size ablation, MNIST)', '']
    out.append('Brevitas best-val accuracy per (model, size, precision). '
               'Validation set = MNIST test set (10 000 images).')
    out.append('')
    out.append('**Training recipes:**')
    out.append('- INT8: lr=1e-3, 10 epochs, batch=64, no warm-start '
               '(cold-init Brevitas QAT). Existing `train_and_export.py` defaults.')
    out.append('- INT4: lr=1e-4, 50 epochs, batch=256, '
               '`--init-from <model>_mnist_<size>.pth`, `--grad-clip 1.0`. '
               'Session 22 warm-start methodology.')
    out.append('- Tiny baselines not retrained — values from existing '
               'checkpoints used in target_fps sweep / prior work. Sources '
               'noted in the per-row Notes column.')
    out.append('')
    out.append('**Caveat:** the INT8 and INT4 training recipes differ in '
               'epoch count (10 vs 50), batch size (64 vs 256), and learning '
               'rate (1e-3 vs 1e-4). INT4 also warm-starts from the matched-'
               'size INT8 checkpoint. Some of the consistently-positive '
               'INT4−INT8 delta is recipe-driven, not precision-driven. The '
               '*sign* of the gap (INT4 ≥ INT8 at all non-tiny sizes) is the '
               'trustworthy signal; the *magnitude* is inflated by recipe.')
    out.append('')
    out.append(render_model_table('mlp', SIZE_ORDER_MLP, results))
    out.append(render_model_table('cnn', SIZE_ORDER_CNN, results))

    out.append('## Headline finding — CNN capacity-precision claim')
    out.append('')
    out.append('Session 22 established that CNN tiny [8,16] INT4 plateaus at '
               '~88% Brevitas best-val (vs INT8 92%) due to insufficient '
               'parameter capacity to absorb sub-byte quantization noise. '
               'This experiment tests whether the gap closes at modestly-'
               'larger CNN topologies.')
    out.append('')
    out.append('Result: **the deficit collapses with one size step.** '
               'CNN small [16,32] (5.2k params, 3.6× tiny) brings INT4 to '
               'parity with INT8 (Δ +0.08pt). CNN medium [32,64] (19.6k '
               'params) has INT4 ahead of INT8 by +0.95pt. The capacity-'
               'precision claim is validated by the curve shape '
               '(monotonic improvement of INT4 relative to INT8 as size '
               'grows), not just by a single point comparison.')
    out.append('')
    out.append('MLP shows no equivalent crossover — at tiny [64,32] the MLP '
               'already has 52k params and INT4 starts ahead of INT8 '
               '(+0.71pt). MLP MNIST is too easy at INT4 for tiny to be a '
               'capacity bottleneck.')
    out.append('')

    OUT_PATH.write_text('\n'.join(out))
    print(f'wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
