#!/usr/bin/env python3
"""Extract resource numbers for the CNN INT8 QI target_fps sweep straight from the
PRIMARY Tier-1 artifacts — independent of README.md / summary.md (which are derived
views). Re-runnable anytime; regenerates resource_summary.csv.

Sources, per build dir under results/finn/cnn_int8_qi_sweep/<label>/:
  report/post_synth_resources.json '(top)'  -> native counts (LUT/FF/BRAM_36K/
                                               BRAM_18K/DSP)
  top_wrapper_utilization_placed.rpt        -> Vivado Util% (and CARRY8, absent from
                                               the JSON); device totals come from the
                                               rpt's own 'Available' column.
BRAM tile (RAMB36 basis, /216) = BRAM_36K + 0.5*BRAM_18K  (per
context/SOURCES_AND_VERIFICATION.md; native units kept, never silently converted).

This script is self-contained (no import of the sweep driver) so it keeps working
even if the build scripts move. Run from repo root: python3 analysis/extract_cnn_int8_qi_sweep.py
"""
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / 'results' / 'finn' / 'cnn_int8_qi_sweep'
OUT = SWEEP / 'resource_summary.csv'

# Build list in sweep order (label, size, fps). Mirrors the driver's BUILDS.
BUILDS = [
    ('cnn_int8_tiny_qi_fps200',   'tiny',   200),
    ('cnn_int8_tiny_qi_fps500',   'tiny',   500),
    ('cnn_int8_tiny_qi_fps3000',  'tiny',   3000),
    ('cnn_int8_tiny_qi_fps5000',  'tiny',   5000),
    ('cnn_int8_small_qi_fps200',  'small',  200),
    ('cnn_int8_small_qi_fps500',  'small',  500),
    ('cnn_int8_small_qi_fps3000', 'small',  3000),
    ('cnn_int8_medium_qi_fps500', 'medium', 500),
    ('cnn_int8_deep_3_qi_fps500', 'deep_3', 500),
    ('cnn_int8_deep_3_qi_fps1000','deep_3', 1000),
]

# Vivado utilization-report "Site Type" row -> our key.
RPT_ROWS = {'CLB LUTs': 'LUT', 'CLB Registers': 'FF', 'CARRY8': 'CARRY8',
            'Block RAM Tile': 'BRAM', 'DSPs': 'DSP'}


def rpt_util(rpt):
    """{key: (used, avail, pct)} from a placed/synth utilization rpt."""
    out = {}
    for line in rpt.read_text(errors='replace').splitlines():
        if not line.startswith('|'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) < 6:
            continue
        k = RPT_ROWS.get(c[0])
        if k and k not in out:
            try:
                out[k] = (float(c[1]), float(c[-2]), float(c[-1]))
            except ValueError:
                pass
    return out


def elapsed_of(console):
    if console.exists():
        m = re.search(r'elapsed_s\s*:\s*(\d+)', console.read_text(errors='replace'))
        if m:
            return int(m.group(1))
    return ''


COLS = ['build', 'size', 'fps', 'verdict', 'binding',
        'LUT', 'LUT_pct', 'FF', 'FF_pct', 'BRAM_36K', 'BRAM_18K', 'BRAM_tile',
        'BRAM_pct', 'DSP', 'DSP_pct', 'CARRY8', 'CARRY8_pct', 'elapsed_s']


def main():
    rows = []
    for label, size, fps in BUILDS:
        pb = SWEEP / label
        placed = pb / 'top_wrapper_utilization_placed.rpt'
        psr = pb / 'report' / 'post_synth_resources.json'
        r = {'build': label, 'size': size, 'fps': fps, 'verdict': 'INCOMPLETE'}

        u = rpt_util(placed) if placed.exists() else {}
        if u:
            r['verdict'] = 'FIT'
            r['binding'] = max(u, key=lambda k: u[k][2])
            for k in ('LUT', 'FF', 'BRAM', 'DSP', 'CARRY8'):
                r[f'{k}_pct'] = round(u[k][2], 2) if k in u else ''
            r['CARRY8'] = int(u['CARRY8'][0]) if 'CARRY8' in u else ''

        if psr.exists():
            t = json.load(open(psr))['(top)']
            r['LUT'], r['FF'], r['DSP'] = t['LUT'], t['FF'], t['DSP']
            r['BRAM_36K'], r['BRAM_18K'] = t['BRAM_36K'], t['BRAM_18K']
            r['BRAM_tile'] = t['BRAM_36K'] + 0.5 * t['BRAM_18K']

        r['elapsed_s'] = elapsed_of(pb / 'capture_console.log')
        rows.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {OUT} ({len(rows)} rows)')
    for r in rows:
        print(f"  {r['build']:28} {r.get('verdict',''):5} LUT%={r.get('LUT_pct','')} "
              f"BRAM%={r.get('BRAM_pct','')} binding={r.get('binding','')}")


if __name__ == '__main__':
    main()
