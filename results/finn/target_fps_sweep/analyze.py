#!/usr/bin/env python3
"""analyze.py — generate sweep_analysis.md from sweep CSV + per-build configs.

Reads:
  resource_summary.csv                                       (this dir)
  ../../../finn/target_fps_sweep_runs/<sweep>_fps<t>/build.log
  ../../../finn/target_fps_sweep_runs/<sweep>_fps<t>/final_hw_config.json
  /tmp/finn_dev_samu/vivado_zynq_proj_*/.../impl_1/runme.log (failure dives)

Writes:
  sweep_analysis.md

Structure of the markdown:
  - Per-sweep tables (one per sweep), one row per target_fps
  - Cross-sweep comparison tables, one per metric (LUT%, DSP%, est FPS, MVAU folding)
  - Partitioning shift analysis (per sweep, list any CPU↔FPGA layer movements)
  - Failure investigation section (each tool_fail/resource_fail/timing_fail
    gets a deeper read of build.log + Vivado runme.log)

No plots — structured data only.
"""

import csv
import json
import os
import re
from pathlib import Path

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent           # finn-vs-vitisai/
CSV_PATH  = THIS_DIR / 'resource_summary.csv'
RUNS_DIR  = REPO_ROOT / 'finn' / 'target_fps_sweep_runs'
OUT_PATH  = THIS_DIR / 'sweep_analysis.md'

SWEEP_ORDER = ['mlp_int8', 'mlp_int4', 'cnn_int8', 'cnn_int4']
ALL_TARGETS = [1000, 10000, 100000, 500000]


# =============================================================================
# Loading
# =============================================================================

def load_csv():
    """Return {sweep: [row, ...]} sorted ascending by target_fps."""
    rows_by_sweep = {s: [] for s in SWEEP_ORDER}
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            sweep = row['sweep']
            if sweep in rows_by_sweep:
                rows_by_sweep[sweep].append(row)
    for s in SWEEP_ORDER:
        rows_by_sweep[s].sort(key=lambda r: int(r['target_fps']))
    return rows_by_sweep


def load_final_hw_config(sweep, target):
    p = RUNS_DIR / f'{sweep}_fps{target}' / 'final_hw_config.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# =============================================================================
# Cell formatting
# =============================================================================

def fmt_count_pct(count_str, pct_str):
    if not count_str or not pct_str:
        return '—'
    return f'{int(float(count_str))} ({float(pct_str):.1f}%)'


def fmt_pct(pct_str):
    if not pct_str:
        return '—'
    return f'{float(pct_str):.1f}%'


def fmt_float(v, precision=2):
    if v == '' or v is None:
        return '—'
    try:
        return f'{float(v):.{precision}f}'
    except Exception:
        return str(v)


def fmt_int(v):
    if v == '' or v is None:
        return '—'
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)


def parse_folding(folding_json_str):
    if not folding_json_str:
        return {}
    try:
        return json.loads(folding_json_str)
    except Exception:
        return {}


def fmt_mvau(folding_json_str):
    f = parse_folding(folding_json_str)
    parts = []
    for k, v in f.items():
        if 'MVAU' not in k:
            continue
        pe = v.get('PE') if v.get('PE') is not None else '—'
        simd = v.get('SIMD') if v.get('SIMD') is not None else '—'
        parts.append(f'{k}=PE{pe}/SIMD{simd}')
    return ', '.join(parts) if parts else '—'


def parse_cpu_layers(cpu_layers_json_str):
    if not cpu_layers_json_str:
        return []
    try:
        return json.loads(cpu_layers_json_str)
    except Exception:
        return []


def fmt_cpu_count(cpu_layers_json_str):
    cl = parse_cpu_layers(cpu_layers_json_str)
    return str(len(cl)) if cl else '—'


def fmt_cpu_list(cpu_layers_json_str):
    cl = parse_cpu_layers(cpu_layers_json_str)
    if not cl:
        return '—'
    return ', '.join(name for op, name in cl)


# =============================================================================
# Per-sweep tables
# =============================================================================

def render_per_sweep_section(sweep, rows):
    lines = []
    lines.append(f'## {sweep}')
    lines.append('')
    if not rows:
        lines.append('No builds.')
        lines.append('')
        return '\n'.join(lines)

    n_success = sum(1 for r in rows if r['status'] == 'success')
    n_total = len(rows)
    statuses = ', '.join(f'{r["target_fps"]}={r["status"]}' for r in rows)
    lines.append(f'{n_success}/{n_total} successful. Sequence: {statuses}.')
    lines.append('')

    lines.append('| target_fps | status | LUT | BRAM18 | DSP | Fmax (MHz) | est FPS | MVAU PE/SIMD | CPU layers |')
    lines.append('|---:|---|---|---|---|---:|---:|---|---|')
    for r in rows:
        lines.append('| ' + ' | '.join([
            r['target_fps'],
            r['status'],
            fmt_count_pct(r['lut'], r['lut_pct']),
            fmt_count_pct(r['bram18'], r['bram18_pct']),
            fmt_count_pct(r['dsp'], r['dsp_pct']),
            fmt_float(r['fmax_mhz']),
            fmt_float(r['est_fps_fpga'], precision=1),
            fmt_mvau(r['folding_json']),
            f'{fmt_cpu_count(r["cpu_layers_json"])} ({fmt_cpu_list(r["cpu_layers_json"])})',
        ]) + ' |')
    lines.append('')
    return '\n'.join(lines)


# =============================================================================
# Cross-sweep tables (one per metric)
# =============================================================================

def cross_sweep_table(rows_by_sweep, metric_label, csv_col, fmt_fn):
    lines = []
    lines.append(f'### {metric_label}')
    lines.append('')
    lines.append('| target_fps | mlp_int8 | mlp_int4 | cnn_int8 | cnn_int4 |')
    lines.append('|---:|---|---|---|---|')
    for t in ALL_TARGETS:
        cells = [str(t)]
        for sweep in SWEEP_ORDER:
            entry = next((r for r in rows_by_sweep[sweep]
                          if int(r['target_fps']) == t), None)
            if entry is None:
                cells.append('—')
            elif entry['status'] not in ('success', 'timing_fail'):
                cells.append(f'({entry["status"]})')
            else:
                cells.append(fmt_fn(entry.get(csv_col, '')))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return '\n'.join(lines)


def render_cross_sweep_section(rows_by_sweep):
    lines = []
    lines.append('Each cell shows the metric for that (sweep, target_fps). '
                 'Failed builds show the failure status in parentheses.')
    lines.append('')
    lines.append(cross_sweep_table(rows_by_sweep, 'LUT %',     'lut_pct',
                                   fmt_pct))
    lines.append(cross_sweep_table(rows_by_sweep, 'BRAM18 %',  'bram18_pct',
                                   fmt_pct))
    lines.append(cross_sweep_table(rows_by_sweep, 'DSP %',     'dsp_pct',
                                   fmt_pct))
    lines.append(cross_sweep_table(rows_by_sweep, 'Fmax (MHz)', 'fmax_mhz',
                                   lambda v: fmt_float(v, 1)))
    lines.append(cross_sweep_table(rows_by_sweep, 'Estimated FPS (FPGA partition)',
                                   'est_fps_fpga',
                                   lambda v: fmt_float(v, 1)))
    lines.append(cross_sweep_table(rows_by_sweep, 'MVAU PE/SIMD',
                                   'folding_json', fmt_mvau))
    lines.append(cross_sweep_table(rows_by_sweep, 'CPU partition layer count',
                                   'cpu_layers_json', fmt_cpu_count))
    return '\n'.join(lines)


# =============================================================================
# Partitioning shifts (CPU ↔ FPGA layer movements)
# =============================================================================

def render_partition_shifts(rows_by_sweep):
    lines = []
    lines.append('## Partitioning shift analysis')
    lines.append('')
    lines.append('For each sweep, comparing CPU-partition layer lists across '
                 '`target_fps` values. A shift = a layer that moved from CPU to '
                 'FPGA (or vice versa) between adjacent successful builds.')
    lines.append('')
    for sweep in SWEEP_ORDER:
        rows = rows_by_sweep[sweep]
        succ = [r for r in rows if r['status'] in ('success', 'timing_fail')]
        if not succ:
            lines.append(f'- **{sweep}**: no successful builds; partition '
                         f'analysis skipped.')
            continue
        # Build (target, sorted layer-name tuple) per build
        per_build = []
        for r in succ:
            cl = parse_cpu_layers(r['cpu_layers_json'])
            names = tuple(sorted(name for op, name in cl))
            per_build.append((int(r['target_fps']), names))

        # Compress to runs of identical sets
        unique_runs = []
        for tgt, names in per_build:
            if not unique_runs or unique_runs[-1][1] != names:
                unique_runs.append((tgt, names))

        if len(unique_runs) == 1:
            tgt0, names0 = unique_runs[0]
            t_min = per_build[0][0]
            t_max = per_build[-1][0]
            lines.append(
                f'- **{sweep}**: no partition changes across '
                f'{len(per_build)} successful builds '
                f'(target_fps {t_min}–{t_max}). All builds keep the same '
                f'{len(names0)}-layer CPU partition: '
                f'[{", ".join(names0)}].')
        else:
            lines.append(
                f'- **{sweep}**: **partition changed {len(unique_runs)-1} '
                f'time(s)** across {len(per_build)} successful builds:')
            for i, (tgt, names) in enumerate(unique_runs):
                if i == 0:
                    lines.append(f'  - target_fps={tgt} (initial): CPU = '
                                 f'[{", ".join(names)}]')
                else:
                    prev = set(unique_runs[i-1][1])
                    curr = set(names)
                    moved_to_fpga = prev - curr
                    moved_to_cpu  = curr - prev
                    parts = []
                    if moved_to_fpga:
                        parts.append(
                            f'CPU→FPGA: {", ".join(sorted(moved_to_fpga))}')
                    if moved_to_cpu:
                        parts.append(
                            f'FPGA→CPU: {", ".join(sorted(moved_to_cpu))}')
                    lines.append(f'  - At target_fps={tgt}: ' +
                                 ('; '.join(parts) if parts
                                  else 'set differs but no add/remove '
                                       '(reordering only)'))
    lines.append('')
    return '\n'.join(lines)


# =============================================================================
# Failure investigation — dig into Vivado runme.log for tool_fail builds
# =============================================================================

VPROJ_HINT_RE = re.compile(r'no bitfile found\.\s*Check logs under (\S+)')


def find_impl_runme(build_log_path):
    """Read build.log, extract the Vivado project hint, return path to impl_1/runme.log."""
    if not build_log_path.exists():
        return None
    with open(build_log_path) as f:
        text = f.read()
    m = VPROJ_HINT_RE.search(text)
    if not m:
        return None
    vproj = Path(m.group(1))
    runme = vproj / 'finn_zynq_link.runs' / 'impl_1' / 'runme.log'
    return runme if runme.exists() else None


def vivado_resource_diagnostics(runme_path):
    """Pull ERROR lines + the 'Number of control sets / capacity' summary."""
    with open(runme_path) as f:
        text = f.read()
    errors = re.findall(r'(ERROR: \[(?:Place|Route|Common|Common) [^\n]{0,300})', text)
    summary_match = re.search(
        r'(Number of control sets and instances[^\n]*\n(?:[^\n]*\n){0,20})',
        text)
    return {
        'errors':  errors,
        'summary': summary_match.group(1).strip() if summary_match else None,
    }


def render_failure_investigation(rows_by_sweep):
    lines = []
    lines.append('## Failure investigation')
    lines.append('')
    failed = []
    for sweep in SWEEP_ORDER:
        for r in rows_by_sweep[sweep]:
            if r['status'] not in ('success', 'timing_fail'):
                failed.append(r)
    if not failed:
        lines.append('No failed builds to investigate.')
        lines.append('')
        return '\n'.join(lines)

    for r in failed:
        sweep = r['sweep']
        target = int(r['target_fps'])
        lines.append(f'### {sweep} target_fps={target}')
        lines.append('')
        lines.append(f'Driver-recorded status: `{r["status"]}`. '
                     f'Last step: `{r["last_step"]}`. '
                     f'Elapsed: {fmt_float(r["elapsed_s"], 0)} s.')
        lines.append('')
        lines.append('**Driver-captured error excerpt** (from `build.log`):')
        lines.append('')
        lines.append('```')
        lines.append((r.get('error_excerpt') or '').strip()[:600])
        lines.append('```')
        lines.append('')

        build_log = REPO_ROOT / r['log_path']
        runme = find_impl_runme(build_log)
        if runme is None:
            lines.append('*No deeper Vivado log located* '
                         '(no `Check logs under` hint in build.log, or the '
                         '`/tmp/finn_dev_samu/vivado_zynq_proj_*` directory '
                         'has been pruned). Without runme.log, the precise '
                         'root cause beyond the FINN-relayed exception '
                         'cannot be determined from artifacts alone.')
            lines.append('')
            continue

        diag = vivado_resource_diagnostics(runme)
        lines.append(f'**Vivado `impl_1/runme.log` excerpt** (from `{runme}`):')
        lines.append('')
        if diag['errors']:
            lines.append('Errors found:')
            lines.append('```')
            for e in diag['errors'][:6]:
                lines.append(e[:300])
            lines.append('```')
            lines.append('')
        if diag['summary']:
            lines.append('Resource demand vs device capacity:')
            lines.append('```')
            lines.append(diag['summary'][:1200])
            lines.append('```')
            lines.append('')

        # Decide actual root cause from runme errors
        errs_joined = ' '.join(diag['errors']).lower()
        is_resource = any(p in errs_joined for p in (
            '[place 30-487',          # CLB packing
            '[place 30-99',           # placer detail-placement failure
            '[route 35-',
            'unable to place',
            'utilization exceeded',
            'insufficient resources',
            'overlap of placement',
            'placement is impossible',
            'too many lut',
            'too many bram',
            'too many dsp',
        ))
        if is_resource:
            lines.append('**Root cause: actual resource exhaustion.** '
                         'The Vivado placer reports the design needs more '
                         'CLBs/LUTs than the ZU3EG provides. The driver '
                         'classified this as `tool_fail` because '
                         '`build.log` only sees FINN\'s relayed '
                         '"Synthesis failed, no bitfile found" exception — '
                         'the deep Vivado errors live in `runme.log` which '
                         'the driver does not currently inspect. **This '
                         'should be classified as `resource_fail`**, which '
                         'means Phase 2 should have run (bracket would have '
                         'been [previous successful target_fps, this '
                         'target_fps]).')
            lines.append('')
        else:
            lines.append('**Root cause: not obviously a resource issue.** '
                         'Check `runme.log` for license, OOM, disk, or '
                         'transient Vivado crash. No automatic '
                         're-classification recommended.')
            lines.append('')

    return '\n'.join(lines)


# =============================================================================
# Top-level
# =============================================================================

def render_overview(rows_by_sweep):
    lines = []
    lines.append('# FINN target_fps sweep — analysis')
    lines.append('')
    lines.append('Generated by `analyze.py`. Source data: `resource_summary.csv` + '
                 '`finn/target_fps_sweep_runs/<sweep>_fps<t>/final_hw_config.json` + '
                 '`build.log`. See `README.md` for sweep methodology.')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append('| sweep | builds | success | failed | ceiling status |')
    lines.append('|---|---:|---:|---:|---|')
    for sweep in SWEEP_ORDER:
        rows = rows_by_sweep[sweep]
        n_total = len(rows)
        n_succ  = sum(1 for r in rows if r['status'] == 'success')
        n_fail  = n_total - n_succ
        all_succ = all(r['status'] == 'success' for r in rows)
        max_succ = max((int(r['target_fps']) for r in rows
                        if r['status'] == 'success'), default=None)
        first_fail = next((int(r['target_fps']) for r in rows
                           if r['status'] != 'success'), None)
        if all_succ and n_total > 0:
            ceiling = f'all four targets passed; cap at {max_succ}'
        elif first_fail is not None and max_succ is not None:
            ceiling = f'bracketed [{max_succ}, {first_fail}]'
        elif first_fail is not None:
            ceiling = f'first build failed at {first_fail}'
        else:
            ceiling = '—'
        lines.append(f'| {sweep} | {n_total} | {n_succ} | {n_fail} | {ceiling} |')
    lines.append('')
    return '\n'.join(lines)


def main():
    rows_by_sweep = load_csv()

    sections = []
    sections.append(render_overview(rows_by_sweep))
    sections.append('# Per-sweep tables')
    sections.append('')
    for sweep in SWEEP_ORDER:
        sections.append(render_per_sweep_section(sweep, rows_by_sweep[sweep]))
    sections.append('# Cross-sweep comparison')
    sections.append('')
    sections.append(render_cross_sweep_section(rows_by_sweep))
    sections.append(render_partition_shifts(rows_by_sweep))
    sections.append(render_failure_investigation(rows_by_sweep))

    OUT_PATH.write_text('\n'.join(sections))
    print(f'wrote {OUT_PATH}  ({OUT_PATH.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
