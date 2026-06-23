#!/usr/bin/env python3
"""Re-run the one build that failed transiently — cnn_int8_deep_3_qi_fps1000 died
at 136s when the FINN container's boot-time pip install hit a DNS outage
(unfoldNd unfetchable -> brevitas_examples import error), crashing compile.py
before any FINN step. Not a resource verdict. Retry up to 3x, distinguishing the
transient env failure from a genuine verdict. Then regenerate summary.md from all
10 captured per-build dirs (read-only re-derive). Always writes RERUN_DONE."""
import importlib.util

spec = importlib.util.spec_from_file_location(
    "drv", "/home/samu/dev/CEN571-final/finn-vs-vitisai/finn/run_cnn_int8_qi_sweep.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

ONNX, FPS, LABEL = 'cnn_mnist_deep_3_qi.onnx', 1000, 'cnn_int8_deep_3_qi_fps1000'
status = '?'
try:
    row = None
    for attempt in range(1, 4):
        drv.log(f'RERUN attempt {attempt} for {LABEL}')
        row = drv.run_one(ONNX, FPS, LABEL)
        if row['verdict'] in ('FIT', 'BUST'):
            drv.log(f'RERUN attempt {attempt}: real verdict {row["verdict"]}')
            break
        fb = drv.RESULTS_ROOT / LABEL / 'finn_build.log'
        txt = fb.read_text(errors='replace') if fb.exists() else ''
        if 'brevitas_examples' in txt or 'name resolution' in txt:
            drv.log(f'RERUN attempt {attempt}: transient env/network failure — retrying')
            continue
        drv.log(f'RERUN attempt {attempt}: INCOMPLETE, non-transient — stopping retries')
        break
    status = row['verdict'] if row else '?'

    # Regenerate summary.md from every captured per-build dir (re-derive, no rebuilds).
    rows = []
    for onnx, fps, label in drv.BUILDS:
        pb = drv.RESULTS_ROOT / label
        el, ls, _ = drv.parse_console(pb / 'capture_console.log')
        rows.append(drv.write_verdict(pb, onnx, fps, label, el, ls))
    drv.write_summary(rows)
except Exception as e:
    status = f'RERUN_SCRIPT_ERROR: {type(e).__name__}: {e}'
    drv.log(status)
finally:
    (drv.RESULTS_ROOT / 'RERUN_DONE').write_text(f'{LABEL} final: {status}\n')
    drv.log(f'RERUN COMPLETE: {LABEL} -> {status}')
    print('RERUN_DONE_MARKER_WRITTEN')
