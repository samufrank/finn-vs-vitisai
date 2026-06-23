#!/usr/bin/env python3
"""capture_build.py — run ONE FINN build and CAPTURE the Vivado impl log.

The size_sweep / target_fps / QI sweep drivers never copied the Vivado
implementation log (impl_1/runme.log) out of the docker build dir
(/tmp/finn_dev_samu) into the repo. For busted builds, build.log only carries
the FINN-relayed "[Common 17-69] ... impl_1 failed. Unable to open"; the real
DRC/placement verdict (Place 30-487 capacity numbers, or DRC UTLZ-1
over-utilized "requires X of Y available") lives only in runme.log, which then
gets pruned from /tmp. This wrapper closes that hole: run one build with the
canonical docker invocation, then copy impl_1/runme.log (+ utilization report)
into the matching results/ subdir and grep the verdict.

Uses the EXACT docker invocation from sweep_driver.docker_cmd (copied verbatim
so this does not depend on the archived sweep modules).

Usage:
  python3 finn/capture_build.py \
      --onnx cnn_mnist_medium.onnx --fps 1000 \
      --out-subdir size_sweep_runs/cnn_int8_medium \
      --results-subdir results/finn/size_sweep \
      --label cnn_int8_medium

Exit code: 0 if runme.log was captured (whether the build fit or busted);
non-zero if the build never reached impl / no runme.log could be found
(i.e. the capture itself failed — the gate condition).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---- Paths / FINN docker env (verbatim from sweep_driver.py) ----------------

REPO_ROOT      = Path(__file__).resolve().parent.parent          # finn-vs-vitisai/
DOCKER_TAG     = 'xilinx/finn:v0.10.1-215-gd90c0878.xrt_202220.2.14.354_22.04-amd64-xrt'
FINN_BUILD_DIR = '/tmp/finn_dev_samu'
FINN_REPO_HOST = str(REPO_ROOT / 'finn-repo')
PROJECT_ROOT   = str(REPO_ROOT)


def docker_cmd(onnx, target_fps, output_dir_in_container):
    """Build a docker run argv list. Verbatim copy of sweep_driver.docker_cmd."""
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


# ---- Verdict patterns (the line(s) we must end up with) ---------------------
# Case-insensitive substring/regex search over runme.log.
VERDICT_PATTERNS = [
    r'\[Place 30-487\][^\n]*',
    r'\[DRC UTLZ-1\][^\n]*',
    r'over-utilized[^\n]*',
    r'requires \d[\d,]* of \d[\d,]*[^\n]*',
    r'\d[\d,]* of \d[\d,]* available[^\n]*',
    r'exceeded[^\n]* capacity[^\n]*',
    r'unplaced[^\n]*',
    r'ERROR: \[[^\]]*\][^\n]*',
]


def now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def clean_finn_cache():
    """Stomp HLS-IP and stitched-IP caches between builds (mirrors
    sweep_driver.clean_finn_cache). Keeps vivado_ip_cache (speeds identical IP
    regen) and the vivado_zynq_proj_* dirs (those hold the runme.logs)."""
    if not os.path.isdir(FINN_BUILD_DIR):
        return
    for entry in os.listdir(FINN_BUILD_DIR):
        if entry.startswith('code_gen_ipgen_') or entry.startswith('vivado_stitch_proj_'):
            full = os.path.join(FINN_BUILD_DIR, entry)
            try:
                shutil.rmtree(full)
            except Exception as e:
                print(f'[capture_build] WARN clean_finn_cache: {full}: {e}', flush=True)


def find_impl_dir_from_log(build_log_text):
    """FINN raises 'Synthesis failed... Check logs under <vivado_zynq_proj dir>'.
    Return that dir if present (most reliable), else None."""
    m = re.search(r'Check logs under (' + re.escape(FINN_BUILD_DIR) +
                  r'/vivado_zynq_proj_\S+)', build_log_text)
    return m.group(1) if m else None


def newest_impl_runme(since_epoch):
    """Fallback: newest vivado_zynq_proj_*/finn_zynq_link.runs/impl_1/runme.log
    whose mtime is at/after the build start. Returns path str or None."""
    pat = os.path.join(FINN_BUILD_DIR,
                       'vivado_zynq_proj_*',
                       'finn_zynq_link.runs', 'impl_1', 'runme.log')
    cands = [p for p in glob.glob(pat) if os.path.getmtime(p) >= since_epoch - 5]
    if not cands:
        # last resort: any matching runme.log regardless of mtime
        cands = glob.glob(pat)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def parse_last_step(text):
    last = None
    for m in re.finditer(r'Running step: (step_\w+) \[(\d+)/19\]', text):
        last = m.group(1)
    return last


def grep_verdict(runme_text):
    """Return list of (lineno, line) matching any verdict pattern."""
    hits = []
    lines = runme_text.splitlines()
    compiled = [re.compile(p, re.IGNORECASE) for p in VERDICT_PATTERNS]
    for i, line in enumerate(lines, 1):
        for c in compiled:
            if c.search(line):
                hits.append((i, line.strip()))
                break
    return hits


def _loud(msg):
    """Print a hard-to-miss banner to BOTH stdout and stderr."""
    bar = '!' * 64
    for stream in (sys.stdout, sys.stderr):
        print(f'\n{bar}\n[capture_build] {msg}\n{bar}', file=stream, flush=True)


def _verify(path, kind, problems):
    """Assert a copied artifact exists AND is non-empty; append to problems if not."""
    p = Path(path)
    if not p.exists():
        problems.append(f'{kind} MISSING after copy: {p}')
    elif p.stat().st_size == 0:
        problems.append(f'{kind} EMPTY (0 bytes) after copy: {p}')


# DRC over-utilization: "<RES> over-utilized ... requires <N> of such cell types
# but only <M> compatible sites are available". Same shape parse_*.py keys on.
_OVERUTIL_RE = re.compile(
    r'Resource utilization:\s*(?P<res>.+?)\s+over-utilized.*?'
    r'requires\s+(?P<req>[\d,]+)\s+of such cell types but only\s+'
    r'(?P<avail>[\d,]+)\s+compatible sites', re.IGNORECASE)


def _parse_over_utilized(runme_text):
    """[{resource, required, available, pct}] per DRC over-util line, deduped by
    resource (highest required kept), sorted by pct desc. The binding bust %."""
    best = {}
    for m in _OVERUTIL_RE.finditer(runme_text):
        res = m.group('res').strip()
        req = int(m.group('req').replace(',', ''))
        avail = int(m.group('avail').replace(',', ''))
        pct = round(100.0 * req / avail, 1) if avail else None
        if res not in best or req > best[res]['required']:
            best[res] = {'resource': res, 'required': req,
                         'available': avail, 'pct': pct}
    return sorted(best.values(), key=lambda d: -(d['pct'] or 0))


def _derive_prec_part(label, onnx):
    """Best-effort precision (int4/int8) and partition (QI/classic) from names."""
    blob = f'{label} {onnx}'.lower()
    prec = 'int4' if 'int4' in blob else 'int8' if 'int8' in blob else 'unknown'
    part = 'QI' if ('_qi' in blob or 'qi_' in blob or ' qi' in blob) else 'classic'
    return prec, part


def _finn_commit():
    """FINN provenance: finn-repo git HEAD if available, else the pinned image tag."""
    try:
        r = subprocess.run(['git', '-C', FINN_REPO_HOST, 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return DOCKER_TAG          # encodes the FINN build (…-gd90c0878…)


def write_manifest(args, results_dir, out_dir_host, build_log, fit, *, status,
                   last_step, elapsed_s, saved, over_util, runme_src,
                   binding=None, verify_ok=None):
    """Persist <label>_manifest.json: files+sizes, build config, FINN commit, date,
    and final status (OK | BUST | NO_IMPL) with the parsed bust % — so the verdict
    lives in a structured record, not only inside the raw runme.log."""
    prec, part = _derive_prec_part(args.label, args.onnx)
    files = []
    for p in [build_log] + list(saved):
        p = Path(p)
        if not p.exists():
            continue
        kind = ('build_log' if p.name == 'build.log'
                else 'impl_runme' if p.name.endswith('impl_runme.log')
                else 'utilization' if 'utilization' in p.name else 'other')
        try:
            rel = str(p.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(p)
        files.append({'name': rel, 'bytes': p.stat().st_size, 'kind': kind})
    manifest = {
        'label':             args.label,
        'model_onnx':        args.onnx,
        'precision':         prec,
        'partition':         part,
        'target_fps':        args.fps,
        'date_utc':          now_iso(),
        'finn_commit':       _finn_commit(),
        'finn_docker_image': DOCKER_TAG,
        'build_dir':         f'finn/{args.out_subdir}',
        'status':            status,                       # OK | BUST | NO_IMPL
        'last_step':         last_step,
        'elapsed_s':         elapsed_s,
        'resource_basis':    'post_route_placed (impl_1/finn_zynq_link top_wrapper)',
        'binding_resource':  binding['resource'] if binding else None,
        'binding_pct':       binding['pct'] if binding else None,
        'over_utilized':     over_util,                    # parsed % persisted here
        'runme_src':         runme_src,
        'files':             files,
        'verify':            (None if verify_ok is None
                              else 'pass' if verify_ok else 'fail'),
    }
    dest = Path(results_dir) / f'{args.label}_manifest.json'
    dest.write_text(json.dumps(manifest, indent=2) + '\n')
    return dest


def main():
    global FINN_BUILD_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True,
                    help='ONNX filename under finn/ (e.g. cnn_mnist_medium.onnx)')
    ap.add_argument('--fps', type=int, default=1000)
    ap.add_argument('--out-subdir', required=True,
                    help='Build output dir relative to finn/ '
                         '(e.g. size_sweep_runs/cnn_int8_medium)')
    ap.add_argument('--results-subdir', required=True,
                    help='Where to save captured logs, relative to repo root '
                         '(e.g. results/finn/size_sweep)')
    ap.add_argument('--label', required=True,
                    help='Filename prefix for captured artifacts '
                         '(e.g. cnn_int8_medium)')
    ap.add_argument('--timeout', type=int, default=10800,
                    help='Build hard cap in seconds (default 3h; NO 75-min cap)')
    ap.add_argument('--finn-build-dir', default=FINN_BUILD_DIR,
                    help='FINN/Vivado scratch dir (default %(default)s). Set to a '
                         'unique path to run safely alongside another FINN build '
                         'using the default dir (isolates clean_finn_cache and the '
                         'newest-runme.log capture so the two builds cannot corrupt '
                         'or cross-capture each other).')
    args = ap.parse_args()

    FINN_BUILD_DIR = args.finn_build_dir

    onnx_full = REPO_ROOT / 'finn' / args.onnx
    if not onnx_full.exists():
        print(f'[capture_build] FATAL: ONNX not found: {onnx_full}', flush=True)
        return 2

    out_dir_host = REPO_ROOT / 'finn' / args.out_subdir
    out_in_container = f'/workspace/project/finn/{args.out_subdir}'
    results_dir = REPO_ROOT / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create the FINN build dir as the invoking user (uid 1000) so the
    # --user 1000:1000 container can write; if docker auto-creates it the dir
    # would be root-owned and the build would fail on permissions.
    os.makedirs(FINN_BUILD_DIR, exist_ok=True)

    # Stomp stale HLS-IP / stitched-IP caches from any prior build so this build
    # doesn't pick them up (matches the sweep drivers). Leaves vivado_zynq_proj_*
    # (prior runme.logs already captured) and vivado_ip_cache intact.
    clean_finn_cache()

    # Give FINN a clean output dir (a stale partial dir can confuse the flow).
    if out_dir_host.exists():
        stale = out_dir_host.with_name(out_dir_host.name + '.stale')
        if stale.exists():
            shutil.rmtree(stale)
        out_dir_host.rename(stale)
        print(f'[capture_build] moved stale output aside -> {stale}', flush=True)
    out_dir_host.mkdir(parents=True, exist_ok=True)

    build_log = out_dir_host / 'build.log'
    cmd = docker_cmd(args.onnx, args.fps, out_in_container)

    print(f'[capture_build] {now_iso()} BEGIN {args.label} '
          f'onnx={args.onnx} fps={args.fps} timeout={args.timeout}s', flush=True)
    start = time.time()
    timed_out = False
    build_status = ['unknown']     # 'completed' | 'timeout' | 'interrupted'
    result_rc = [0]

    def do_capture():
        """Locate + copy + VERIFY impl artifacts and write the manifest. Runs in a
        finally so a raised/interrupted build still saves whatever reached impl_1/.
        Sets result_rc[0] to the process exit code (0 ok / 3 no-impl / 4 verify-fail)."""
        log_text = build_log.read_text(errors='replace') if build_log.exists() else ''
        last_step = parse_last_step(log_text)
        completed_ok = ('Completed successfully' in log_text or
                        '--- Build complete' in log_text)
        bitfile_present = any((out_dir_host / 'deploy' / 'bitfile').glob('*.bit')) \
            if (out_dir_host / 'deploy' / 'bitfile').is_dir() else False
        fit = (build_status[0] == 'completed' and completed_ok and bitfile_present)
        elapsed = round(time.time() - start)

        print(f'[capture_build] {now_iso()} END   {args.label} '
              f'status={build_status[0]} elapsed={elapsed}s last_step={last_step} '
              f'fit={fit}', flush=True)

        # ---- Locate the impl log (unchanged logic) --------------------------
        impl_dir = find_impl_dir_from_log(log_text)
        runme = None
        if impl_dir:
            cand = os.path.join(impl_dir, 'finn_zynq_link.runs', 'impl_1', 'runme.log')
            if os.path.exists(cand):
                runme = cand
        if runme is None:
            runme = newest_impl_runme(start)

        if runme is None:
            _loud(f'NO impl_1/runme.log under {FINN_BUILD_DIR}/vivado_zynq_proj_*/ '
                  f'— build did not reach implementation (last_step={last_step}).')
            write_manifest(args, results_dir, out_dir_host, build_log, fit,
                           status='NO_IMPL', last_step=last_step, elapsed_s=elapsed,
                           saved=[], over_util=[], runme_src=None, verify_ok=False)
            result_rc[0] = 3
            return

        # ---- Copy impl artifacts (unchanged: runme + placed util rpt(s)) ----
        impl_1 = os.path.dirname(runme)
        saved_runme = results_dir / f'{args.label}_impl_runme.log'
        shutil.copy2(runme, saved_runme)
        saved = [saved_runme]
        util_saved = []
        for rpt in glob.glob(os.path.join(impl_1, '*utilization*.rpt')):
            dest = results_dir / f'{args.label}_{os.path.basename(rpt)}'
            shutil.copy2(rpt, dest)
            saved.append(dest)
            util_saved.append(dest)

        # ---- (a) VERIFY each copied artifact exists AND is non-empty --------
        problems = []
        _verify(saved_runme, 'impl runme.log', problems)
        for d in util_saved:
            _verify(d, 'utilization rpt', problems)
        if fit and not util_saved:
            problems.append('FIT build but NO *utilization*.rpt captured — a build '
                            'that fit must have a placed util report; capture missed it')

        runme_text = Path(runme).read_text(errors='replace')
        hits = grep_verdict(runme_text)
        over_util = _parse_over_utilized(runme_text)
        binding = over_util[0] if over_util else None
        status = 'OK' if fit else 'BUST'

        # ---- (b) MANIFEST (parsed bust % persisted here, not just the log) --
        manifest_path = write_manifest(
            args, results_dir, out_dir_host, build_log, fit, status=status,
            last_step=last_step, elapsed_s=elapsed, saved=saved,
            over_util=over_util, runme_src=runme, binding=binding,
            verify_ok=(not problems))

        # ---- Report ---------------------------------------------------------
        print('', flush=True)
        print('================ CAPTURE SUMMARY ================', flush=True)
        print(f'label        : {args.label}', flush=True)
        print(f'verdict      : {status}', flush=True)
        if binding:
            print(f'binding      : {binding["resource"]} '
                  f'{binding["required"]}/{binding["available"]} = {binding["pct"]}%',
                  flush=True)
        print(f'saved runme  : {saved_runme.relative_to(REPO_ROOT)} '
              f'({saved_runme.stat().st_size} bytes)', flush=True)
        for d in util_saved:
            print(f'saved rpt    : {d.relative_to(REPO_ROOT)} '
                  f'({d.stat().st_size} bytes)', flush=True)
        print(f'manifest     : {manifest_path.relative_to(REPO_ROOT)}', flush=True)
        for lineno, line in hits[:15]:
            print(f'  L{lineno}: {line}', flush=True)
        if problems:
            _loud('CAPTURE VERIFICATION FAILED — exit 4:\n  - ' +
                  '\n  - '.join(problems))
            result_rc[0] = 4
        else:
            result_rc[0] = 0
        print('================================================', flush=True)

    try:
        with open(build_log, 'w') as f, open(os.devnull) as devnull:
            try:
                proc = subprocess.run(cmd, stdin=devnull, stdout=f,
                                      stderr=subprocess.STDOUT, timeout=args.timeout)
                build_status[0] = 'completed'
                _ = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                build_status[0] = 'timeout'
                f.write(f'\n[capture_build] TIMEOUT after {args.timeout}s\n')
    except (KeyboardInterrupt, Exception) as e:            # capture still runs (finally)
        build_status[0] = 'interrupted'
        try:
            with open(build_log, 'a') as f:
                f.write(f'\n[capture_build] BUILD RAISED/INTERRUPTED: '
                        f'{type(e).__name__}: {e}\n')
        except Exception:
            pass
        _loud(f'build raised/interrupted ({type(e).__name__}) — capturing anyway')
    finally:
        do_capture()
    return result_rc[0]


if __name__ == '__main__':
    sys.exit(main())
