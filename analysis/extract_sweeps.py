#!/usr/bin/env python3
"""
extract_sweeps.py - Extract resources, compile times, and benchmark results
from all FINN build directories, plus per-sweep deep-dive analyses.

Run from finn-vs-vitisai repo root:
    python3 analysis/extract_sweeps.py

Sources:
  - finn/output_*                                       (default builds)
  - finn/target_fps_sweep_runs/*                        (target_fps sweep)
  - finn/size_sweep_runs/*                              (model size sweep)
  - results/finn/{target_fps_sweep,size_sweep}/resource_summary.csv
  - results/finn/{target_fps_sweep,size_sweep}/benchmarks/*  (benchmark JSONs)
  - <runme.log via 'Check logs under' hint in build.log>  (failure dives)

Outputs:
  - analysis/finn_sweep_summary.md             (cross-sweep wide view, +csv)
  - analysis/finn_sweep_summary.csv            (machine-readable)
  - results/finn/size_sweep/sweep_analysis.md  (per-sweep deep dive: status,
                                                cross-precision, partition
                                                shifts, failure forensics)
  - results/finn/target_fps_sweep/sweep_analysis.md  (same shape)

The two `sweep_analysis.md` files were previously written by per-sweep
`analyze.py` scripts; this consolidates that logic so a single
invocation regenerates everything FINN-related.
"""

import json
import csv
import re
import sys
from pathlib import Path

BUDGET = {
    "LUT": 70560,
    "FF": 141120,
    "BRAM_18K": 432,
    "DSP": 360,
}

REPO = Path(".")

# ── Build directory sources ──────────────────────────────────────────────────

BUILD_SOURCES = [
    {
        "label": "default",
        "base": REPO / "finn",
        "pattern": "output_*",
        "filter": None,
    },
    {
        "label": "target_fps_sweep",
        "base": REPO / "finn" / "target_fps_sweep_runs",
        "pattern": "*",
        "filter": None,
    },
    {
        # Classic (non-QI) builds in the size_sweep_runs dir.
        "label": "size_sweep",
        "base": REPO / "finn" / "size_sweep_runs",
        "pattern": "*",
        "filter": lambda name: "_qi" not in name,
    },
    {
        # QI variants — CNN ONLY. MLP QI was empirically slower on board
        # than the classic partition (715 vs 1575 FPS for tiny @ 1000),
        # so QI is treated as a CNN-only optimization. Classic MLP stays
        # in the size_sweep entry above.
        "label": "size_sweep_qi",
        "base": REPO / "finn" / "size_sweep_runs",
        "pattern": "*",
        "filter": lambda name: "_qi" in name and name.startswith("cnn_"),
    },
]

# Benchmark directories - tier-specific to prevent cross-matching
BENCHMARK_DIRS = {
    "default": REPO / "results" / "finn",
    "target_fps_sweep": REPO / "results" / "finn" / "target_fps_sweep" / "benchmarks",
    "size_sweep": REPO / "results" / "finn" / "size_sweep",
    "size_sweep_qi": REPO / "results" / "finn" / "size_sweep_qi",
}

# Trainable parameter counts - verified from Brevitas checkpoints
# Precision-independent (INT4 and INT8 have identical counts)
PARAMS = {
    "mlp_tiny":       52650,
    "mlp_tiny_plus":  80506,
    "mlp_small":     109386,
    "mlp_small_plus":170218,
    "mlp_medium":    235146,
    "mlp_large":     535818,
    "mlp_original":  300938,
    "mlp_tfc":        59210,   # 784->64->64->64->10, recognized FINN benchmark
    "cnn_tiny":        1442,
    "cnn_small":       5178,
    "cnn_medium":     19562,
    "cnn_deep_3":     24058,
    "cnn_large":      94186,
    # CIFAR-10 variant: 3 input channels vs MNIST 1, so first conv has
    # +288 weights (3*16*3*3 - 1*16*3*3). Verified at train time:
    # cnn_cifar10_small_qi reports 5469 params (= 5466 base + 3 QI scales).
    "cnn_cifar10_small": 5466,
}

# Map (model, size_label) to the channel-string token used in benchmark
# JSON filenames (e.g. cnn 'tiny' -> '8x16', mlp 'tfc' -> '64x64x64').
# Source of truth: models/{cnn,mlp}.py get_*_config() dicts.
SIZE_TO_CHANNELS = {
    ('cnn', 'tiny'):        '8x16',
    ('cnn', 'small'):       '16x32',
    ('cnn', 'medium'):      '32x64',
    ('cnn', 'deep_3'):      '16x32x64',
    ('cnn', 'large'):       '32x64x128',
    ('mlp', 'tiny'):        '64x32',
    ('mlp', 'tiny_plus'):   '96x48',
    ('mlp', 'small'):       '128x64',
    ('mlp', 'small_plus'):  '192x96',
    ('mlp', 'medium'):      '256x128',
    ('mlp', 'large'):       '512x256',
    ('mlp', 'original'):    '256x256x128',
    ('mlp', 'tfc'):         '64x64x64',
}


def _benchmark_canonical(model, size, dataset, precision, variant, target_fps):
    """Token a benchmark JSON should contain to match this build.

    Format: <prefix>_<dataset>_<precision>[_qi][_fps<K>k] where prefix is
    '<model>-<channels>' for size sweeps and 'tfc' for the TFC special case.
    fps suffix uses board-side k shorthand (target_fps / 1000).
    Returns None when the size has no channel mapping.
    """
    if size == 'tfc':
        prefix = 'tfc'
    else:
        chan = SIZE_TO_CHANNELS.get((model.lower(), size))
        if chan is None:
            return None
        prefix = f"{model.lower()}-{chan}"
    parts = [prefix, dataset, precision.lower()]
    if variant == 'qi':
        parts.append('qi')
    if target_fps and target_fps != 1000:
        parts.append(f"fps{target_fps // 1000}k")
    return '_'.join(parts)


def extract_resources(build_dir):
    """Extract post-synth resources from a FINN build."""
    rpt = build_dir / "report" / "post_synth_resources.json"
    if not rpt.exists():
        # Check for resource_report.json (sweep builds)
        rpt = build_dir / "resource_report.json"
    if not rpt.exists():
        return None

    data = json.load(open(rpt))
    top = data.get("(top)", data)  # resource_report.json might have different structure

    result = {
        "LUT": top.get("LUT", 0),
        "FF": top.get("FF", 0),
        "BRAM_36K": top.get("BRAM_36K", 0),
        "BRAM_18K_raw": top.get("BRAM_18K", 0),
        "BRAM_18K": top.get("BRAM_36K", 0) * 2 + top.get("BRAM_18K", 0),
        "DSP": top.get("DSP", 0),
        "URAM": top.get("URAM", 0),
    }

    # Check if build failed due to resource overflow
    if result["LUT"] > BUDGET["LUT"] or result["BRAM_18K"] > BUDGET["BRAM_18K"] or result["DSP"] > BUDGET["DSP"]:
        result["resource_fail"] = True
        result["fail_reason"] = []
        if result["LUT"] > BUDGET["LUT"]:
            result["fail_reason"].append(f"LUT {result['LUT']/BUDGET['LUT']*100:.0f}%")
        if result["BRAM_18K"] > BUDGET["BRAM_18K"]:
            result["fail_reason"].append(f"BRAM {result['BRAM_18K']/BUDGET['BRAM_18K']*100:.0f}%")
        if result["DSP"] > BUDGET["DSP"]:
            result["fail_reason"].append(f"DSP {result['DSP']/BUDGET['DSP']*100:.0f}%")
    else:
        result["resource_fail"] = False

    return result


def extract_compile_time(build_dir):
    """Extract total compile time and per-step breakdown from a FINN build."""
    tps = build_dir / "time_per_step.json"
    if not tps.exists():
        return None, None
    data = json.load(open(tps))
    total = sum(data.values())
    return total, data


def extract_folding(build_dir):
    """Extract PE/SIMD folding config from a FINN build."""
    for fname in ("final_hw_config.json", "auto_folding_config.json"):
        cfg_path = build_dir / fname
        if cfg_path.exists():
            try:
                data = json.load(open(cfg_path))
                return data
            except Exception:
                pass
    return None


def get_params(model, size_label, dataset="mnist", variant="classic"):
    """Look up trainable parameter count from verified table.

    QI variants prepend a QuantIdentity (1 scale param per input channel),
    so cnn QI on MNIST (1ch) adds 1, cnn QI on CIFAR-10 (3ch) adds 3, MLP
    QI adds 1 (single input row). Approximated as +1 here since the offset
    is dwarfed by weight counts in every model size we ship.
    """
    if model == "?" or size_label is None:
        return None
    # MNIST keys are stored without dataset prefix (legacy). Non-MNIST
    # carries the dataset segment.
    if dataset and dataset != "mnist":
        key = f"{model.lower()}_{dataset}_{size_label}"
    else:
        key = f"{model.lower()}_{size_label}"
    base = PARAMS.get(key)
    if base is None:
        return None
    return base + 1 if variant == "qi" else base


def extract_wns(build_dir):
    """Extract WNS from post_route_timing.rpt."""
    rpt = build_dir / "report" / "post_route_timing.rpt"
    if not rpt.exists():
        return None
    text = rpt.read_text()
    m = re.search(r"^\s*([-\d.]+)\s+[-\d.]+\s+\d+\s+\d+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def parse_build_name(dirname, source_label):
    """Parse model, precision, target_fps, variant, dataset from a build dir name.

    Recognized formats:
      - target_fps sweep:  cnn_int8_fps10000           (tiny, classic)
      - classic size:      cnn_int8_tiny               (default fps=1000)
      - QI MNIST:          cnn_int4_tiny_qi
      - QI MNIST + fps:    cnn_int8_medium_qi_fps200
                           cnn_int8_tiny_qi_fps10000
      - QI CIFAR-10:       cnn_int8_cifar10_small_qi
    """
    name = dirname
    result = {
        "model": "?",
        "precision": "?",
        "target_fps": None,
        "size_label": None,
        "variant": "classic",
        "dataset": "mnist",
    }

    # target_fps sweep (legacy, no _qi): "mlp_int4_fps500000"
    fps_match = re.match(r"^(mlp|cnn)_(int\d+)_fps(\d+)$", name)
    if fps_match:
        result["model"] = "MLP" if fps_match.group(1) == "mlp" else "CNN"
        result["precision"] = fps_match.group(2).upper()
        result["target_fps"] = int(fps_match.group(3))
        result["size_label"] = "tiny"  # all target_fps builds use the tiny model
        return result

    # QI variant (with optional cifar10 dataset and trailing _fps<N>):
    # Note the lazy match on size_label to avoid swallowing the _qi.
    qi_match = re.match(
        r"^(mlp|cnn)_(int\d+)(?:_(cifar10))?_(.+?)_qi(?:_fps(\d+))?$", name
    )
    if qi_match:
        result["model"] = "MLP" if qi_match.group(1) == "mlp" else "CNN"
        result["precision"] = qi_match.group(2).upper()
        if qi_match.group(3):
            result["dataset"] = qi_match.group(3)
        result["size_label"] = qi_match.group(4)
        result["variant"] = "qi"
        # No fps suffix on QI dir = the QI sweep's default of 1000.
        result["target_fps"] = int(qi_match.group(5)) if qi_match.group(5) else 1000
        return result

    # Classic size sweep: "mlp_int8_tiny", "cnn_int4_tiny", "mlp_int8_tiny_plus"
    size_match = re.match(r"^(mlp|cnn)_(int\d+)_(.+)$", name)
    if size_match:
        result["model"] = "MLP" if size_match.group(1) == "mlp" else "CNN"
        result["precision"] = size_match.group(2).upper()
        result["size_label"] = size_match.group(3)
        return result

    # Default builds: "output_mlp_mnist_tiny", "output_cnn_mnist_tiny_int4",
    # "output_tfc_mnist_int8", "output_autoencoder_toycar_brevitas".
    if "mlp" in name:
        result["model"] = "MLP"
    elif "cnn" in name:
        result["model"] = "CNN"
    elif "tfc" in name:
        # Recognized FINN benchmark — TFC (Tiny Fully Connected) is an MLP.
        result["model"] = "MLP"
    elif "autoencoder" in name:
        result["model"] = "AE"

    # Precision: int4 / int8 / w<W>a<A> (binary/ternary FINN-style names
    # like w1a1 = weights 1-bit, activations 1-bit). Without this the
    # default-INT8 fallback would silently match a w1a1 build to the INT8
    # benchmark JSON (TFC w1a1 vs TFC int8 are different models).
    wa_match = re.search(r"w(\d+)a(\d+)", name)
    if "int4" in name:
        result["precision"] = "INT4"
    elif wa_match:
        result["precision"] = f"W{wa_match.group(1)}A{wa_match.group(2)}".upper()
    else:
        result["precision"] = "INT8"

    # Detect dataset segment so default-source benchmark matching can avoid
    # cross-dataset false positives (e.g. cnn_cifar10 build ≠ mnist JSON).
    if "_cifar10" in name:
        result["dataset"] = "cifar10"

    if source_label == "default":
        result["target_fps"] = 1000  # FINN default
        if "tfc" in name:
            result["size_label"] = "tfc"
        else:
            result["size_label"] = "tiny"

    return result


def find_benchmark(build_name, model, precision, target_fps, source_label,
                   dataset="mnist", size=None, variant="classic"):
    """Find a benchmark JSON matching this build.

    Primary: canonical channel-string match (e.g. cnn-8x16_mnist_int4_qi for
    cnn tiny INT4 QI). Works for size_sweep, size_sweep_qi, target_fps_sweep,
    and default-tier builds whose JSONs follow the standard naming.

    Fallback (default tier only): looser model+precision+dataset match for
    JSONs without a channel token (early/legacy artifacts) — guards against
    fps/qi/cifar10 cross-matches.
    """
    bdir = BENCHMARK_DIRS.get(source_label)
    if bdir is None or not bdir.exists():
        return None

    canonical = _benchmark_canonical(
        model, size, dataset, precision, variant, target_fps)

    # Primary pass: canonical channel-string match. The token already
    # encodes model, channels, dataset, precision, qi flag, and (k-shorthand)
    # fps, so substring presence is enough.
    if canonical is not None:
        for f in bdir.iterdir():
            if not f.name.endswith(".json"):
                continue
            if canonical in f.name.lower():
                return f

    # Fallback for default-tier JSONs that don't carry channels (TFC,
    # autoencoder, the very early '_c' artifacts). Skip qi/fps JSONs to
    # avoid grabbing a sweep build's data.
    if source_label == "default":
        ms = model.lower()
        ps = precision.lower()
        for f in bdir.iterdir():
            if not f.name.endswith(".json"):
                continue
            fname = f.name.lower()
            if ms not in fname or ps not in fname:
                continue
            if dataset == "cifar10" and "_cifar10" not in fname:
                continue
            if dataset != "cifar10" and "_cifar10" in fname:
                continue
            if "fps" in fname or "_qi" in fname:
                continue
            return f

    return None


def steady_dynamic_w(data, fallback=None):
    """Steady-state dynamic power: mean of runs 2..N `fnb58_power.dynamic_power_w`
    (run 1 dropped as cold-start transient; see results/POWER_REPORTING_POLICY.md).
    Falls back to `fallback` (the all-3 summary value) when per-run power is absent."""
    vals = [r["fnb58_power"]["dynamic_power_w"]
            for r in (data.get("runs") or [])[1:]
            if isinstance(r.get("fnb58_power"), dict)
            and r["fnb58_power"].get("dynamic_power_w") is not None]
    return sum(vals) / len(vals) if vals else fallback


def extract_benchmark(bench_path):
    """Extract key metrics from a benchmark JSON."""
    if bench_path is None:
        return {}
    try:
        data = json.load(open(bench_path))
    except Exception:
        return {}

    summary = data.get("summary", {})
    result = {}
    result["accuracy"] = summary.get("accuracy")
    result["fps"] = summary.get("throughput_fps_mean", summary.get("throughput_fps"))
    result["energy_mj"] = summary.get("energy_per_image_mj_mean", summary.get("energy_per_image_mj"))
    result["dynamic_w"] = steady_dynamic_w(data, summary.get("dynamic_power_w"))  # steady-state (runs 2-3)
    result["idle_w"] = summary.get("idle_power_w")
    result["active_w"] = summary.get("avg_power_w_mean", summary.get("avg_power_w"))
    result["bench_file"] = bench_path.name
    return result


def pct(val, resource):
    if val is None or val == 0:
        return "-"
    return f"{val / BUDGET[resource] * 100:.1f}%"


def fmt(val, decimals=2):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def main():
    rows = []

    for source in BUILD_SOURCES:
        base = source["base"]
        if not base.exists():
            print(f"SKIP: {base} not found")
            continue

        for build_dir in sorted(base.glob(source["pattern"])):
            if not build_dir.is_dir():
                continue

            dirname = build_dir.name
            source_label = source["label"]

            # Per-source filter splits dirs that share a base path (e.g.
            # size_sweep vs size_sweep_qi both glob finn/size_sweep_runs/).
            sf = source.get("filter")
            if sf is not None and not sf(dirname):
                continue

            meta = parse_build_name(dirname, source_label)

            resources = extract_resources(build_dir)
            compile_time, time_steps = extract_compile_time(build_dir)
            wns = extract_wns(build_dir)
            params = get_params(
                meta["model"], meta["size_label"],
                dataset=meta.get("dataset", "mnist"),
                variant=meta.get("variant", "classic"),
            )

            # Find matching benchmark (tier-specific, no cross-matching)
            bench_path = find_benchmark(
                dirname, meta["model"], meta["precision"], meta["target_fps"],
                source_label,
                dataset=meta.get("dataset", "mnist"),
                size=meta.get("size_label"),
                variant=meta.get("variant", "classic"),
            )
            bench = extract_benchmark(bench_path)

            row = {
                "source": source_label,
                "dirname": dirname,
                "model": meta["model"],
                "precision": meta["precision"],
                "target_fps": meta["target_fps"],
                "size_label": meta["size_label"],
                "variant": meta.get("variant", "classic"),
                "dataset": meta.get("dataset", "mnist"),
                "params": params,
                "compile_s": compile_time,
                "wns": wns,
                "resource_fail": resources["resource_fail"] if resources else None,
                "LUT": resources["LUT"] if resources else None,
                "FF": resources["FF"] if resources else None,
                "BRAM_18K": resources["BRAM_18K"] if resources else None,
                "DSP": resources["DSP"] if resources else None,
                **bench,
            }
            rows.append(row)

            status = "FAIL" if (resources and resources["resource_fail"]) else "OK"
            fps_str = f"{bench.get('fps', 0):.0f} FPS" if bench.get("fps") else "no bench"
            time_str = f"{compile_time:.0f}s ({compile_time/60:.1f}m)" if compile_time else "no timing"
            lut_str = f"{resources['LUT']:,} ({pct(resources['LUT'], 'LUT')})" if resources else "no resources"
            print(f"  {status:4s}  {source_label:16s} {dirname:30s} {lut_str:25s} {time_str:20s} {fps_str}")

    if not rows:
        print("ERROR: No FINN builds found.")
        sys.exit(1)

    # ── Generate markdown ──
    analysis_dir = REPO / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    lines = []
    lines.append("# FINN Build Summary - All Configurations")
    lines.append("")
    lines.append(f"ZU3EG budget: {BUDGET['LUT']:,} LUT | {BUDGET['FF']:,} FF | "
                 f"{BUDGET['BRAM_18K']} BRAM_18K | {BUDGET['DSP']} DSP")
    lines.append("")

    # ── Compile time summary ──
    lines.append("## Compile Times")
    lines.append("")
    lines.append("| Source | Build | Model | Prec | target_fps | Compile (min) |")
    lines.append("|--------|-------|-------|------|------------|--------------|")
    for r in sorted(rows, key=lambda x: (x["source"], x["model"], x["precision"], x.get("compile_s") or 0)):
        if r["compile_s"] is None:
            continue
        fps_str = f"{r['target_fps']:,}" if r["target_fps"] else "-"
        lines.append(
            f"| {r['source']} | {r['dirname']} | {r['model']} | {r['precision']} "
            f"| {fps_str} | {r['compile_s']/60:.1f} |"
        )

    # ── Resource table ──
    lines.append("")
    lines.append("## Resource Utilization")
    lines.append("")
    lines.append("| Source | Build | Model | Prec | Params | target_fps | LUT | LUT% | FF | BRAM_18K | BRAM% | DSP | WNS (ns) | Status |")
    lines.append("|--------|-------|-------|------|--------|------------|-----|------|----|----------|-------|-----|----------|--------|")
    for r in sorted(rows, key=lambda x: (x["model"], x["precision"], x.get("target_fps") or 0, x.get("size_label") or "")):
        if r["LUT"] is None:
            continue
        lut = r["LUT"]
        status = "FAIL" if r.get("resource_fail") else "OK"
        fps_str = f"{r['target_fps']:,}" if r["target_fps"] else "-"
        params_str = f"{r['params']:,}" if r.get("params") else "-"
        wns_str = f"+{r['wns']:.3f}" if r.get("wns") is not None and r["wns"] >= 0 else (f"{r['wns']:.3f}" if r.get("wns") is not None else "-")
        lines.append(
            f"| {r['source']} | {r['dirname']} | {r['model']} | {r['precision']} "
            f"| {params_str} | {fps_str} "
            f"| {lut:,} | {pct(lut, 'LUT')} "
            f"| {r['FF']:,} "
            f"| {r['BRAM_18K']} | {pct(r['BRAM_18K'], 'BRAM_18K')} "
            f"| {r['DSP']} "
            f"| {wns_str} "
            f"| {status} |"
        )

    # ── Benchmarked results ──
    benched = [r for r in rows if r.get("fps")]
    if benched:
        lines.append("")
        lines.append("## Benchmarked Configurations")
        lines.append("")
        lines.append("| Source | Build | Model | Prec | Params | target_fps | LUT% | Acc (%) | FPS | E/inf (mJ) | Dyn W | Bench file |")
        lines.append("|--------|-------|-------|------|--------|------------|------|---------|-----|------------|-------|------------|")
        for r in sorted(benched, key=lambda x: (x["model"], x["precision"], x.get("energy_mj") or 9999)):
            fps_str = f"{r['target_fps']:,}" if r["target_fps"] else "-"
            params_str = f"{r['params']:,}" if r.get("params") else "-"
            lines.append(
                f"| {r['source']} | {r['dirname']} | {r['model']} | {r['precision']} "
                f"| {params_str} | {fps_str} "
                f"| {pct(r['LUT'], 'LUT') if r['LUT'] else '-'} "
                f"| {fmt(r.get('accuracy'))} "
                f"| {fmt(r.get('fps'), 1)} "
                f"| {fmt(r.get('energy_mj'))} "
                f"| {fmt(r.get('dynamic_w'))} "
                f"| `{r.get('bench_file', '-')}` |"
            )

    # ── Failed builds + resource constraints ──
    # Pulls failure rows from each sweep's CSV, then follows the build's
    # compile log to the Vivado runme.log to extract specific over-utilized
    # resources (LUT/CARRY8/BRAM/DSP/etc) with used/available ratios.
    failure_csvs = [
        ("size_sweep",    REPO / "results" / "finn" / "size_sweep" / "resource_summary.csv"),
        ("size_sweep_qi", REPO / "results" / "finn" / "size_sweep_qi" / "resource_summary.csv"),
    ]
    all_failures = []
    for src_label, fpath in failure_csvs:
        for f in extract_failed_builds(fpath):
            # Skip MLP QI failures — QI is CNN-only in this project.
            if src_label == "size_sweep_qi" and not f.get("sweep", "").startswith("cnn"):
                continue
            f["source"] = src_label
            all_failures.append(f)

    # Cross-row inference: when a tier (b)/(c) row has no direct DRC, look
    # for a smaller-size confirmed bust in the same sweep + target_fps. If
    # found, the larger build is implied also-bust because adding channels
    # only adds parallelism cost (LUT/BRAM/DSP all monotonic in size).
    _SIZE_ORDER = {
        'cnn': ['tiny', 'small', 'medium', 'deep_3', 'large'],
        'mlp': ['tiny', 'tiny_plus', 'small', 'small_plus',
                'medium', 'original', 'large'],
    }

    def _model_kind(sweep):
        return 'cnn' if 'cnn' in sweep else ('mlp' if 'mlp' in sweep else None)

    def _implied_note(f):
        """If f has no constraints AND a smaller-size sibling at the same
        target_fps did bust on DRC, return an inference note."""
        if f['constraints']:
            return ''
        kind = _model_kind(f['sweep'])
        if kind is None or kind not in _SIZE_ORDER:
            return ''
        order = _SIZE_ORDER[kind]
        if f['size'] not in order:
            return ''
        my_idx = order.index(f['size'])
        # Find smaller-size confirmed busts in the same sweep + target_fps
        candidates = []
        for other in all_failures:
            if other is f or other['sweep'] != f['sweep']:
                continue
            if other.get('target_fps') != f.get('target_fps'):
                continue
            if other['size'] not in order or not other['constraints']:
                continue
            other_idx = order.index(other['size'])
            if other_idx < my_idx:
                candidates.append((other_idx, other))
        if not candidates:
            return ''
        candidates.sort(key=lambda x: -x[0])  # nearest smaller first
        ref = candidates[0][1]
        res_names = ', '.join(c['resource'] for c in ref['constraints'][:2])
        return (f"_implied bust — `{ref['size']}` at same target_fps "
                f"confirmed {res_names} bust; `{f['size']}` strictly larger_")

    if all_failures:
        lines.append("")
        lines.append("## Failed Builds — Resource Constraints")
        lines.append("")
        lines.append("Informative failures only — manual kills are "
                     "filtered out. Rows fall into three tiers: "
                     "**(a)** `[DRC UTLZ-1]` errors from `impl_1/runme.log` "
                     "under `/tmp/finn_dev_samu/vivado_zynq_proj_*` give "
                     "the specific over-utilized resource + used/available "
                     "ratio. "
                     "**(b)** `[Common 17-69]` rows mean Vivado reached "
                     "impl_1 and synth_design failed, but the runme.log "
                     "was later pruned — we know the build is too big to "
                     "fit, just not which resource busted first. "
                     "**(c)** Timeout rows (elapsed = 4500s) hit the "
                     "75-min driver cap before synth could produce a "
                     "verdict — informative as a softer 'too big' signal "
                     "than (b). ZU3EG budget: 70,560 LUT / 8,820 CARRY8 / "
                     "432 BRAM_18K / 360 DSP.")
        lines.append("")
        lines.append("| source | sweep | size | target_fps | last_step | elapsed (s) | over-utilized resource(s) |")
        lines.append("|--------|-------|------|-----------:|-----------|------------:|---------------------------|")
        for f in all_failures:
            if f["constraints"]:
                cons = "<br>".join(
                    f"**{c['resource']}** {c['used']:,} / {c['avail']:,} "
                    f"({c['pct']:.1f}%)"
                    for c in f["constraints"]
                )
            elif f.get("is_timeout") and not f["error_excerpt"]:
                cons = ("_timed out at 75-min driver cap "
                        "(synth never produced a verdict; specifics unknown)_")
            else:
                # Tier (b): Vivado errored but DRC log not located.
                excerpt = f["error_excerpt"].replace("|", "/")[:120]
                cons = f"_no Vivado DRC found; excerpt: {excerpt or '(empty)'}_"
            implied = _implied_note(f)
            if implied:
                cons = cons + "<br>" + implied
            lines.append(
                f"| {f['source']} | {f['sweep']} | {f['size']} "
                f"| {f['target_fps']} | {f['last_step'] or '-'} "
                f"| {f['elapsed_s']} | {cons} |"
            )

    # ── Overlay comparison note ──
    # Numbers (counts, ranges, totals) regenerate from the rows actually
    # extracted this run — kept honest as the sweep evolves.
    timed_rows = [r for r in rows if r.get("compile_s")]
    n_total = len(timed_rows)
    successes = [r for r in timed_rows if not r.get("resource_fail")]
    finn_min_s = min((r["compile_s"] for r in successes), default=0)
    finn_max_s = max((r["compile_s"] for r in successes), default=0)
    finn_total_h = sum(r["compile_s"] for r in timed_rows) / 3600.0
    fastest = min(successes, key=lambda r: r["compile_s"], default=None)
    slowest = max(successes, key=lambda r: r["compile_s"], default=None)

    lines.append("")
    lines.append("## Overlay Compile Time Comparison")
    lines.append("")
    lines.append("VTA and DPU deploy new models without bitstream recompilation.")
    lines.append("Bitstream build is a one-time cost; model deployment is weight loading + instruction generation.")
    lines.append("")
    lines.append("| Framework | Bitstream build (one-time) | Per-model deploy | Source |")
    lines.append("|-----------|--------------------------|-----------------|--------|")
    lines.append(
        f"| FINN | {finn_min_s/60:.0f}–{finn_max_s/60:.0f} min per "
        f"model×precision×folding ({n_total} builds measured) | "
        f"N/A — model IS the bitstream | `time_per_step.json` from each build |")
    lines.append("| VTA | ~12 min Vivado synth+impl (HLS separate, in Docker) | 2.6 s (weight export + TVM cross-compile) | Vivado `wait_on_runs` elapsed; `time export_vta_model.py` |")
    lines.append("| DPU | 15.5 min (Vivado synth+impl+bitstream) | ~1 min (vai_c_xir, unmeasured) | `runme.log` timestamps: 15:39:57 to 15:55:29 |")
    lines.append("")
    if fastest and slowest:
        lines.append(
            f"FINN compile times scale with model + folding: fastest "
            f"`{fastest['dirname']}` at {fastest['compile_s']/60:.1f} min, "
            f"slowest `{slowest['dirname']}` at {slowest['compile_s']/60:.1f} min. "
            f"All models here are small (1.4k–536k params); production "
            f"models would take significantly longer.")
    lines.append(
        f"This sweep set ({n_total} builds counting successes + failures) "
        f"required **~{finn_total_h:.1f} hours** of FINN compilation total "
        f"on a single machine. An overlay user deploys the same {n_total} "
        f"models in under 2 minutes total (VTA: 2.6 s × {n_total}; DPU: ~1 min × {n_total}).")

    md_text = "\n".join(lines)
    md_path = analysis_dir / "finn_sweep_summary.md"
    md_path.write_text(md_text)
    print(f"\nWrote {md_path}")

    # ── CSV ──
    csv_path = analysis_dir / "finn_sweep_summary.csv"
    fieldnames = [
        "source", "dirname", "model", "precision", "params", "target_fps", "size_label",
        "compile_s", "resource_fail", "LUT", "FF", "BRAM_18K", "DSP", "wns",
        "accuracy", "fps", "energy_mj", "dynamic_w", "idle_w", "active_w", "bench_file",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {csv_path}")

    # ── Per-sweep deep-dive markdowns ──
    # Replaces results/finn/{size,target_fps}_sweep/analyze.py.
    for sweep_id, sweep_cfg in DEEP_DIVE_SWEEPS.items():
        try:
            md_path = render_sweep_deep_dive(sweep_cfg)
            print(f"Wrote {md_path}")
        except FileNotFoundError as e:
            print(f"SKIP deep-dive for {sweep_id}: {e}")


# =============================================================================
# Per-sweep deep-dive (ported from results/finn/{size,target_fps}_sweep/analyze.py)
# =============================================================================
#
# Each sweep is parameterized over its "axis" — the column that varies within a
# single (model, precision) combination. size_sweep varies `size`; target_fps
# sweep varies `target_fps`. Cross-sweep tables reflect the axis: size_sweep has
# different size sets per model so we render one table per model;
# target_fps_sweep uses one table with all 4 sweeps as columns.
#
# Outputs go alongside each sweep's resource_summary.csv so the markdown sits
# next to the data it describes. The two file paths are also referenced
# directly by the playbook.

SWEEP_ORDER = ['mlp_int8', 'mlp_int4', 'cnn_int8', 'cnn_int4']

SIZES_BY_MODEL = {
    'mlp': ['tiny', 'tiny_plus', 'small', 'small_plus',
            'medium', 'large', 'original'],
    'cnn': ['tiny', 'small', 'medium', 'deep_3', 'large'],
}
ALL_TARGET_FPS = [1000, 10000, 100000, 500000]

DEEP_DIVE_SWEEPS = {
    'size_sweep': {
        'csv_path':    REPO / 'results' / 'finn' / 'size_sweep' / 'resource_summary.csv',
        'runs_dir':    REPO / 'finn' / 'size_sweep_runs',
        'out_path':    REPO / 'results' / 'finn' / 'size_sweep' / 'sweep_analysis.md',
        'axis_col':    'size',
        'axis_label':  'size',
        'title':       'FINN size sweep — analysis',
        'description': (
            'Companion experiment to the `target_fps` sweep. Same four '
            '(model, precision) combinations; this sweep varies '
            '**model size** at fixed `target_fps=1000`. Goals: (1) '
            'characterize how FINN resource utilization scales with '
            'topology size; (2) test the design-space-headroom claim — '
            'that INT4 admits larger models than INT8 on the same ZU3EG.'),
        'cross_layout': 'per_model',  # MLP and CNN have different size sets
    },
    'target_fps_sweep': {
        'csv_path':    REPO / 'results' / 'finn' / 'target_fps_sweep' / 'resource_summary.csv',
        'runs_dir':    REPO / 'finn' / 'target_fps_sweep_runs',
        'out_path':    REPO / 'results' / 'finn' / 'target_fps_sweep' / 'sweep_analysis.md',
        'axis_col':    'target_fps',
        'axis_label':  'target_fps',
        'title':       'FINN target_fps sweep — analysis',
        'description': (
            'Sweep of the `target_fps` build parameter (1k → 500k) for '
            'each (model, precision) combination at fixed `tiny` size. '
            'Goal: find the FPGA throughput ceiling per combination, '
            'i.e. the largest `target_fps` for which FINN can still place '
            'and route the design.'),
        'cross_layout': 'all_in_one',
    },
}


def _ddd_load_csv(csv_path, axis_col):
    rows_by_sweep = {s: [] for s in SWEEP_ORDER}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            sweep = row['sweep']
            if sweep in rows_by_sweep:
                rows_by_sweep[sweep].append(row)
    if axis_col == 'target_fps':
        for s in SWEEP_ORDER:
            rows_by_sweep[s].sort(key=lambda r: int(r['target_fps']))
    else:
        # size_sweep: sort by canonical size order per model
        for s in SWEEP_ORDER:
            model = 'mlp' if s.startswith('mlp') else 'cnn'
            order = {sz: i for i, sz in enumerate(SIZES_BY_MODEL[model])}
            rows_by_sweep[s].sort(key=lambda r: order.get(r['size'], 999))
    return rows_by_sweep


def _ddd_fmt_count_pct(count_str, pct_str):
    if not count_str or not pct_str:
        return '—'
    return f'{int(float(count_str))} ({float(pct_str):.1f}%)'


def _ddd_fmt_pct(pct_str):
    if not pct_str:
        return '—'
    return f'{float(pct_str):.1f}%'


def _ddd_fmt_float(v, precision=2):
    if v == '' or v is None:
        return '—'
    try:
        return f'{float(v):.{precision}f}'
    except Exception:
        return str(v)


def _ddd_parse_folding(folding_json_str):
    if not folding_json_str:
        return {}
    try:
        return json.loads(folding_json_str)
    except Exception:
        return {}


def _ddd_fmt_mvau(folding_json_str):
    f = _ddd_parse_folding(folding_json_str)
    parts = []
    for k, v in f.items():
        if 'MVAU' not in k:
            continue
        pe = v.get('PE') if v.get('PE') is not None else '—'
        simd = v.get('SIMD') if v.get('SIMD') is not None else '—'
        parts.append(f'{k}=PE{pe}/SIMD{simd}')
    return ', '.join(parts) if parts else '—'


def _ddd_parse_cpu_layers(cpu_layers_json_str):
    if not cpu_layers_json_str:
        return []
    try:
        return json.loads(cpu_layers_json_str)
    except Exception:
        return []


def _ddd_fmt_cpu_count(cpu_layers_json_str):
    cl = _ddd_parse_cpu_layers(cpu_layers_json_str)
    return str(len(cl)) if cl else '—'


def _ddd_fmt_cpu_list(cpu_layers_json_str):
    cl = _ddd_parse_cpu_layers(cpu_layers_json_str)
    if not cl:
        return '—'
    return ', '.join(name for op, name in cl)


def _ddd_render_overview(rows_by_sweep, cfg):
    axis = cfg['axis_col']
    lines = [f'# {cfg["title"]}', '']
    lines.append('Generated by `analysis/extract_sweeps.py`. Source data: '
                 '`resource_summary.csv` + per-build `final_hw_config.json` + '
                 '`build.log`. To regenerate, run `python3 analysis/extract_sweeps.py` '
                 'from the repo root.')
    lines.append('')
    lines.append(cfg['description'])
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    if axis == 'target_fps':
        lines.append('| sweep | builds | success | failed | ceiling status |')
        lines.append('|---|---:|---:|---:|---|')
        for sweep in SWEEP_ORDER:
            rows = rows_by_sweep[sweep]
            n_total = len(rows)
            n_succ = sum(1 for r in rows if r['status'] == 'success')
            n_fail = n_total - n_succ
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
    else:
        lines.append('| sweep | builds | success | failed | range |')
        lines.append('|---|---:|---:|---:|---|')
        for sweep in SWEEP_ORDER:
            rows = rows_by_sweep[sweep]
            n_total = len(rows)
            n_succ = sum(1 for r in rows if r['status'] == 'success')
            n_fail = n_total - n_succ
            succ_axis = [r[axis] for r in rows if r['status'] == 'success']
            fail_axis = [r[axis] for r in rows if r['status'] != 'success']
            if not fail_axis:
                range_note = f'all {n_total} {axis}s succeeded'
            else:
                range_note = (f'success: [{", ".join(succ_axis)}]; '
                              f'fail: [{", ".join(fail_axis)}]')
            lines.append(f'| {sweep} | {n_total} | {n_succ} | {n_fail} | {range_note} |')
    lines.append('')
    return '\n'.join(lines)


def _ddd_render_per_sweep(sweep, rows, cfg):
    axis = cfg['axis_col']
    label = cfg['axis_label']
    lines = [f'## {sweep}', '']
    if not rows:
        lines.append('No builds.')
        lines.append('')
        return '\n'.join(lines)
    n_succ = sum(1 for r in rows if r['status'] == 'success')
    statuses = ', '.join(f'{r[axis]}={r["status"]}' for r in rows)
    lines.append(f'{n_succ}/{len(rows)} successful. Sequence: {statuses}.')
    lines.append('')
    if axis == 'target_fps':
        lines.append(f'| {label} | status | LUT | BRAM18 | DSP | Fmax (MHz) | est FPS | MVAU PE/SIMD | CPU layers |')
        lines.append('|---:|---|---|---|---|---:|---:|---|---|')
    else:
        lines.append(f'| {label} | status | LUT | BRAM18 | DSP | Fmax (MHz) | est FPS | MVAU PE/SIMD | CPU layers |')
        lines.append('|---|---|---|---|---|---:|---:|---|---|')
    for r in rows:
        lines.append('| ' + ' | '.join([
            r[axis],
            r['status'],
            _ddd_fmt_count_pct(r['lut'], r['lut_pct']),
            _ddd_fmt_count_pct(r['bram18'], r['bram18_pct']),
            _ddd_fmt_count_pct(r['dsp'], r['dsp_pct']),
            _ddd_fmt_float(r['fmax_mhz']),
            _ddd_fmt_float(r['est_fps_fpga'], precision=1),
            _ddd_fmt_mvau(r['folding_json']),
            f'{_ddd_fmt_cpu_count(r["cpu_layers_json"])} ({_ddd_fmt_cpu_list(r["cpu_layers_json"])})',
        ]) + ' |')
    lines.append('')
    return '\n'.join(lines)


def _ddd_cross_table_per_model(rows_by_sweep, model, metric_label, csv_col, fmt_fn):
    lines = []
    sweeps = [f'{model}_int8', f'{model}_int4']
    sizes = SIZES_BY_MODEL[model]
    lines.append(f'### {metric_label}  ({model.upper()})')
    lines.append('')
    lines.append(f'| size | {sweeps[0]} | {sweeps[1]} |')
    lines.append('|---|---|---|')
    for sz in sizes:
        cells = [sz]
        for sweep in sweeps:
            entry = next((r for r in rows_by_sweep[sweep] if r['size'] == sz), None)
            if entry is None:
                cells.append('—')
            elif entry['status'] not in ('success', 'timing_fail'):
                cells.append(f'({entry["status"]})')
            else:
                cells.append(fmt_fn(entry.get(csv_col, '')))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return '\n'.join(lines)


def _ddd_cross_table_all_in_one(rows_by_sweep, metric_label, csv_col, fmt_fn):
    lines = [f'### {metric_label}', '']
    lines.append('| target_fps | mlp_int8 | mlp_int4 | cnn_int8 | cnn_int4 |')
    lines.append('|---:|---|---|---|---|')
    for t in ALL_TARGET_FPS:
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


def _ddd_render_cross_section(rows_by_sweep, cfg):
    axis = cfg['axis_col']
    layout = cfg['cross_layout']
    lines = []
    if layout == 'per_model':
        lines.append('Each cell shows the metric for that (sweep, size). '
                     'Failed builds show the failure status in parentheses. '
                     'Two tables per metric since MLP and CNN have different size sets.')
    else:
        lines.append('Each cell shows the metric for that (sweep, target_fps). '
                     'Failed builds show the failure status in parentheses.')
    lines.append('')
    metrics = [
        ('LUT %',                          'lut_pct',     _ddd_fmt_pct),
        ('BRAM18 %',                       'bram18_pct',  _ddd_fmt_pct),
        ('DSP %',                          'dsp_pct',     _ddd_fmt_pct),
        ('Fmax (MHz)',                     'fmax_mhz',    lambda v: _ddd_fmt_float(v, 1)),
        ('Estimated FPS (FPGA partition)', 'est_fps_fpga', lambda v: _ddd_fmt_float(v, 1)),
        ('MVAU PE/SIMD',                   'folding_json', _ddd_fmt_mvau),
        ('CPU partition layer count',      'cpu_layers_json', _ddd_fmt_cpu_count),
    ]
    for label, col, fmt in metrics:
        if layout == 'per_model':
            lines.append(_ddd_cross_table_per_model(rows_by_sweep, 'mlp', label, col, fmt))
            lines.append(_ddd_cross_table_per_model(rows_by_sweep, 'cnn', label, col, fmt))
        else:
            lines.append(_ddd_cross_table_all_in_one(rows_by_sweep, label, col, fmt))
    return '\n'.join(lines)


def _ddd_render_partition_shifts(rows_by_sweep, cfg):
    axis = cfg['axis_col']
    label = cfg['axis_label']
    lines = ['## Partitioning shift analysis', '']
    lines.append(f'For each sweep, comparing CPU-partition layer lists '
                 f'across `{label}` values. A shift = a layer that moved '
                 f'from CPU to FPGA (or vice versa) between adjacent '
                 f'successful builds.')
    lines.append('')
    for sweep in SWEEP_ORDER:
        rows = rows_by_sweep[sweep]
        succ = [r for r in rows if r['status'] in ('success', 'timing_fail')]
        if not succ:
            lines.append(f'- **{sweep}**: no successful builds; partition '
                         f'analysis skipped.')
            continue
        per_build = []
        for r in succ:
            cl = _ddd_parse_cpu_layers(r['cpu_layers_json'])
            names = tuple(sorted(name for op, name in cl))
            axis_val = int(r[axis]) if axis == 'target_fps' else r[axis]
            per_build.append((axis_val, names))
        unique_runs = []
        for v, names in per_build:
            if not unique_runs or unique_runs[-1][1] != names:
                unique_runs.append((v, names))
        if len(unique_runs) == 1:
            _, names0 = unique_runs[0]
            v_min = per_build[0][0]
            v_max = per_build[-1][0]
            lines.append(
                f'- **{sweep}**: no partition changes across '
                f'{len(per_build)} successful builds '
                f'({label} {v_min} – {v_max}). All builds keep the same '
                f'{len(names0)}-layer CPU partition: '
                f'[{", ".join(names0)}].')
        else:
            lines.append(
                f'- **{sweep}**: **partition changed {len(unique_runs)-1} '
                f'time(s)** across {len(per_build)} successful builds:')
            for i, (v, names) in enumerate(unique_runs):
                if i == 0:
                    lines.append(f'  - {label}={v} (initial): CPU = '
                                 f'[{", ".join(names)}]')
                else:
                    prev = set(unique_runs[i-1][1])
                    curr = set(names)
                    moved_to_fpga = prev - curr
                    moved_to_cpu  = curr - prev
                    parts = []
                    if moved_to_fpga:
                        parts.append(f'CPU→FPGA: {", ".join(sorted(moved_to_fpga))}')
                    if moved_to_cpu:
                        parts.append(f'FPGA→CPU: {", ".join(sorted(moved_to_cpu))}')
                    lines.append(f'  - At {label}={v}: ' +
                                 ('; '.join(parts) if parts
                                  else 'set differs but no add/remove '
                                       '(reordering only)'))
    lines.append('')
    return '\n'.join(lines)


_VPROJ_HINT_RE = re.compile(r'no bitfile found\.\s*Check logs under (\S+)')

# Vivado [DRC UTLZ-1] reports over-utilized resources with the resource
# name plus a "requires N of such cell types but only M compatible sites"
# clause. Extracting both gives the specific over-used resource and the
# overflow ratio (e.g. "LUT as Logic 73318/70560 = 103.9%").
_DRC_UTLZ_RE = re.compile(
    r'\[DRC UTLZ-1\][^(]*:\s*(.+?)\s+over-utilized'
    r'.*?requires (\d+) of such cell types but only (\d+)',
    re.DOTALL,
)


def _classify_resource_failure(errors):
    """Pick over-utilized resources out of Vivado DRC error strings.
    Returns a list of dicts in input order; each: resource/used/avail/pct.
    Deduplicates by resource name so reading the same DRC error twice
    doesn't double-list."""
    results = []
    seen = set()
    for err in errors:
        for m in _DRC_UTLZ_RE.finditer(err):
            resource = m.group(1).strip()
            if resource in seen:
                continue
            seen.add(resource)
            used = int(m.group(2))
            avail = int(m.group(3))
            results.append({
                'resource': resource,
                'used':  used,
                'avail': avail,
                'pct':   100.0 * used / avail if avail else 0.0,
            })
    return results


def extract_failed_builds(csv_path):
    """Read a sweep's resource_summary.csv; return informative failed-row
    dicts (filters out manual kills and timeout-with-no-output that don't
    teach us anything).

    Each row: sweep, size, target_fps, status, last_step, elapsed_s,
    error_excerpt (from CSV), and constraints (list of resource overflow
    dicts, or empty if no DRC UTLZ-1 errors were located).

    Filter rules (skip if all conditions hold):
      - No constraints (Vivado runme.log absent or no DRC UTLZ-1 lines)
      - Empty error_excerpt (FINN driver caught nothing in stdout)
    These together indicate a manual kill or 75-min timeout that aborted
    before Vivado emitted any error worth recording. Rows that hit a real
    Vivado [Common 17-69] error stay even if the runme.log was later pruned
    — we still know synthesis was attempted and failed."""
    if not csv_path.exists():
        return []
    out = []
    seen = set()  # dedupe (sweep, size, target_fps) — retries from sweep restarts
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'success':
                continue
            key = (row.get('sweep', ''), row.get('size', ''),
                   row.get('target_fps', ''))
            if key in seen:
                continue
            log_rel = row.get('log_path', '')
            compile_log = REPO / log_rel if log_rel else None
            constraints = []
            errors = []
            if compile_log and compile_log.exists():
                runme = _ddd_find_impl_runme(compile_log)
                if runme is not None:
                    diag = _ddd_vivado_diagnostics(runme)
                    errors = diag.get('errors', [])
                    constraints = _classify_resource_failure(errors)
            excerpt = (row.get('error_excerpt', '') or '').strip()
            # Detect 75-min driver-cap timeout (subprocess.call timeout=4500
            # in qi_overnight_sweep.py). These are informative — the build
            # is so resource-heavy synth couldn't produce a verdict in 75
            # min — even though we lack DRC or Vivado-error specifics.
            try:
                elapsed = float(row.get('elapsed_s', 0) or 0)
            except (TypeError, ValueError):
                elapsed = 0.0
            is_timeout = elapsed >= 4500
            # Skip uninformative manual kills (no constraints, no excerpt,
            # not a timeout).
            if not constraints and not excerpt and not is_timeout:
                continue
            seen.add(key)
            out.append({
                'sweep':         row.get('sweep', ''),
                'size':          row.get('size', ''),
                'target_fps':    row.get('target_fps', ''),
                'status':        row.get('status', ''),
                'last_step':     row.get('last_step', ''),
                'elapsed_s':     row.get('elapsed_s', ''),
                'error_excerpt': excerpt,
                'constraints':   constraints,
                'errors_raw':    errors,
                'is_timeout':    is_timeout,
            })
    return out


def _ddd_find_impl_runme(build_log_path):
    if not build_log_path.exists():
        return None
    text = build_log_path.read_text()
    m = _VPROJ_HINT_RE.search(text)
    if not m:
        return None
    vproj = Path(m.group(1))
    runme = vproj / 'finn_zynq_link.runs' / 'impl_1' / 'runme.log'
    return runme if runme.exists() else None


def _ddd_vivado_diagnostics(runme_path):
    text = runme_path.read_text()
    # Catch resource-exhaustion errors at any Vivado stage:
    #   [DRC UTLZ-1]   pre-placer over-utilization
    #   [Place 30-X]   placer-time failure
    #   [Route 35-X]   router-time failure
    #   [Common 17-X]  top-level wrappers re-raising the above
    errors = re.findall(r'(ERROR: \[(?:Place|Route|Common|DRC) [^\n]{0,300})', text)
    summary_match = re.search(
        r'(Number of control sets and instances[^\n]*\n(?:[^\n]*\n){0,20})', text)
    return {
        'errors':  errors,
        'summary': summary_match.group(1).strip() if summary_match else None,
    }


def _ddd_render_failure_investigation(rows_by_sweep, cfg):
    axis = cfg['axis_col']
    label = cfg['axis_label']
    lines = ['## Failure investigation', '']
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
        v = r[axis]
        lines.append(f'### {sweep} {label}={v}')
        lines.append('')
        lines.append(f'Driver-recorded status: `{r["status"]}`. '
                     f'Last step: `{r["last_step"]}`. '
                     f'Elapsed: {_ddd_fmt_float(r["elapsed_s"], 0)} s.')
        lines.append('')
        lines.append('**Driver-captured error excerpt** (from `build.log`):')
        lines.append('')
        lines.append('```')
        lines.append((r.get('error_excerpt') or '').strip()[:600])
        lines.append('```')
        lines.append('')

        build_log = REPO / r['log_path']
        runme = _ddd_find_impl_runme(build_log)
        if runme is None:
            lines.append('*No deeper Vivado log located* — runme.log under '
                         '`/tmp/finn_dev_samu/vivado_zynq_proj_*/` may have '
                         "been pruned by a subsequent build's cache stomp "
                         '(`clean_finn_cache` removes vivado_stitch_proj_* '
                         'and code_gen_ipgen_* but leaves vivado_zynq_proj_* '
                         'alone; PYNQ stale-state mitigation may still '
                         'remove them across reboots).')
            lines.append('')
            continue

        diag = _ddd_vivado_diagnostics(runme)
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

        errs_joined = ' '.join(diag['errors']).lower()
        is_resource = any(p in errs_joined for p in (
            '[drc utlz-1', '[place 30-487', '[place 30-99', '[route 35-',
            'unable to place', 'utilization exceeded', 'insufficient resources',
            'overlap of placement', 'placement is impossible',
            'too many lut', 'too many bram', 'too many dsp', 'over-utilized',
        ))
        if is_resource:
            lines.append('**Root cause: actual resource exhaustion.** '
                         'Vivado placer rejects the design because the '
                         "ZU3EG can't fit it. FINN re-raises as "
                         '"Synthesis failed, no bitfile found" which the '
                         'driver classifies as `tool_fail`. Should be '
                         're-classified as `resource_fail`.')
            lines.append('')
        else:
            lines.append('**Root cause: not obviously a resource issue.** '
                         'Check `runme.log` for license, OOM, disk, or '
                         'transient Vivado crash. No automatic '
                         're-classification recommended.')
            lines.append('')
    return '\n'.join(lines)


def render_sweep_deep_dive(cfg):
    """Render one sweep's deep-dive markdown. Returns the output path."""
    if not cfg['csv_path'].exists():
        raise FileNotFoundError(f"resource_summary.csv not found: {cfg['csv_path']}")
    rows_by_sweep = _ddd_load_csv(cfg['csv_path'], cfg['axis_col'])
    sections = []
    sections.append(_ddd_render_overview(rows_by_sweep, cfg))
    sections.append('# Per-sweep tables')
    sections.append('')
    for sweep in SWEEP_ORDER:
        sections.append(_ddd_render_per_sweep(sweep, rows_by_sweep[sweep], cfg))
    sections.append('# Cross-sweep comparison')
    sections.append('')
    sections.append(_ddd_render_cross_section(rows_by_sweep, cfg))
    sections.append(_ddd_render_partition_shifts(rows_by_sweep, cfg))
    sections.append(_ddd_render_failure_investigation(rows_by_sweep, cfg))
    cfg['out_path'].write_text('\n'.join(sections))
    return cfg['out_path']


if __name__ == "__main__":
    main()
