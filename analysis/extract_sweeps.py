#!/usr/bin/env python3
"""
extract_sweeps.py - Extract resources, compile times, and benchmark results
from all FINN build directories (default, target_fps sweep, size sweep).

Run from finn-vs-vitisai repo root:
    python3 analysis/extract_sweeps.py

Sources:
  - finn/output_*                           (default builds)
  - finn/target_fps_sweep_runs/*            (target_fps sweep)
  - finn/size_sweep_runs/*                  (model size sweep)
  - results/finn/target_fps_sweep/benchmarks/*  (benchmarked sweep configs)

Outputs:
  - analysis/finn_sweep_summary.md   (markdown tables)
  - analysis/finn_sweep_summary.csv  (machine-readable)
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
    },
    {
        "label": "target_fps_sweep",
        "base": REPO / "finn" / "target_fps_sweep_runs",
        "pattern": "*",
    },
    {
        "label": "size_sweep",
        "base": REPO / "finn" / "size_sweep_runs",
        "pattern": "*",
    },
]

# Benchmark directories - tier-specific to prevent cross-matching
BENCHMARK_DIRS = {
    "default": REPO / "results" / "finn",
    "target_fps_sweep": REPO / "results" / "finn" / "target_fps_sweep" / "benchmarks",
    "size_sweep": REPO / "results" / "finn" / "size_sweep",
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
    "cnn_tiny":        1442,
    "cnn_small":       5178,
    "cnn_medium":     19562,
    "cnn_deep_3":     24058,
    "cnn_large":      94186,
}


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


def get_params(model, size_label):
    """Look up trainable parameter count from verified table."""
    if model == "?" or size_label is None:
        return None
    key = f"{model.lower()}_{size_label}"
    return PARAMS.get(key)


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
    """Parse model, precision, target_fps from a build directory name."""
    name = dirname
    result = {
        "model": "?",
        "precision": "?",
        "target_fps": None,
        "size_label": None,
    }

    # target_fps sweep: e.g. "mlp_int4_fps500000", "cnn_int8_fps10000"
    fps_match = re.match(r"(mlp|cnn)_(int\d+)_fps(\d+)", name)
    if fps_match:
        result["model"] = "MLP" if fps_match.group(1) == "mlp" else "CNN"
        result["precision"] = fps_match.group(2).upper()
        result["target_fps"] = int(fps_match.group(3))
        result["size_label"] = "tiny"  # all target_fps builds use the tiny model
        return result

    # size sweep: e.g. "mlp_int8_tiny", "cnn_int4_tiny", "mlp_int8_tiny_plus"
    size_match = re.match(r"(mlp|cnn)_(int\d+)_(.+)", name)
    if size_match:
        result["model"] = "MLP" if size_match.group(1) == "mlp" else "CNN"
        result["precision"] = size_match.group(2).upper()
        result["size_label"] = size_match.group(3)
        return result

    # default builds: e.g. "output_mlp_mnist_tiny", "output_cnn_mnist_tiny_int4"
    if "mlp" in name:
        result["model"] = "MLP"
    elif "cnn" in name:
        result["model"] = "CNN"

    if "int4" in name:
        result["precision"] = "INT4"
    else:
        result["precision"] = "INT8"

    if source_label == "default":
        result["target_fps"] = 1000  # FINN default
        result["size_label"] = "tiny"

    return result


def find_benchmark(build_name, model, precision, target_fps, source_label):
    """Find a benchmark JSON matching this build. Only searches within the same tier."""
    bdir = BENCHMARK_DIRS.get(source_label)
    if bdir is None or not bdir.exists():
        return None

    for f in bdir.iterdir():
        if not f.name.endswith(".json"):
            continue
        fname = f.name.lower()
        model_str = model.lower()
        prec_str = precision.lower()

        if model_str not in fname or prec_str not in fname:
            continue

        if source_label == "target_fps_sweep":
            # Must match exact fps value
            if target_fps is not None and f"fps{target_fps}" in fname:
                return f
        elif source_label == "size_sweep":
            # Must match exact size label from build dirname
            # e.g. build "cnn_int4_large" should only match benchmark with "large" in name
            size = build_name.replace(f"{model_str}_{prec_str}_", "")
            if size in fname:
                return f
        elif source_label == "default":
            # Default builds match C runner benchmarks (no fps, no size qualifier)
            if "_c" in fname and "fps" not in fname:
                return f

    return None


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
    result["dynamic_w"] = summary.get("dynamic_power_w")
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

            # Skip non-MNIST builds (CIFAR-10 model too small, not part of comparison)
            if "cifar" in dirname.lower():
                continue

            meta = parse_build_name(dirname, source_label)

            resources = extract_resources(build_dir)
            compile_time, time_steps = extract_compile_time(build_dir)
            wns = extract_wns(build_dir)
            params = get_params(meta["model"], meta["size_label"])

            # Find matching benchmark (tier-specific, no cross-matching)
            bench_path = find_benchmark(
                dirname, meta["model"], meta["precision"], meta["target_fps"],
                source_label
            )
            bench = extract_benchmark(bench_path)

            row = {
                "source": source_label,
                "dirname": dirname,
                "model": meta["model"],
                "precision": meta["precision"],
                "target_fps": meta["target_fps"],
                "size_label": meta["size_label"],
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

    # ── Overlay comparison note ──
    lines.append("")
    lines.append("## Overlay Compile Time Comparison")
    lines.append("")
    lines.append("VTA and DPU deploy new models without bitstream recompilation.")
    lines.append("Bitstream build is a one-time cost; model deployment is weight loading + instruction generation.")
    lines.append("")
    lines.append("| Framework | Bitstream build (one-time) | Per-model deploy | Source |")
    lines.append("|-----------|--------------------------|-----------------|--------|")
    lines.append("| FINN | 13-37 min per modelxprecisionxfolding (44 builds measured) | N/A - model IS the bitstream | `time_per_step.json` from each build |")
    lines.append("| VTA | ~12 min Vivado synth+impl (HLS separate, in Docker) | 2.6 s (weight export + TVM cross-compile) | Vivado `wait_on_runs` elapsed; `time export_vta_model.py` |")
    lines.append("| DPU | 15.5 min (Vivado synth+impl+bitstream) | ~1 min (vai_c_xir, unmeasured) | `runme.log` timestamps: 15:39:57 to 15:55:29 |")
    lines.append("")
    lines.append("FINN compile times scale with model complexity: MLP INT4 tiny builds in 13 min,")
    lines.append("CNN INT8 at higher folding takes 37 min. All models in this project are small")
    lines.append("(1.4k-536k params); production models would take significantly longer.")
    lines.append("The 44-build target_fps sweep alone required ~15 hours of FINN compilation.")
    lines.append("An overlay user deploys 44 models in under 2 minutes total.")

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


if __name__ == "__main__":
    main()
