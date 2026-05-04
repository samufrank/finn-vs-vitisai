#!/usr/bin/env python3
"""
extract_results.py - Read all benchmark JSONs in results/ and produce a verified master table.

Run from the finn-vs-vitisai repo root:
    python3 analysis/extract_results.py

Outputs:
    - analysis/verified_results.md   (markdown table)
    - analysis/verified_results.csv  (machine-readable, for diffing against sweep data)
"""

import json
import os
import re
import csv
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
SKIP_DIRS = {"kv260_archive"}

# Filename convention from STATUS.md:
# {toolchain}_{model}_{dataset}_{precision}[_{clock}][_{runtime}].json
# e.g. vta_mlp-64x32_mnist_int8_250mhz_c.json
#      finn_cnn-8x16_mnist_int8_c.json
#      dpu_mlp-64x32_mnist_int8_300mhz.json
#      finn_t_radioml_int4_merged.json

# (model, size_label) -> channel-string used in JSON filenames.
# Single source of truth: models/{cnn,mlp}.py get_*_config() dicts.
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
# Reverse: per-model {channels -> size_label}.
CHANNELS_TO_SIZE = {}
for (_m, _s), _c in SIZE_TO_CHANNELS.items():
    CHANNELS_TO_SIZE.setdefault(_m, {})[_c] = _s

# Trainable parameter counts — verified from Brevitas checkpoints.
# Precision-independent (INT4/INT8/W1A1 share weights, only quantization differs).
PARAMS = {
    ('mlp', 'tiny'):         52650,
    ('mlp', 'tiny_plus'):    80506,
    ('mlp', 'small'):       109386,
    ('mlp', 'small_plus'):  170218,
    ('mlp', 'medium'):      235146,
    ('mlp', 'large'):       535818,
    ('mlp', 'original'):    300938,
    ('mlp', 'tfc'):          59210,
    ('cnn', 'tiny'):          1442,
    ('cnn', 'small'):         5178,
    ('cnn', 'medium'):       19562,
    ('cnn', 'deep_3'):       24058,
    ('cnn', 'large'):        94186,
    # CIFAR-10 variants: extra weights in Conv1 from 3-channel input.
    ('cnn_cifar10', 'tiny'):  1610,  # plain torch CNN with bias
    ('cnn_cifar10', 'small'): 5466,  # CNN_Brevitas_qi (verified at train time)
    # Special: Transformer (FINN-T) and ResNet8 don't use the same size scheme.
    ('transformer', 'finn_t_radioml'): 122768,
}


def parse_channels_from_filename(fname):
    """Detect the FINN/VTA channel-string token, e.g. 'cnn-8x16' or
    'mlp-256x128' or 'mlp-256x256x128'. Returns (model, channels) or None."""
    m = re.search(r"(cnn|mlp)-(\d+(?:x\d+)*)", fname.lower())
    if m:
        return m.group(1), m.group(2)
    return None


def parse_vitis_b1_naming(fname):
    """Parse the Vitis AI per-model timestamp naming: <model>_<size>_<dataset>_b1_<ts>.json.
    e.g. 'mlp_large_mnist_b1_20260502_032731' -> ('mlp', 'large', 'mnist').
    Also handles 'cnn_mnist_tiny_b512' (legacy DPU) and resnet8_cifar10_*.
    Returns (model_lower, size_label, dataset) or None."""
    s = fname.lower()
    # New per-model b1 timestamp naming
    m = re.match(r"(cnn|mlp)_([a-z0-9_]+?)_(mnist|cifar10)_b1_\d+", s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    # Legacy DPU: cnn_mnist_tiny_b512
    m = re.match(r"(cnn|mlp)_(mnist|cifar10)_([a-z_]+?)_b512", s)
    if m:
        return m.group(1), m.group(3), m.group(2)
    # ResNet8 special case
    if s.startswith("resnet8"):
        ds = "cifar10" if "cifar10" in s else "mnist"
        return "resnet8", "resnet8", ds
    return None

def parse_filename(filepath, data=None):
    """Parse framework, model, precision, clock, runtime from filename, parent dir, and JSON config."""
    fname = filepath.stem
    parent = filepath.parent.name
    config = data.get("config", {}) if data else {}

    # Determine framework from parent directory (walk up past archive/python_reference)
    framework_dir = parent
    if parent in ("archive", "python_reference"):
        framework_dir = filepath.parent.parent.name

    # FINN-T prefix needs the trailing underscore — without it, "finn_tfc_*"
    # (the recognized TFC MLP benchmark, which is plain FINN) gets mis-routed
    # to FINN-T because "finn_tfc" starts with "finn_t".
    if framework_dir == "finn-t" or fname.startswith("finn_t_"):
        framework = "FINN-T"
    elif framework_dir == "finn" or fname.startswith("finn_"):
        framework = "FINN"
    elif framework_dir == "vta" or fname.startswith("vta_"):
        framework = "VTA"
    elif framework_dir in ("vitis_ai",) or fname.startswith("dpu_") or fname.startswith("vitisai_") or fname.startswith("cnn_mnist_tiny"):
        framework = "Vitis AI"
    else:
        framework = config.get("toolchain", framework_dir)

    # Determine subdirectory context (archive, python_reference, etc.)
    subdir = None
    if parent in ("archive", "python_reference"):
        subdir = parent
    elif parent == "target_fps_sweep":
        subdir = "target_fps_sweep"

    # Parse precision from filename
    precision = "?"
    for token in fname.split("_"):
        if re.match(r"int\d+", token, re.IGNORECASE):
            precision = token.upper()
            break
    # Handle INT4-o8 convention
    if "o8" in fname and "int4" in fname.lower():
        precision = "INT4-o8"
    # Fallback: check config for DPU/other non-standard filenames
    if precision == "?" and config:
        if "b512" in fname or config.get("toolchain") == "vitis_ai":
            precision = "INT8"  # DPU is always INT8

    # Parse runtime - check JSON config FIRST, then fall back to filename
    runtime = None
    if config.get("finn_runtime") == "c" or config.get("runtime") == "c":
        runtime = "C"
    elif config.get("toolchain") == "vitis_ai":
        runtime = "C++ (VART)"
    elif fname.endswith("_c") or "_c_" in fname:
        runtime = "C"
    elif "merged" in fname:
        runtime = "C"  # FINN-T merged is C runner
    elif framework == "Vitis AI":
        runtime = "C++ (VART)"

    if runtime is None:
        runtime = "Python"  # default

    # Parse clock from filename
    clock = None
    clock_match = re.search(r"(\d+)mhz", fname, re.IGNORECASE)
    if clock_match:
        clock = f"{clock_match.group(1)} MHz"
    # FINN and FINN-T all use 100 MHz (confirmed session 22 Vivado: 10ns target period)
    if clock is None and framework in ("FINN", "FINN-T"):
        clock = "100 MHz"
    # VTA clock depends on precision config
    if clock is None and framework == "VTA":
        if precision == "INT4-o8":
            clock = "166 MHz"
        elif precision == "INT4":
            clock = "200 MHz"
        else:
            clock = "250 MHz"
    # DPU B512 runs at 300/600 MHz (dual clock domain)
    if framework == "Vitis AI":
        clock = "300/600 MHz"

    # FINN-T only has a C runner (no Python runner exists)
    if framework == "FINN-T":
        runtime = "C"

    # Per-row model/size/channels — three fallback paths:
    #   1. <model>-<channels> token in fname (FINN/VTA standard naming)
    #   2. <model>_<size>_<dataset>_b1_<ts> (Vitis AI per-model naming)
    #   3. transformer / radioml / autoencoder special cases
    model_class = "?"           # 'MLP'|'CNN'|'Transformer'|'AE'|'ResNet8'
    size_label = None
    channels = None
    dataset = "MNIST"
    if "radioml" in fname:
        dataset = "RadioML"
    elif "cifar10" in fname:
        dataset = "CIFAR-10"

    chan_hit = parse_channels_from_filename(fname)
    if chan_hit:
        model_lower, channels = chan_hit
        model_class = model_lower.upper()
        size_label = CHANNELS_TO_SIZE.get(model_lower, {}).get(channels)
    else:
        b1_hit = parse_vitis_b1_naming(fname)
        if b1_hit:
            model_lower, size_label, ds_lower = b1_hit
            if model_lower == "resnet8":
                model_class = "ResNet8"
            else:
                model_class = model_lower.upper()
                channels = SIZE_TO_CHANNELS.get((model_lower, size_label))
            dataset = "CIFAR-10" if ds_lower == "cifar10" else "MNIST"

    # Special-case names without channels in filename.
    if model_class == "?":
        if "tfc" in fname:
            model_class, size_label = "MLP", "tfc"
            channels = SIZE_TO_CHANNELS.get(("mlp", "tfc"))
        elif "radioml" in fname or "transformer" in fname or framework == "FINN-T":
            model_class = "Transformer"
            size_label = "finn_t_radioml"
        elif "autoencoder" in fname:
            model_class = "AE"
        elif "mlp" in fname:
            model_class = "MLP"
        elif "cnn" in fname:
            model_class = "CNN"

    # Look up params. CIFAR-10 CNN has its own entries (different first-conv
    # weight count); other datasets share the model_class-based key.
    params = None
    if size_label:
        if model_class == "CNN" and dataset == "CIFAR-10":
            params = PARAMS.get(("cnn_cifar10", size_label))
        elif model_class == "Transformer":
            params = PARAMS.get(("transformer", size_label))
        else:
            params = PARAMS.get((model_class.lower(), size_label))

    # Display string for the Model column. Use channel-string when available
    # since that's what the JSON name encodes (no ambiguity with size labels).
    if channels:
        # Compact bracket form, e.g. '8x16' -> '[8,16]'
        bracket = "[" + ",".join(channels.split("x")) + "]"
        model_disp = f"{model_class} {bracket}"
    elif size_label and model_class != "?":
        model_disp = f"{model_class} ({size_label})"
    elif model_class == "Transformer":
        model_disp = "Transformer (122k)"
    else:
        model_disp = model_class

    # Vitis AI/DPU is always INT8 — DPU has no INT4/W1A1 path in this project.
    if framework == "Vitis AI" and precision == "?":
        precision = "INT8"

    return {
        "framework":   framework,
        "model":       model_disp,    # display column
        "model_class": model_class,   # MLP|CNN|Transformer|AE|ResNet8
        "size_label":  size_label,
        "channels":    channels,
        "dataset":     dataset,
        "precision":   precision,
        "runtime":     runtime,
        "clock":       clock,
        "subdir":      subdir,
        "params":      params,
        "filename":    filepath.name,
        "relpath":     str(filepath.relative_to(RESULTS_DIR)),
    }


def extract_metrics(data):
    """Extract key metrics from a benchmark JSON."""
    summary = data.get("summary", {})
    power = data.get("power_measurement", {})
    idle_block = data.get("idle", {})
    idle_section = power.get("idle") or idle_block.get("power") or {}
    config = data.get("config", {})

    result = {}

    # Accuracy
    result["accuracy"] = summary.get("accuracy")

    # Throughput
    result["fps"] = summary.get("throughput_fps_mean")
    if result["fps"] is None:
        result["fps"] = summary.get("throughput_fps")

    # Latency
    result["latency_ms"] = summary.get("latency_ms_mean")
    if result["latency_ms"] is None:
        result["latency_ms"] = summary.get("latency_ms")

    # Idle power
    result["idle_w"] = summary.get("idle_power_w")
    if result["idle_w"] is None:
        result["idle_w"] = idle_section.get("power_w_mean")

    # Active power
    result["active_w"] = summary.get("avg_power_w_mean")
    if result["active_w"] is None:
        result["active_w"] = summary.get("avg_power_w")

    # Dynamic power
    result["dynamic_w"] = summary.get("dynamic_power_w")
    if result["dynamic_w"] is None and result["active_w"] and result["idle_w"]:
        result["dynamic_w"] = result["active_w"] - result["idle_w"]

    # Energy per image
    result["energy_mj"] = summary.get("energy_per_image_mj_mean")
    if result["energy_mj"] is None:
        result["energy_mj"] = summary.get("energy_per_image_mj")

    # Number of runs
    runs = data.get("runs", [])
    result["n_runs"] = len(runs)

    # Board and FPGA from config
    result["board"] = config.get("board", "?")
    result["fpga_part"] = config.get("fpga_part", "?")

    # Power method
    pm = config.get("power_method", "?")
    if pm == "none" and power.get("method"):
        pm = power["method"]
    result["power_method"] = pm

    return result


def fmt(val, decimals=2, suffix=""):
    """Format a numeric value, returning '-' for None."""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return f"{val}{suffix}"


def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found. Run from finn-vs-vitisai repo root.")
        sys.exit(1)

    rows = []

    for dirpath, dirnames, filenames in os.walk(RESULTS_DIR):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in sorted(filenames):
            if not fname.endswith(".json"):
                continue
            # Skip non-benchmark files
            if fname in ("sweep_state.json", "resource_summary.csv"):
                continue

            fpath = Path(dirpath) / fname
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                print(f"WARNING: Could not parse {fpath}: {e}")
                continue

            # Must have a summary section to be a benchmark result
            if "summary" not in data:
                continue

            # MLP QI was empirically slower than classic (715 vs 1575 FPS for
            # tiny @ 1000), so QI is treated as a CNN-only optimization in
            # this project — drop MLP QI rows from the cross-framework view.
            cfg_local = data.get("config", {})
            if (cfg_local.get("partition") == "qi"
                    and "mlp" in fname.lower()):
                continue

            # Skip FINN-T raw files when merged version exists
            if fname.startswith("finn_t_") and "merged" not in fname:
                merged = fpath.parent / "finn_t_radioml_int4_merged.json"
                if merged.exists():
                    continue

            # Skip VTA transformer o8-only (superseded by o32 output tiling version)
            if "vta_transformer" in fname and "o32" not in fname:
                o32 = [f for f in fpath.parent.iterdir() if "vta_transformer" in f.name and "o32" in f.name and f.name.endswith(".json")]
                if o32:
                    continue

            # Skip archive entries that have been re-run at 9V in python_reference
            # (archive = pre-9V runs, python_reference = 9V re-runs)
            parent_dir = Path(dirpath).name
            if parent_dir == "archive":
                framework_dir = Path(dirpath).parent
                pyref_dir = framework_dir / "python_reference"
                if pyref_dir.exists() and pyref_dir.is_dir():
                    # Extract framework+model+precision prefix for matching
                    # e.g. "finn_cnn-8x16_mnist_int8" or "vta_cnn-8x16_mnist_int8"
                    parts = fname.replace(".json", "").split("_mnist_")
                    if len(parts) == 2:
                        prefix = parts[0]  # e.g. "finn_cnn-8x16" or "vta_cnn-8x16"
                        prec = parts[1].split("_")[0]  # e.g. "int8"
                        has_rerun = any(
                            prefix in f.name and prec in f.name
                            for f in pyref_dir.iterdir()
                            if f.name.endswith(".json")
                        )
                        if has_rerun:
                            continue

            meta = parse_filename(fpath, data)
            metrics = extract_metrics(data)

            rows.append({**meta, **metrics})

    if not rows:
        print("ERROR: No benchmark JSONs found.")
        sys.exit(1)

    # Sort: C runners first, then Python; within each group sort by model class then energy
    def sort_key(r):
        runtime_order = 0 if r["runtime"] in ("C", "C++ (VART)") else 1
        model_order = {"MLP": 0, "CNN": 1, "Transformer": 2, "AE": 3, "ResNet8": 4}.get(
            r.get("model_class"), 9)
        energy = r["energy_mj"] if r["energy_mj"] is not None else 9999
        return (runtime_order, model_order, energy)

    rows.sort(key=sort_key)

    # Separate C and Python rows
    c_rows = [r for r in rows if r["runtime"] in ("C", "C++ (VART)")]
    py_rows = [r for r in rows if r["runtime"] not in ("C", "C++ (VART)")]

    # Generate markdown
    header = "| Framework | Model | Dataset | Prec | Runtime | Clock | Params | Acc (%) | FPS | Lat (ms) | Idle W | Active W | Dyn W | E/inf (mJ) | Runs | Source file |"
    sep    = "|-----------|-------|---------|------|---------|-------|--------|---------|-----|----------|--------|----------|-------|------------|------|-------------|"

    def row_to_md(r):
        params_str = f"{r['params']:,}" if r.get('params') else "-"
        return (
            f"| {r['framework']} "
            f"| {r['model']} "
            f"| {r['dataset']} "
            f"| {r['precision']} "
            f"| {r['runtime']} "
            f"| {r['clock'] or '-'} "
            f"| {params_str} "
            f"| {fmt(r['accuracy'])} "
            f"| {fmt(r['fps'], 1)} "
            f"| {fmt(r['latency_ms'], 3)} "
            f"| {fmt(r['idle_w'])} "
            f"| {fmt(r['active_w'])} "
            f"| {fmt(r['dynamic_w'])} "
            f"| {fmt(r['energy_mj'])} "
            f"| {r['n_runs']} "
            f"| `{r['relpath']}` |"
        )

    lines = []
    lines.append("# Verified Results - extracted from benchmark JSONs")
    lines.append(f"")
    lines.append(f"Generated by `extract_results.py` on {os.popen('date -Iseconds').read().strip()}")
    lines.append(f"Source: `{RESULTS_DIR}/` ({len(rows)} benchmark files found, {len(SKIP_DIRS)} directories skipped)")
    lines.append("")

    # ── Cross-framework comparison (grouped by model + dataset + precision) ──
    # Pulls C-runner rows that have a known model_class + size_label, groups
    # them by (model, size, dataset, precision), and writes one small table
    # per group so FINN/VTA/Vitis AI/FINN-T appear side by side at the same
    # operating point. Variants (FINN classic / QI / fps10K / db) get their
    # own row within a group.

    def derive_variant(r):
        fname = r["filename"].lower()
        fw = r["framework"]
        if fw == "FINN":
            parts = []
            if "_qi" in fname:
                parts.append("QI")
            if "_db" in fname:
                parts.append("DB")
            m = re.search(r"fps(\d+k?)", fname)
            if m:
                parts.append(f"fps{m.group(1)}")
            return " ".join(parts) if parts else "classic"
        if fw == "VTA":
            return r["precision"]
        if fw == "Vitis AI":
            return "DPU B512"
        if fw == "FINN-T":
            return "C runner"
        return ""

    cnn_size_order = {s: i for i, s in enumerate(
        ['tiny', 'small', 'medium', 'deep_3', 'large'])}
    mlp_size_order = {s: i for i, s in enumerate(
        ['tiny', 'tiny_plus', 'small', 'small_plus',
         'medium', 'large', 'original', 'tfc'])}

    def group_sort_key(g):
        mc, sz, ds, prec = g
        order_dict = mlp_size_order if mc == "MLP" else cnn_size_order
        return (
            {"CNN": 0, "MLP": 1, "Transformer": 2, "AE": 3, "ResNet8": 4}.get(mc, 9),
            order_dict.get(sz, 999),
            ds,
            prec,
        )

    groups = {}
    for r in c_rows:
        if r.get("model_class") in (None, "?", "AE") or r.get("size_label") is None:
            continue
        key = (r["model_class"], r["size_label"], r["dataset"], r["precision"])
        groups.setdefault(key, []).append(r)

    lines.append("## Cross-framework comparison")
    lines.append("")
    lines.append(
        "Each section pins a (model, size, dataset, precision) tuple and "
        "lists every framework that produced a benchmark for it. Variant "
        "column distinguishes FINN classic / QI / DB / fps suffix from the "
        "filename. AE and unmatched-channel rows are excluded — see the "
        "C/C++ runner master table below for those.")
    lines.append("")
    for key in sorted(groups, key=group_sort_key):
        mc, sz, ds, prec = key
        lines.append(f"### {mc} {sz} | {ds} | {prec}")
        lines.append("")
        lines.append(
            "| Framework | Variant | Clock | Params | FPS | Acc (%) "
            "| E/inf (mJ) | Dyn W | Source |")
        lines.append(
            "|---|---|---|---:|---:|---:|---:|---:|---|")
        # Within a group, sort by framework then by FPS desc
        for r in sorted(groups[key],
                        key=lambda x: (x["framework"], -(x["fps"] or 0))):
            params_str = f"{r['params']:,}" if r.get("params") else "-"
            lines.append(
                f"| {r['framework']} | {derive_variant(r)} "
                f"| {r['clock'] or '-'} | {params_str} "
                f"| {fmt(r['fps'], 1)} | {fmt(r['accuracy'])} "
                f"| {fmt(r['energy_mj'])} | {fmt(r['dynamic_w'])} "
                f"| `{r['relpath']}` |")
        lines.append("")

    lines.append("## C/C++ Runner Results (master table)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in c_rows:
        lines.append(row_to_md(r))
    lines.append("")

    if py_rows:
        lines.append("## Python Runner Results (historical reference)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for r in py_rows:
            lines.append(row_to_md(r))
        lines.append("")

    # Summary: which files had no power data
    no_power = [r for r in rows if r["idle_w"] is None]
    if no_power:
        lines.append("## Files with missing power data")
        lines.append("")
        for r in no_power:
            lines.append(f"- `{r['relpath']}`: idle_w={r['idle_w']}, energy_mj={r['energy_mj']}")
        lines.append("")

    # FINN clock check
    finn_rows = [r for r in rows if r["framework"] == "FINN" and r["clock"] is None]
    if finn_rows:
        lines.append("## FINN clock unknown (not encoded in filename)")
        lines.append("")
        lines.append("FINN auto-selects clock. From session 22 Vivado reports: target period = 10 ns = 100 MHz.")
        lines.append("To confirm for each build, check the deploy's Vivado report or `settings.json`:")
        lines.append("```bash")
        lines.append('# From each FINN deploy directory:')
        lines.append('cat deploy/*/settings.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get(\'clk_period\', \'not found\'))"')
        lines.append("```")
        lines.append("")

    md_text = "\n".join(lines)
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / "verified_results.md"
    md_path.write_text(md_text)
    print(f"Wrote {md_path} ({len(rows)} results)")

    # Also write CSV for easy diffing with sweep results later
    csv_path = out_dir / "verified_results.csv"
    fieldnames = [
        "framework", "model", "model_class", "size_label", "channels",
        "dataset", "precision", "runtime", "clock", "subdir", "params",
        "accuracy", "fps", "latency_ms", "idle_w", "active_w", "dynamic_w",
        "energy_mj", "n_runs", "power_method", "board", "relpath",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {csv_path}")

    # Print quick summary to stdout
    print(f"\n{'='*80}")
    print(f"QUICK SUMMARY - C/C++ runners ({len(c_rows)} results)")
    print(f"{'='*80}")
    print(f"{'Framework':<12} {'Model':<18} {'Prec':<8} {'FPS':>8} {'E/inf':>10} {'Acc':>7} {'DynW':>6}")
    print(f"{'-'*12} {'-'*18} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*6}")
    for r in c_rows:
        print(
            f"{r['framework']:<12} {r['model']:<18} {r['precision']:<8} "
            f"{fmt(r['fps'], 1):>8} {fmt(r['energy_mj']):>8} mJ "
            f"{fmt(r['accuracy']):>6}% {fmt(r['dynamic_w']):>5}W"
        )

    if py_rows:
        print(f"\nPython runners ({len(py_rows)} results) - see verified_results.md for details")


if __name__ == "__main__":
    main()
