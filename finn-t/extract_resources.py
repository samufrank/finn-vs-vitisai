#!/usr/bin/env python3
"""
Extract and display FINN-T resource estimation results.

Usage:
    python3 extract_resources.py outputs/estimate_N
    python3 extract_resources.py outputs/estimate_N --json    # machine-readable output
"""
import json
import sys
from pathlib import Path


# ZU3EG (AUP-ZU3) and ZU5EG (PYNQ-ZU) resource budgets
BOARDS = {
    # Vivado-verified: xczu3eg-sfvc784-1-e (BLOCK_RAMS=216, DSP=360, no URAM)
    # BRAM_18K = BLOCK_RAMS * 2 (each RAMB36 tile splits into two RAMB18)
    "ZU3EG": {"LUT": 70560, "FF": 141120, "BRAM_18K": 432, "DSP": 360, "URAM": 0},
    # Vivado-verified: xczu5eg-sfvc784-2-e (BLOCK_RAMS=144, DSP=1248, URAM=64)
    # BRAM_18K = BLOCK_RAMS * 2 (each RAMB36 tile splits into two RAMB18)
    "ZU5EG": {"LUT": 117120, "FF": 234240, "BRAM_18K": 288, "DSP": 1248, "URAM": 64},
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def extract(output_dir):
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"

    # Load available reports
    est = load_json(report_dir / "estimate_layer_resources.json")
    hls = load_json(report_dir / "estimate_layer_resources_hls.json")
    perf = load_json(report_dir / "estimate_network_performance.json")
    cycles = load_json(report_dir / "estimate_layer_cycles.json")

    has_hls = hls is not None
    source = "HLS-synthesized" if has_hls else "estimate-only"

    # Combine: prefer HLS numbers, fall back to estimates for RTL-only nodes
    combined = {}
    if est:
        for name, res in est.items():
            if name == "total":
                continue
            if has_hls and name in hls:
                combined[name] = hls[name]
            else:
                combined[name] = res

    # If only HLS report exists (no estimate)
    if not est and hls:
        combined = {k: v for k, v in hls.items() if k != "total"}

    if not combined:
        print(f"No resource reports found in {report_dir}")
        return None

    # Try to find build config info (target_fps, model params)
    config_info = {}

    # Search for build config YAML in multiple locations
    search_dirs = [
        output_dir,             # inside output dir
        output_dir.parent,      # parent (e.g. outputs/)
        output_dir.parent.parent,  # grandparent (e.g. finn-t/)
        Path.cwd(),             # current working directory
    ]
    config_names = [
        "transformer_estimate.yml", "build_config.yaml",
        "build_config.yml", "build_config.json",
    ]
    for search_dir in search_dirs:
        if config_info.get("target_fps"):
            break
        for config_name in config_names:
            config_path = search_dir / config_name
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        import yaml as _yaml
                        cfg = _yaml.safe_load(f)
                    if cfg.get("target_fps"):
                        config_info["target_fps"] = cfg["target_fps"]
                    if cfg.get("synth_clk_period_ns"):
                        config_info["clock_mhz"] = 1000 / cfg["synth_clk_period_ns"]
                    if cfg.get("board"):
                        config_info["board"] = cfg["board"]
                    if cfg.get("fpga_part"):
                        config_info["fpga_part"] = cfg["fpga_part"]
                    break
                except Exception:
                    pass

    # Try params.yaml for model config in same search locations
    for search_dir in search_dirs:
        params_path = search_dir / "params.yaml"
        if params_path.exists():
            try:
                with open(params_path) as f:
                    import yaml as _yaml
                    params = _yaml.safe_load(f)
                if "model" in params:
                    m = params["model"]
                    config_info["model"] = (
                        f"{m.get('num_heads','')}h, D={m.get('emb_dim','')}, "
                        f"T={m.get('seq_len','')}, {m.get('bits','')}b, "
                        f"L={m.get('num_layers','')}"
                    )
                if "build" in params and not config_info.get("target_fps"):
                    config_info["target_fps"] = params["build"].get("target_fps")
                break
            except Exception:
                pass

    # Also check auto_folding_config for PE/SIMD info
    folding_path = output_dir / "auto_folding_config.yaml"
    if folding_path.exists():
        try:
            with open(folding_path) as f:
                import yaml as _yaml
                folding = _yaml.safe_load(f)
            # Get MVAU PE/SIMD from first MVAU
            for name, attrs in folding.items():
                if name.startswith("MVAU") and isinstance(attrs, dict):
                    config_info["mvau_pe"] = attrs.get("PE")
                    config_info["mvau_simd"] = attrs.get("SIMD")
                    config_info["mvau_resType"] = attrs.get("resType")
                    break
        except Exception:
            pass

    # Compute totals
    resource_keys = ["LUT", "BRAM_18K", "DSP", "FF", "URAM"]
    totals = {k: 0 for k in resource_keys}
    for res in combined.values():
        for k in resource_keys:
            totals[k] += res.get(k, 0)

    # Group by operator type
    prefixes = [
        "ScaledDotProductAttention",
        "MVAU",
        "Thresholding",
        "ElementwiseAdd",
        "ElementwiseMul",
        "ReplicateStream",
        "SplitMultiHeads",
        "MergeMultiHeads",
        "Squeeze",
        "Unsqueeze",
    ]
    groups = {}
    for prefix in prefixes:
        matching = {n: r for n, r in combined.items() if n.startswith(prefix)}
        if matching:
            g = {k: sum(r.get(k, 0) for r in matching.values()) for k in resource_keys}
            g["count"] = len(matching)
            groups[prefix] = g

    # Catch ungrouped
    grouped_names = set()
    for prefix in prefixes:
        grouped_names.update(n for n in combined if n.startswith(prefix))
    ungrouped = {n: r for n, r in combined.items() if n not in grouped_names}
    if ungrouped:
        g = {k: sum(r.get(k, 0) for r in ungrouped.values()) for k in resource_keys}
        g["count"] = len(ungrouped)
        groups["Other"] = g

    return {
        "source": source,
        "has_hls": has_hls,
        "config": config_info,
        "groups": groups,
        "totals": totals,
        "per_operator": combined,
        "performance": perf,
        "cycles": cycles,
    }


def print_report(data, output_dir):
    print(f"\n{'='*72}")
    print(f"  FINN-T Resource Report: {output_dir}")
    print(f"  Source: {data['source']}")
    cfg = data.get('config', {})
    if cfg.get('model'):
        print(f"  Model:  {cfg['model']}")
    parts = []
    if cfg.get('target_fps'):
        parts.append(f"target_fps={cfg['target_fps']}")
    if cfg.get('clock_mhz'):
        parts.append(f"clock={cfg['clock_mhz']:.0f}MHz")
    if cfg.get('mvau_pe'):
        parts.append(f"PE={cfg['mvau_pe']}")
    if cfg.get('mvau_simd'):
        parts.append(f"SIMD={cfg['mvau_simd']}")
    if cfg.get('mvau_resType'):
        parts.append(f"resType={cfg['mvau_resType']}")
    if parts:
        print(f"  Config: {', '.join(parts)}")
    if cfg.get('board'):
        print(f"  Board:  {cfg.get('board', '')} {cfg.get('fpga_part', '')}")
    print(f"{'='*72}\n")

    # Grouped summary
    print(f"  {'Operator Group':<30s} {'#':>3s} {'LUT':>8s} {'BRAM':>5s} {'DSP':>5s} {'FF':>7s}")
    print(f"  {'-'*30} {'-'*3} {'-'*8} {'-'*5} {'-'*5} {'-'*7}")

    for name, g in data["groups"].items():
        ff = f"{g['FF']:>7,.0f}" if g.get('FF', 0) > 0 else "      —"
        print(f"  {name:<30s} {g['count']:>3d} {g['LUT']:>8,.0f} {g['BRAM_18K']:>5.0f} {g['DSP']:>5.0f} {ff}")

    t = data["totals"]
    print(f"  {'-'*30} {'-'*3} {'-'*8} {'-'*5} {'-'*5} {'-'*7}")
    print(f"  {'TOTAL':<30s}     {t['LUT']:>8,.0f} {t['BRAM_18K']:>5.0f} {t['DSP']:>5.0f} {t['FF']:>7,.0f}")

    # Board fit
    print(f"\n  Board Utilization:")
    for board_name, budget in BOARDS.items():
        fits = all(t.get(k, 0) <= budget[k] for k in ["LUT", "DSP", "BRAM_18K"])
        status = "FITS" if fits else "EXCEEDS"
        print(f"    {board_name}: {status}")
        for k in ["LUT", "DSP", "BRAM_18K", "FF"]:
            if budget.get(k, 0) > 0:
                used = t.get(k, 0)
                pct = used / budget[k] * 100
                bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
                print(f"      {k:>8s}: {used:>8,.0f} / {budget[k]:>8,d}  ({pct:5.1f}%)  {bar}")

    # Attention breakdown
    if "ScaledDotProductAttention" in data["groups"]:
        attn = data["groups"]["ScaledDotProductAttention"]
        pct_lut = attn["LUT"] / t["LUT"] * 100 if t["LUT"] > 0 else 0
        pct_dsp = attn["DSP"] / t["DSP"] * 100 if t["DSP"] > 0 else 0
        print(f"\n  Attention dominance:")
        print(f"    {pct_lut:.0f}% of LUT, {pct_dsp:.0f}% of DSP")
        per_head_lut = attn["LUT"] / attn["count"]
        per_head_dsp = attn["DSP"] / attn["count"]
        per_head_bram = attn["BRAM_18K"] / attn["count"]
        print(f"    Per head: {per_head_lut:,.0f} LUT, {per_head_dsp:.0f} DSP, {per_head_bram:.0f} BRAM")

    # Performance
    if data["performance"]:
        p = data["performance"]
        print(f"\n  Performance Estimates:")
        print(f"    Throughput:     {p.get('estimated_throughput_fps', 0):,.0f} FPS")
        print(f"    Latency:        {p.get('estimated_latency_ns', 0)/1000:.1f} µs")
        print(f"    Critical path:  {p.get('critical_path_cycles', 0):,} cycles")
        print(f"    Bottleneck:     {p.get('max_cycles_node_name', 'N/A')} ({p.get('max_cycles', 0):,} cycles)")

    if not data["has_hls"]:
        print(f"\n  ⚠  Estimate-only: ScaledDotProductAttention reports 0 resources.")
        print(f"     Run through step_hw_ipgen for actual attention numbers.")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir> [--json]")
        sys.exit(1)

    output_dir = sys.argv[1]
    data = extract(output_dir)

    if data is None:
        sys.exit(1)

    if "--json" in sys.argv:
        # Machine-readable output
        out = {
            "source": data["source"],
            "config": data.get("config", {}),
            "totals": data["totals"],
            "groups": data["groups"],
            "performance": data["performance"],
        }
        print(json.dumps(out, indent=2))
    else:
        print_report(data, output_dir)
