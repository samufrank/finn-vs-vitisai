#!/usr/bin/env python3
"""
extract_resources.py - Extract FPGA resource utilization from all framework builds.

Run from finn-vs-vitisai repo root:
    python3 analysis/extract_resources.py

Sources:
  - VTA/DPU/FINN-T: Vivado utilization_placed.rpt files
  - FINN MLP/CNN: post_synth_resources.json from FINN build outputs
  - Timing: WNS from timing reports or post_route_timing.rpt

Outputs:
  - analysis/resource_utilization.md        (consolidated markdown table)
  - analysis/vivado_utilization_reports/    (archived copies of source reports)
"""

import json
import re
import shutil
from pathlib import Path

BUDGET = {
    "LUT": 70560,
    "FF": 141120,
    "BRAM_18K": 432,
    "DSP": 360,
}

BASE = Path.home() / "dev" / "CEN571-final"
REPO = BASE / "finn-vs-vitisai"

# ── Vivado .rpt sources ──────────────────────────────────────────────────────

VIVADO_REPORTS = {
    "VTA INT8 (250 MHz)": {
        "path": BASE / "tvm-v0.12.0/3rdparty/vta-hw/hardware/xilinx/vta.runs/impl_1/vta_wrapper_utilization_placed.rpt",
        "timing": BASE / "tvm-v0.12.0/3rdparty/vta-hw/hardware/xilinx/vta.runs/impl_1/vta_wrapper_timing_summary_routed.rpt",
        "clock": "250 MHz",
        "note": "Rebuild needed - currently overwritten by INT4-o8. Use pasted numbers until rebuilt.",
    },
    "VTA INT4-o4 (200 MHz)": {
        "path": BASE / "tvm-v0.12.0/3rdparty/vta-hw/build/hardware/xilinx/vivado/ultra96_1x16_i4w4a32_15_14_17_17/vta.runs/impl_1/vta_wrapper_utilization_placed.rpt",
        "timing": None,
        "clock": "200 MHz",
        "note": "Option A buffer-halved INT4. Used for MLP INT4.",
    },
    "VTA INT4-o8 (166 MHz)": {
        "path": REPO / "bitstreams/int4_o8_166mhz_build_artifacts/vta_wrapper_utilization_placed.rpt",
        "timing": REPO / "bitstreams/int4_o8_166mhz_build_artifacts/vta_wrapper_timing_summary_routed.rpt",
        "clock": "166 MHz",
        "note": "Mixed-precision INT4 compute / INT8 output. Used for CNN INT4.",
    },
    "DPU B512 (300/600 MHz)": {
        "path": BASE / "aup_zu3_dpu/aup_zu3_dpu.runs/impl_1/dpu_wrapper_utilization_placed.rpt",
        "timing": None,
        "clock": "300/600 MHz",
        "note": "DPUCZDX8G B512.",
    },
    "FINN-T Transformer INT4": {
        "path": BASE / "finn-t/FINN_TMP/vivado_zynq_proj_48s7dd2z/finn_zynq_link.runs/impl_1/top_wrapper_utilization_placed.rpt",
        "timing": BASE / "finn-t/FINN_TMP/vivado_zynq_proj_48s7dd2z/finn_zynq_link.runs/impl_1/top_wrapper_timing_summary_routed.rpt",
        "clock": "100 MHz",
        "note": "fix9 trained transformer. 122k params, RadioML.",
    },
}

# ── FINN JSON sources ────────────────────────────────────────────────────────

FINN_BUILDS = {
    "FINN MLP INT8": {
        "dir": REPO / "finn/output_mlp_mnist_tiny",
        "clock": "100 MHz",
        "note": "MLP [64,32], PE=SIMD=1, default target_fps.",
    },
    "FINN MLP INT4": {
        "dir": REPO / "finn/output_mlp_mnist_tiny_int4",
        "clock": "100 MHz",
        "note": "MLP [64,32], PE=SIMD=1, default target_fps.",
    },
    "FINN CNN INT8": {
        "dir": REPO / "finn/output_cnn_mnist_tiny",
        "clock": "100 MHz",
        "note": "CNN [8,16], PE=1/SIMD=3, default target_fps.",
    },
    "FINN CNN INT4": {
        "dir": REPO / "finn/output_cnn_mnist_tiny_int4",
        "clock": "100 MHz",
        "note": "CNN [8,16], PE=1/SIMD=3, default target_fps.",
    },
}


def extract_vivado_utilization(rpt_path):
    """Parse a Vivado utilization_placed.rpt."""
    if not rpt_path.exists():
        return None
    text = rpt_path.read_text()
    result = {}

    m = re.search(r"\| Date\s+:\s+(.+)", text)
    result["report_date"] = m.group(1).strip() if m else "?"

    m = re.search(r"\| Device\s+:\s+(\S+)", text)
    result["device"] = m.group(1).strip() if m else "?"

    m = re.search(r"\|\s*CLB LUTs\s+\|\s*(\d+)\s*\|", text)
    if m:
        result["LUT"] = int(m.group(1))

    m = re.search(r"\|\s*CLB Registers\s+\|\s*(\d+)\s*\|", text)
    if m:
        result["FF"] = int(m.group(1))

    m_r36 = re.search(r"\|\s*RAMB36/FIFO\*?\s+\|\s*(\d+)\s*\|", text)
    m_r18 = re.search(r"\|\s*RAMB18\s+\|\s*(\d+)\s*\|", text)
    if m_r36 and m_r18:
        result["BRAM_18K"] = int(m_r36.group(1)) * 2 + int(m_r18.group(1))

    m = re.search(r"\|\s*DSPs?\s+\|\s*(\d+)\s*\|.*?\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", text)
    if m:
        result["DSP"] = int(m.group(1))

    return result


def extract_vivado_wns(timing_path):
    """Extract WNS from a Vivado timing_summary_routed.rpt."""
    if timing_path is None or not timing_path.exists():
        return None
    text = timing_path.read_text()
    # Format: "WNS(ns)      TNS(ns)  ..."  or table row
    m = re.search(r"^\s*([-\d.]+)\s+[-\d.]+\s+\d+\s+\d+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    m = re.search(r"WNS\(ns\)\s*:\s*([-\d.]+)", text)
    if m:
        return float(m.group(1))
    return None


def extract_finn_resources(build_dir):
    """Extract resources from a FINN build's post_synth_resources.json."""
    rpt = build_dir / "report" / "post_synth_resources.json"
    if not rpt.exists():
        return None
    data = json.load(open(rpt))
    top = data.get("(top)", {})
    return {
        "LUT": top.get("LUT", 0),
        "FF": top.get("FF", 0),
        "BRAM_18K": top.get("BRAM_36K", 0) * 2 + top.get("BRAM_18K", 0),
        "DSP": top.get("DSP", 0),
        "report_date": "FINN build",
        "device": "xczu3eg (Ultra96 def)",
    }


def extract_finn_wns(build_dir):
    """Extract WNS from a FINN build's post_route_timing.rpt."""
    rpt = build_dir / "report" / "post_route_timing.rpt"
    if not rpt.exists():
        return None
    text = rpt.read_text()
    m = re.search(r"^\s*([-\d.]+)\s+[-\d.]+\s+\d+\s+\d+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    m = re.search(r"WNS\(ns\)\s*:\s*([-\d.]+)", text)
    if m:
        return float(m.group(1))
    return None


def pct(val, resource):
    if val is None:
        return "-"
    return f"{val / BUDGET[resource] * 100:.1f}%"


def main():
    archive_dir = REPO / "analysis" / "vivado_utilization_reports"
    archive_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ── Vivado reports ──
    # First pass: collect all results so we can detect overwrites
    vivado_results = {}
    print("=== Vivado reports ===")
    for label, info in VIVADO_REPORTS.items():
        result = extract_vivado_utilization(info["path"])
        if result is None:
            print(f"  MISSING: {label}")
            rows.append({"label": label, "status": "MISSING", "note": info["note"],
                         "clock": info["clock"]})
            continue
        vivado_results[label] = (result, info)

    # Detect VTA INT8 overwrite: if it matches INT4-o8 exactly, it was overwritten
    int8 = vivado_results.get("VTA INT8 (250 MHz)")
    o8 = vivado_results.get("VTA INT4-o8 (166 MHz)")
    if int8 and o8:
        r8, r_o8 = int8[0], o8[0]
        if (r8.get("LUT") == r_o8.get("LUT") and r8.get("FF") == r_o8.get("FF")
                and r8.get("DSP") == r_o8.get("DSP")):
            print(f"  OVERWRITTEN: VTA INT8 (250 MHz) - using canonical values from earlier build report.")
            del vivado_results["VTA INT8 (250 MHz)"]
            # Canonical values from Vivado post-implementation report (pre-INT4 build)
            # DSP/BRAM structurally fixed by HLS IPs; LUT/FF may differ slightly at 250 MHz
            rows.append({
                "label": "VTA INT8 (250 MHz)", "status": "OK",
                "LUT": 22173, "FF": 31001, "BRAM_18K": 273, "DSP": 277,
                "report_date": "pre-INT4 build (on-disk report overwritten)",
                "clock": "250 MHz", "wns": 0.146,
                "note": "Canonical values from earlier Vivado report. On-disk report overwritten by INT4-o8.",
                "source": "Archived Vivado .rpt (overwritten on disk)",
            })

    for label, (result, info) in vivado_results.items():
        wns = extract_vivado_wns(info.get("timing"))

        safe = label.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "")
        shutil.copy2(info["path"], archive_dir / f"{safe}.rpt")
        if info.get("timing") and info["timing"].exists():
            shutil.copy2(info["timing"], archive_dir / f"{safe}_timing.rpt")

        rows.append({
            "label": label, "status": "OK", **result,
            "clock": info["clock"], "wns": wns, "note": info["note"],
            "source": "Vivado .rpt",
        })
        print(f"  OK:  {label:<30} LUT={result.get('LUT','?'):>6}  FF={result.get('FF','?'):>6}  "
              f"BRAM_18K={result.get('BRAM_18K','?'):>4}  DSP={result.get('DSP','?'):>4}  "
              f"WNS={wns}")

    # ── FINN JSON reports ──
    print("\n=== FINN build reports ===")
    for label, info in FINN_BUILDS.items():
        result = extract_finn_resources(info["dir"])
        if result is None:
            print(f"  MISSING: {label}")
            rows.append({"label": label, "status": "MISSING", "note": info["note"],
                         "clock": info["clock"]})
            continue

        wns = extract_finn_wns(info["dir"])

        safe = label.lower().replace(" ", "_")
        src_json = info["dir"] / "report" / "post_synth_resources.json"
        shutil.copy2(src_json, archive_dir / f"{safe}_resources.json")
        src_timing = info["dir"] / "report" / "post_route_timing.rpt"
        if src_timing.exists():
            shutil.copy2(src_timing, archive_dir / f"{safe}_timing.rpt")

        rows.append({
            "label": label, "status": "OK", **result,
            "clock": info["clock"], "wns": wns, "note": info["note"],
            "source": "FINN JSON",
        })
        print(f"  OK:  {label:<30} LUT={result.get('LUT','?'):>6}  FF={result.get('FF','?'):>6}  "
              f"BRAM_18K={result.get('BRAM_18K','?'):>4}  DSP={result.get('DSP','?'):>4}  "
              f"WNS={wns}")

    # ── Generate markdown ──
    ok_rows = [r for r in rows if r["status"] == "OK"]
    missing_rows = [r for r in rows if r["status"] == "MISSING"]

    lines = []
    lines.append("# FPGA Resource Utilization - Post-Implementation")
    lines.append("")
    lines.append(f"ZU3EG budget: {BUDGET['LUT']:,} LUT | {BUDGET['FF']:,} FF | "
                 f"{BUDGET['BRAM_18K']} BRAM_18K | {BUDGET['DSP']} DSP")
    lines.append("")
    lines.append("| Design | Clock | LUT | LUT% | FF | FF% | BRAM_18K | BRAM% | DSP | DSP% | WNS (ns) | Source |")
    lines.append("|--------|-------|-----|------|----|-----|----------|-------|-----|------|----------|--------|")

    for r in ok_rows:
        lut = r.get("LUT", 0)
        ff = r.get("FF", 0)
        bram = r.get("BRAM_18K", 0)
        dsp = r.get("DSP", 0)
        wns = r.get("wns")
        wns_str = f"+{wns:.3f}" if wns is not None and wns >= 0 else (f"{wns:.3f}" if wns is not None else "-")
        lines.append(
            f"| {r['label']} "
            f"| {r['clock']} "
            f"| {lut:,} | {pct(lut, 'LUT')} "
            f"| {ff:,} | {pct(ff, 'FF')} "
            f"| {bram} | {pct(bram, 'BRAM_18K')} "
            f"| {dsp} | {pct(dsp, 'DSP')} "
            f"| {wns_str} "
            f"| {r.get('source', '?')} |"
        )

    if missing_rows:
        lines.append("")
        lines.append("### Missing")
        for r in missing_rows:
            lines.append(f"- **{r['label']}**: {r['note']}")

    lines.append("")
    lines.append("### Notes")
    lines.append("- **VTA INT8** on-disk report was overwritten by INT4-o8 build. Values shown are from the")
    lines.append("  pre-overwrite Vivado report. DSP/BRAM are structurally fixed by HLS IPs (identical across clocks).")
    lines.append("  LUT/FF may differ by a few percent at 250 MHz vs the original build clock.")
    lines.append("- FINN uses Ultra96 board definition (same ZU3EG die, different package, package irrelevant for resources).")
    lines.append("- BRAM_18K = RAMB36 x 2 + RAMB18 (Vivado) or BRAM_36K x 2 + BRAM_18K (FINN JSON).")
    lines.append("- All FINN builds at default target_fps. See finn_sweep_summary for resource data at higher folding.")
    lines.append("- WNS shown as '-' when timing report is missing or not archived.")

    md_path = REPO / "analysis" / "resource_utilization.md"
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {md_path}")
    print(f"Archived {len([r for r in rows if r['status']=='OK'])} reports to {archive_dir}/")


if __name__ == "__main__":
    main()
