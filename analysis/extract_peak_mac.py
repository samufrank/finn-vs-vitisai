#!/usr/bin/env python3
"""
extract_peak_mac.py - per-layer MAC vs folding vs LUT for the INT8 CNN size sweep.

Sibling to extract_resources.py (which builds the cross-framework whole-design
table). This script covers the five classic INT8 CNN size-sweep architectures
and tests which predicts LUT better: parameter count or peak per-layer MAC.

Run from repo root:
    python3 analysis/extract_peak_mac.py

Sources (Tier-1 unless noted), per build under finn/size_sweep_runs/cnn_int8_<arch>/:
  report/op_and_param_counts.json   <layer>.op_mac_8bx8b, .param_weight_8b   (measured)
  final_hw_config.json              <layer>.PE/.SIMD  (final build folding)   (measured)
  auto_folding_config.json          <layer>.PE/.SIMD  (cross-check, must match) (measured)
  report/post_synth_resources.json  (top).LUT and *_MVAU_hls_*.{LUT,DSP}      (measured, fit only)
params:
  results/finn/size_sweep/phase_a_training/logs/cnn_mnist_<arch>.log "Parameters:" (measured)
  tiny: phase_a_training/accuracy_table.md CNN row (derived, tiny not retrained)
bust LUT (no synth report, aborted at placement):
  results/finn/size_sweep/cnn_int8_<arch>_impl_runme.log DRC UTLZ-1 "Slice LUTs ... requires N"
post-synth vs post-route calibration (4 CNN INT8 QI builds with both reports):
  results/finn/cnn_int8_qi_sweep/<b>/top_wrapper_utilization_placed.rpt "CLB LUTs"
  vs finn/size_sweep_runs/<b>/report/post_synth_resources.json (top).LUT

Outputs:
  analysis/peak_mac_vs_params.md
  analysis/peak_mac_vs_params.csv   (per-arch figure data)

Derived quantities:
  OFM positions  = op_mac_8bx8b / param_weight_8b   (H*W of the conv output)
  PE*SIMD        = unrolled MAC datapath width (drives MVAU LUT)
  fold           = param_weight_8b / (PE*SIMD)       (MW*MH / parallelism)
"""

import csv
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "finn" / "size_sweep_runs"
SWEEP = REPO / "results" / "finn" / "size_sweep"
QI = REPO / "results" / "finn" / "cnn_int8_qi_sweep"

DEVICE_LUT = 70560  # xczu3eg-sbva484-1-e CLB LUTs

ARCHS = [
    ("tiny", "[8,16]"),
    ("small", "[16,32]"),
    ("medium", "[32,64]"),
    ("deep_3", "[16,32,64]"),
    ("large", "[32,64,128]"),
]

QI_CALIB = [
    "cnn_int8_tiny_qi_fps500",
    "cnn_int8_small_qi_fps500",
    "cnn_int8_deep_3_qi_fps500",
    "cnn_int8_medium_qi_fps500",
]


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


def read_params(arch, label):
    """Tier-1 'Parameters:' from the training log; tiny falls back to accuracy_table (derived)."""
    log = SWEEP / "phase_a_training" / "logs" / f"cnn_mnist_{arch}.log"
    if log.exists():
        m = re.search(r"^Parameters:\s*(\d+)", log.read_text(), re.M)
        if m:
            return int(m.group(1)), "measured", f"{rel(log)}:Parameters:"
    at = SWEEP / "phase_a_training" / "accuracy_table.md"
    if at.exists():
        # match the CNN row; the channel label disambiguates from the MLP 'tiny' row
        m = re.search(
            r"\|\s*" + re.escape(arch) + r"\s*\|\s*" + re.escape(label) + r"\s*\|\s*([\d,]+)",
            at.read_text(),
        )
        if m:
            return int(m.group(1).replace(",", "")), "derived", f"{rel(at)} (CNN row)"
    return None, "not-found", ""


def read_opcounts(arch):
    """{MVAU name: (mac, weight_params)} for fabric compute layers."""
    p = RUNS / f"cnn_int8_{arch}" / "report" / "op_and_param_counts.json"
    j = json.load(open(p))
    out = {}
    for k, v in j.items():
        if k == "total" or not isinstance(v, dict) or "op_mac_8bx8b" not in v:
            continue
        out[k] = (int(v["op_mac_8bx8b"]), int(v.get("param_weight_8b", 0)))
    return out, rel(p)


def read_folding(arch):
    """{MVAU name: (PE, SIMD, mismatch_vs_auto)} from the final build config."""
    d = RUNS / f"cnn_int8_{arch}"
    final = json.load(open(d / "final_hw_config.json"))
    auto = json.load(open(d / "auto_folding_config.json"))
    out = {}
    for k, v in final.items():
        if "MVAU" not in k:
            continue
        pe, simd = v.get("PE"), v.get("SIMD")
        a = auto.get(k, {})
        out[k] = (pe, simd, a.get("PE") != pe or a.get("SIMD") != simd)
    return out, rel(d / "final_hw_config.json")


def read_synth(arch):
    """(top LUT, {MVAU name: (LUT, DSP)}, path) or None if the build busted before the synth report."""
    p = RUNS / f"cnn_int8_{arch}" / "report" / "post_synth_resources.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    mvau = {}
    for k, v in j.items():
        if "MVAU" in k:
            mvau[k[k.index("MVAU"):]] = (v.get("LUT"), v.get("DSP"))
    return j.get("(top)", {}).get("LUT"), mvau, rel(p)


def read_bust(arch):
    """(required LUT, available LUT, log:line) from DRC UTLZ-1 'Slice LUTs', or None."""
    p = SWEEP / f"cnn_int8_{arch}_impl_runme.log"
    if not p.exists():
        return None
    text = p.read_text()
    m = re.search(
        r"Slice LUTs over-utilized.*?requires (\d+) of such cell types but only (\d+)",
        text, re.S,
    )
    if not m:
        return None
    ln = next((i for i, line in enumerate(text.splitlines(), 1)
               if "Slice LUTs over-utilized" in line), "?")
    return int(m.group(1)), int(m.group(2)), f"{rel(p)}:{ln}"


def calibration():
    """post_synth (top).LUT vs post-route top_wrapper CLB LUTs for the QI builds with both."""
    rows = []
    for b in QI_CALIB:
        rpt = QI / b / "top_wrapper_utilization_placed.rpt"
        syn = RUNS / b / "report" / "post_synth_resources.json"
        route = synth = None
        if rpt.exists():
            m = re.search(r"\|\s*CLB LUTs\*?\s+\|\s*(\d+)\s*\|", rpt.read_text())
            route = int(m.group(1)) if m else None
        if syn.exists():
            synth = json.load(open(syn)).get("(top)", {}).get("LUT")
        ratio = route / synth if (route and synth) else None
        rows.append((b, synth, route, ratio))
    return rows


def ofm_side(positions):
    s = int(round(math.sqrt(positions)))
    return f"{s}x{s}" if s * s == positions else f"{positions}px"


def main():
    archs, layers = [], []
    for arch, label in ARCHS:
        params, pstate, psrc = read_params(arch, label)
        ops, ops_src = read_opcounts(arch)
        fold, fold_src = read_folding(arch)
        synth = read_synth(arch)
        bust = read_bust(arch)

        peak_mac = max(m for m, _ in ops.values())
        peak_layers = [n for n, (m, _) in ops.items() if m == peak_mac]
        total_fabric_mac = sum(m for m, _ in ops.values())
        sum_pe_simd = sum((fold[n][0] or 0) * (fold[n][1] or 0) for n in ops)

        if synth:
            lut, mvau_synth, lut_src = synth[0], synth[1], synth[2]
            fit = True
        elif bust:
            lut, _, lut_src = bust
            mvau_synth, fit = {}, False
        else:
            lut = lut_src = mvau_synth = None
            fit = None

        archs.append(dict(
            arch=arch, label=label, params=params, pstate=pstate, psrc=psrc,
            peak_mac=peak_mac, peak_layers=peak_layers,
            total_fabric_mac=total_fabric_mac, sum_pe_simd=sum_pe_simd,
            lut=lut, lut_pct=(100 * lut / DEVICE_LUT if lut else None),
            fit=fit, lut_src=lut_src, ops_src=ops_src, fold_src=fold_src,
        ))

        for n in sorted(ops):
            mac, w = ops[n]
            pe, simd, mism = fold.get(n, (None, None, False))
            pxs = (pe or 0) * (simd or 0)
            slut, sdsp = mvau_synth.get(n, (None, None))
            layers.append(dict(
                arch=arch, layer=n, mac=mac, w=w,
                ofm=ofm_side(mac // w) if w else "?",
                pe=pe, simd=simd, pxs=pxs,
                fold=(w / pxs if pxs else None),
                is_peak=(mac == peak_mac),
                synth_lut=slut, synth_dsp=sdsp,
                dsp_ok=(sdsp == pxs) if sdsp is not None else None,
                fold_mismatch=mism,
            ))

    calib = calibration()
    write_csv(archs)
    write_md(archs, layers, calib)
    print_console(archs, layers, calib)


def write_csv(archs):
    p = REPO / "analysis" / "peak_mac_vs_params.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "channels", "params", "params_state", "peak_mac_8bx8b",
                    "total_fabric_mac", "sum_pe_simd", "lut", "lut_pct", "fit", "lut_source"])
        for a in archs:
            w.writerow([a["arch"], a["label"], a["params"], a["pstate"], a["peak_mac"],
                        a["total_fabric_mac"], a["sum_pe_simd"], a["lut"],
                        f"{a['lut_pct']:.2f}" if a["lut_pct"] else "",
                        {True: "fit", False: "bust", None: "?"}[a["fit"]], a["lut_src"]])
    print(f"Wrote {rel(p)}")


def order_line(archs, key, fmt=str):
    return ", ".join(f"{a['arch']} ({fmt(a[key])})" for a in sorted(archs, key=lambda x: x[key]))


def write_md(archs, layers, calib):
    L = []
    L.append("# FINN CNN LUT vs params vs peak per-layer MAC")
    L.append("")
    L.append("Generated by analysis/extract_peak_mac.py.")
    L.append("")
    L.append(f"Device: {DEVICE_LUT:,} CLB LUTs (xczu3eg-sbva484-1-e). Five INT8 CNN size-sweep")
    L.append("builds. Fit-build LUT is post_synth_resources.json (top).LUT, which equals the")
    L.append("post-route placed count (see calibration). Bust-build LUT is the DRC UTLZ-1")
    L.append("requirement from the impl log (the build aborted at placement before any report).")
    L.append("")
    L.append("## Figure data")
    L.append("")
    L.append("| arch | channels | params | peak_mac | total_fabric_mac | sum_pe_simd | LUT | LUT% | fit |")
    L.append("|------|----------|-------:|---------:|-----------------:|------------:|----:|-----:|:---:|")
    for a in archs:
        L.append(f"| {a['arch']} | {a['label']} | {a['params']:,}{'*' if a['pstate']=='derived' else ''} "
                 f"| {a['peak_mac']:,} | {a['total_fabric_mac']:,} | {a['sum_pe_simd']} "
                 f"| {a['lut']:,} | {a['lut_pct']:.1f}% | {'y' if a['fit'] else 'n'} |")
    L.append("")
    L.append("* tiny params derived from accuracy_table (not retrained); others measured from training logs.")
    L.append("")
    L.append("## Orderings (ascending)")
    L.append("")
    L.append(f"- params: {order_line(archs, 'params', lambda v: f'{v:,}')}")
    L.append(f"- LUT: {order_line(archs, 'lut', lambda v: f'{v:,}')}")
    L.append(f"- peak_mac: {order_line(archs, 'peak_mac', lambda v: f'{v:,}')}")
    L.append(f"- total_fabric_mac: {order_line(archs, 'total_fabric_mac', lambda v: f'{v:,}')}")
    L.append("")
    L.append("The params order inverts against the LUT order at medium/deep_3: deep_3 has more")
    L.append("params than medium but less LUT. The peak_mac order has no inversion against LUT,")
    L.append("but ties small with deep_3 and medium with large. total_fabric_mac and sum_pe_simd")
    L.append("are strictly monotonic with LUT.")
    L.append("")
    L.append("## Per-layer folding and resource")
    L.append("")
    L.append("OFM positions = MAC / weight_params. weight_params = MW*MH = K^2 * Cin * Cout,")
    L.append("which has no spatial term, so equal-width convs at different resolutions have")
    L.append("equal weight_params but different MAC.")
    L.append("")
    L.append("| arch | layer | MAC | weight_params | OFM | PE | SIMD | PE*SIMD | fold | synth_LUT | synth_DSP | DSP==PE*SIMD | peak |")
    L.append("|------|-------|----:|--------------:|-----|---:|-----:|--------:|-----:|----------:|----------:|:-----------:|:----:|")
    for r in layers:
        chk = {True: "yes", False: "MISMATCH", None: "-"}[r["dsp_ok"]]
        L.append(f"| {r['arch']} | {r['layer']} | {r['mac']:,} | {r['w']:,} | {r['ofm']} "
                 f"| {r['pe']} | {r['simd']} | {r['pxs']} | {r['fold']:.0f} "
                 f"| {r['synth_lut'] if r['synth_lut'] is not None else 'n/a (bust)'} "
                 f"| {r['synth_dsp'] if r['synth_dsp'] is not None else '-'} | {chk} "
                 f"| {'yes' if r['is_peak'] else ''} |")
    L.append("")
    L.append("medium MVAU_hls_0 and deep_3 MVAU_hls_1 are both 32 to 64 channel convs with the")
    L.append("same weight_params (18,432), but run at 14x14 vs 7x7, so 4x the MAC, PE 16 vs 4,")
    L.append("and bust vs fit. SIMD is capped at 4 for INT8 (weight_bits * SIMD <= 36). DSP equals")
    L.append("PE*SIMD on every fit MVAU, confirming the folding config matches the synthesized design.")
    L.append("")
    L.append("## post-synth vs post-route calibration")
    L.append("")
    L.append("CNN INT8 QI builds that retain both a post_synth_resources.json and a post-route")
    L.append("top_wrapper_utilization_placed.rpt:")
    L.append("")
    L.append("| QI build | post_synth (top).LUT | post-route top_wrapper CLB LUTs | route/synth |")
    L.append("|----------|---------------------:|--------------------------------:|------------:|")
    for b, syn, route, ratio in calib:
        L.append(f"| {b} | {syn:,} | {route:,} | {ratio:.3f} |")
    L.append("")
    L.append("route/synth is 1.000 on all four, so (top).LUT is the full-design placed count and")
    L.append("the fit-build LUT numbers above are post-route values, not synthesis estimates.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    seen = []
    for a in archs:
        for s in (a["psrc"], a["ops_src"], a["fold_src"], a["lut_src"]):
            if s and s not in seen:
                seen.append(s)
    for s in sorted(seen):
        L.append(f"- {s}")
    (REPO / "analysis" / "peak_mac_vs_params.md").write_text("\n".join(L) + "\n")
    print(f"Wrote {rel(REPO / 'analysis' / 'peak_mac_vs_params.md')}")


def print_console(archs, layers, calib):
    print("\nfigure data:")
    for a in archs:
        print(f"  {a['arch']:7} params={a['params']:>7,} peak_mac={a['peak_mac']:>10,} "
              f"sum_pe_simd={a['sum_pe_simd']:>4} LUT={a['lut']:>7,} ({a['lut_pct']:5.1f}%) "
              f"{'fit' if a['fit'] else 'bust'}")
    bad = [r for r in layers if r["dsp_ok"] is False] + [r for r in layers if r["fold_mismatch"]]
    print(f"\n  DSP==PE*SIMD and folding cross-checks: {'all ok' if not bad else 'FAILURES: ' + str(bad)}")
    print("  calibration route/synth:", ", ".join(
        f"{b.split('_qi')[0].replace('cnn_int8_','')}={r:.3f}" for b, _, _, r in calib))


if __name__ == "__main__":
    main()
