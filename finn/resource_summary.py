#!/usr/bin/env python3
"""Summarize a FINN build for the apples-to-apples comparison.

Run on the host (not inside FINN docker — uses onnx + stdlib only).

Reads from a FINN build's output_<name>/ folder:
  report/post_synth_resources.json    (Vivado post-synth utilization)
  report/estimate_network_performance.json  (estimated FPS / latency)
  report/post_route_timing.rpt        (Vivado timing — WNS for Fmax)
  final_hw_config.json                (PE/SIMD chosen by the auto-folder)
  intermediate_models/dataflow_parent.onnx  (CPU vs FPGA partition)
  deploy/bitfile/*.bit                (bitstream size)

Writes:
  output_<name>/resource_report.json  structured archival
  output_<name>/resource_summary.md   markdown table fragment
  stdout one-liner

Usage:
  python finn/resource_summary.py --output-dir output_cnn_mnist_tiny_int4

ZU3EG resource totals (matches finn/estimate_resources.py):
  LUT=70560, BRAM_18K=432, DSP=360
"""
import argparse
import json
import os
import re
import sys

ZU3EG = {'LUT': 70560, 'BRAM_18K': 432, 'DSP': 360}


def parse_wns(rpt_path):
    """Extract Worst Negative Slack from Vivado timing report.

    Looks for the 'Setup' summary line, e.g.:
      Setup :    0  Failing Endpoints,  Worst Slack    4.006ns,  ...
    Returns the WNS in ns (positive = timing met, negative = timing missed),
    or None if not found.
    """
    if not os.path.isfile(rpt_path):
        return None
    pat = re.compile(r'Setup\s*:.*Worst Slack\s+(-?[0-9.]+)ns')
    with open(rpt_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                return float(m.group(1))
    return None


def total_bram_18k(post_synth_top):
    """Convert BRAM_36K + BRAM_18K to BRAM_18K-equivalent total.
    Each 36K = 2 18K. ZU3EG has 432 18K-equivalent total."""
    b18 = post_synth_top.get('BRAM_18K', 0)
    b36 = post_synth_top.get('BRAM_36K', 0)
    return b18 + 2 * b36


def cpu_partition_layers(dataflow_parent_path):
    """Return [(op_type, name), ...] of nodes outside StreamingDataflowPartition.

    These are the CPU-side layers that the C runner has to execute. Returns
    None if onnx isn't available or the file doesn't exist.
    """
    if not os.path.isfile(dataflow_parent_path):
        return None
    try:
        import onnx
    except ImportError:
        return 'onnx-not-installed'
    m = onnx.load(dataflow_parent_path)
    return [(n.op_type, n.name) for n in m.graph.node
            if n.op_type != 'StreamingDataflowPartition']


def folding_per_layer(final_hw_config_path):
    """Extract PE/SIMD per layer from final_hw_config.json."""
    if not os.path.isfile(final_hw_config_path):
        return None
    with open(final_hw_config_path) as f:
        cfg = json.load(f)
    out = {}
    for layer, params in cfg.items():
        if not isinstance(params, dict):
            continue
        pe = params.get('PE')
        simd = params.get('SIMD')
        if pe is not None or simd is not None:
            out[layer] = {'PE': pe, 'SIMD': simd}
    return out


def find_bitstream(deploy_dir):
    """Return (path, bytes) for the bitstream, or (None, 0)."""
    bit_dir = os.path.join(deploy_dir, 'bitfile')
    if not os.path.isdir(bit_dir):
        return None, 0
    for f in os.listdir(bit_dir):
        if f.endswith('.bit'):
            p = os.path.join(bit_dir, f)
            return p, os.path.getsize(p)
    return None, 0


def summarize(output_dir):
    """Build the structured resource report dict."""
    rep_dir = os.path.join(output_dir, 'report')

    # Post-synth Vivado utilization
    psr_path = os.path.join(rep_dir, 'post_synth_resources.json')
    if os.path.isfile(psr_path):
        with open(psr_path) as f:
            psr = json.load(f)
        top = psr.get('(top)', {})
        lut = top.get('LUT', 0)
        ff = top.get('FF', 0)
        dsp = top.get('DSP', 0)
        bram = total_bram_18k(top)
        uram = top.get('URAM', 0)
        srl = top.get('SRL', 0)
    else:
        psr = None
        lut = ff = dsp = bram = uram = srl = None

    # Estimated network performance
    perf_path = os.path.join(rep_dir, 'estimate_network_performance.json')
    perf = None
    if os.path.isfile(perf_path):
        with open(perf_path) as f:
            perf = json.load(f)

    # Vivado timing (WNS for Fmax)
    rpt_path = os.path.join(rep_dir, 'post_route_timing.rpt')
    wns_ns = parse_wns(rpt_path)
    period_ns = 10.0  # compile.py uses synth_clk_period_ns=10.0
    if wns_ns is not None:
        fmax_mhz = 1000.0 / (period_ns - wns_ns) if wns_ns < period_ns else None
    else:
        fmax_mhz = None

    # Folding (PE / SIMD per layer)
    folding = folding_per_layer(os.path.join(output_dir, 'final_hw_config.json'))

    # CPU partition layers
    cpu_layers = cpu_partition_layers(
        os.path.join(output_dir, 'intermediate_models', 'dataflow_parent.onnx'))

    # Bitstream
    bit_path, bit_bytes = find_bitstream(os.path.join(output_dir, 'deploy'))

    report = {
        'output_dir': os.path.abspath(output_dir),
        'utilization': {
            'LUT': lut,
            'LUT_pct': round(100 * lut / ZU3EG['LUT'], 2) if lut is not None else None,
            'FF': ff,
            'SRL': srl,
            'BRAM_18K_equiv': bram,
            'BRAM_18K_pct': round(100 * bram / ZU3EG['BRAM_18K'], 2) if bram is not None else None,
            'DSP': dsp,
            'DSP_pct': round(100 * dsp / ZU3EG['DSP'], 2) if dsp is not None else None,
            'URAM': uram,
        },
        'timing': {
            'synth_period_ns': period_ns,
            'wns_ns': wns_ns,
            'fmax_mhz': round(fmax_mhz, 2) if fmax_mhz is not None else None,
        },
        'estimate_performance': perf,
        'folding': folding,
        'cpu_partition_layers': cpu_layers,
        'bitfile': {
            'path': bit_path,
            'bytes': bit_bytes,
            'mb': round(bit_bytes / 1024 / 1024, 2) if bit_bytes else 0,
        },
    }
    return report


def render_markdown(report):
    """Markdown table fragment ready to paste into the report."""
    u = report['utilization']
    t = report['timing']
    p = report['estimate_performance'] or {}
    folding = report['folding'] or {}
    cpu = report['cpu_partition_layers'] or []

    lines = []
    lines.append('## ' + os.path.basename(report['output_dir']))
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---|')
    lines.append(f"| LUT | {u['LUT']} ({u['LUT_pct']}%) |" if u['LUT'] is not None else '| LUT | — |')
    lines.append(f"| FF | {u['FF']} |" if u['FF'] is not None else '| FF | — |')
    lines.append(f"| BRAM (18K-equiv) | {u['BRAM_18K_equiv']} ({u['BRAM_18K_pct']}%) |"
                 if u['BRAM_18K_equiv'] is not None else '| BRAM (18K-equiv) | — |')
    lines.append(f"| DSP | {u['DSP']} ({u['DSP_pct']}%) |" if u['DSP'] is not None else '| DSP | — |')
    lines.append(f"| Fmax | {t['fmax_mhz']} MHz (WNS={t['wns_ns']} ns) |"
                 if t['fmax_mhz'] is not None else '| Fmax | — |')
    if 'estimated_throughput_fps' in p:
        lines.append(f"| Est FPS (FPGA only) | {round(p['estimated_throughput_fps'], 1)} |")
        lines.append(f"| Est latency | {round(p['estimated_latency_ns'] / 1000, 2)} µs |")
    lines.append(f"| Bitstream | {report['bitfile']['mb']} MB |")
    lines.append('')
    if folding:
        lines.append('### Folding (PE/SIMD)')
        lines.append('| Layer | PE | SIMD |')
        lines.append('|---|---|---|')
        for layer, vals in folding.items():
            lines.append(f"| {layer} | {vals.get('PE', '—')} | {vals.get('SIMD', '—')} |")
        lines.append('')
    if cpu and cpu != 'onnx-not-installed':
        lines.append('### CPU-partitioned layers (run outside FPGA)')
        for op, name in cpu:
            lines.append(f"- `{op}: {name}`")
        lines.append('')
    return '\n'.join(lines)


def render_one_liner(report):
    u = report['utilization']
    t = report['timing']
    folding = report['folding'] or {}
    cpu = report['cpu_partition_layers'] or []

    name = os.path.basename(report['output_dir'])
    parts = [name]
    if u['LUT'] is not None:
        parts.append(f"{u['LUT']} LUT ({u['LUT_pct']}%)")
    if u['BRAM_18K_equiv'] is not None:
        parts.append(f"{u['BRAM_18K_equiv']} BRAM18 ({u['BRAM_18K_pct']}%)")
    if u['DSP'] is not None:
        parts.append(f"{u['DSP']} DSP ({u['DSP_pct']}%)")
    if t['fmax_mhz']:
        parts.append(f"Fmax={t['fmax_mhz']}MHz")
    if report['bitfile']['mb']:
        parts.append(f"{report['bitfile']['mb']}MB bit")
    # Compact folding string
    folding_str = ', '.join(
        f"{k.split('_')[0]}={v.get('PE','-')}/{v.get('SIMD','-')}"
        for k, v in folding.items() if 'MVAU' in k or 'MatrixVector' in k
    )
    if folding_str:
        parts.append(f"folding {folding_str}")
    if cpu and cpu != 'onnx-not-installed':
        parts.append(f"CPU: {len(cpu)} layers ({', '.join(op for op, _ in cpu[:5])}{'...' if len(cpu) > 5 else ''})")
    return ' | '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True,
                    help='FINN build output_<name>/ directory')
    ap.add_argument('--quiet', action='store_true',
                    help='Suppress one-liner stdout (for batch mode).')
    args = ap.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"error: {args.output_dir} not found", file=sys.stderr)
        return 1

    report = summarize(args.output_dir)

    json_path = os.path.join(args.output_dir, 'resource_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(args.output_dir, 'resource_summary.md')
    with open(md_path, 'w') as f:
        f.write(render_markdown(report))

    if not args.quiet:
        print(render_one_liner(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
