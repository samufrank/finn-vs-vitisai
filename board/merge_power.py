#!/usr/bin/env python3
"""
Merge FNB58 power log with benchmark results.

Usage:
  python3 merge_power.py --benchmark results.json --power power_log.csv

Extracts power samples matching the precise UNIX timestamps recorded
by benchmark.py for idle and each inference run, computes average power,
dynamic power, and energy per image.

Clock sync note: the board and host clocks must be reasonably aligned.
If the board's clock is off (common on PetaLinux without NTP), use
--clock-offset to compensate. Positive = board clock is behind host.
"""

import json
import csv
import argparse
import numpy as np
import os


def load_power_log(csv_path):
    """Load FNB58 power CSV into list of (timestamp, voltage, current, power, temp)."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((
                float(row['timestamp']),
                float(row['voltage_v']),
                float(row['current_a']),
                float(row['power_w']),
                float(row['temp_c']),
            ))
    return samples


def extract_window(samples, t_start, t_end):
    """Extract power samples within [t_start, t_end]."""
    return [s for s in samples if t_start <= s[0] <= t_end]


def window_stats(window):
    """Compute stats for a power window. Returns dict or None."""
    if not window:
        return None
    powers = [s[3] for s in window]
    voltages = [s[1] for s in window]
    currents = [s[2] for s in window]
    return {
        'power_w_mean': float(np.mean(powers)),
        'power_w_std': float(np.std(powers)),
        'power_w_min': float(np.min(powers)),
        'power_w_max': float(np.max(powers)),
        'voltage_v_mean': float(np.mean(voltages)),
        'current_a_mean': float(np.mean(currents)),
        'n_samples': len(window),
    }


def main():
    parser = argparse.ArgumentParser(description='Merge FNB58 power data with benchmark results')
    parser.add_argument('--benchmark', required=True, help='Benchmark JSON file from board')
    parser.add_argument('--power', required=True, help='FNB58 power CSV from host')
    parser.add_argument('--output', default=None, help='Output JSON (default: overwrite benchmark file)')
    parser.add_argument('--plot', default=None, nargs='?', const='auto',
                        help='Generate power timeline PNG. Optionally specify output path (default: auto from output name)')
    parser.add_argument('--clock-offset', type=float, default=0.0,
                        help='Seconds to add to board timestamps to align with host clock. '
                             'Positive if board clock is behind host.')
    args = parser.parse_args()

    with open(args.benchmark, 'r') as f:
        bench = json.load(f)

    power_samples = load_power_log(args.power)
    if not power_samples:
        print("ERROR: No power samples found in CSV.")
        return

    t_pow_start = power_samples[0][0]
    t_pow_end = power_samples[-1][0]
    print(f"Power log: {len(power_samples)} samples, {t_pow_end - t_pow_start:.1f}s duration")

    runs = bench.get('runs', [])
    idle = bench.get('idle', {})
    offset = args.clock_offset

    # Idle power from FNB58
    idle_t_start = idle.get('t_start')
    idle_t_end = idle.get('t_end')
    idle_stats = None
    if idle_t_start and idle_t_end:
        idle_window = extract_window(power_samples, idle_t_start + offset, idle_t_end + offset)
        idle_stats = window_stats(idle_window)
        if idle_stats:
            print(f"Idle:  {idle_stats['power_w_mean']:.3f} +/- {idle_stats['power_w_std']:.3f} W  "
                  f"({idle_stats['n_samples']} samples, "
                  f"{idle_stats['voltage_v_mean']:.2f}V / {idle_stats['current_a_mean']:.3f}A)")
        else:
            print(f"WARNING: No power samples in idle window "
                  f"[{idle_t_start + offset:.1f}, {idle_t_end + offset:.1f}]")
            print(f"  Power log range: [{t_pow_start:.1f}, {t_pow_end:.1f}]")
            print(f"  Check --clock-offset (board vs host clock drift)")
    else:
        print("WARNING: No idle timestamps in benchmark JSON. Update benchmark.py.")

    # Per-run power
    run_stats_list = []
    num_images = bench.get('config', {}).get('num_images', 10000)

    for r in runs:
        t_start = r.get('t_start')
        t_end = r.get('t_end')
        if t_start is None or t_end is None:
            print(f"  Run {r['run']}: no timestamps, skipping")
            run_stats_list.append(None)
            continue

        run_window = extract_window(power_samples, t_start + offset, t_end + offset)
        rs = window_stats(run_window)
        if rs:
            dynamic = rs['power_w_mean'] - idle_stats['power_w_mean'] if idle_stats else None
            energy_j = rs['power_w_mean'] * r['time_s']
            energy_per_img_mj = 1000 * energy_j / num_images

            rs['dynamic_power_w'] = dynamic
            rs['energy_total_j'] = energy_j
            rs['energy_per_image_mj'] = energy_per_img_mj

            dyn_str = f"dynamic: {dynamic:.3f} W, " if dynamic is not None else ""
            print(f"  Run {r['run']}: {rs['power_w_mean']:.3f} +/- {rs['power_w_std']:.3f} W  "
                  f"({dyn_str}{rs['n_samples']} samples)")
        else:
            print(f"  Run {r['run']}: WARNING no power samples in window "
                  f"[{t_start + offset:.1f}, {t_end + offset:.1f}]")
            print(f"    Power log range: [{t_pow_start:.1f}, {t_pow_end:.1f}]")
        run_stats_list.append(rs)

    # Update benchmark JSON
    bench['power_measurement'] = {
        'method': 'fnb58_external',
        'power_log': os.path.basename(args.power),
        'clock_offset_applied': offset,
        'total_power_samples': len(power_samples),
        'idle': idle_stats,
    }

    # Update each run
    for r, rs in zip(runs, run_stats_list):
        if rs:
            r['fnb58_power'] = rs
            r['avg_power_w'] = rs['power_w_mean']
            r['energy_total_j'] = rs['energy_total_j']
            r['energy_per_image_mj'] = rs['energy_per_image_mj']

    # Update summary
    powered_runs = [rs for rs in run_stats_list if rs is not None]
    if powered_runs and 'summary' in bench:
        s = bench['summary']
        s['avg_power_w_mean'] = float(np.mean([rs['power_w_mean'] for rs in powered_runs]))
        s['avg_power_w_std'] = float(np.std([rs['power_w_mean'] for rs in powered_runs]))
        if idle_stats:
            s['idle_power_w'] = idle_stats['power_w_mean']
            s['idle_power_std'] = idle_stats['power_w_std']
            s['dynamic_power_w'] = s['avg_power_w_mean'] - idle_stats['power_w_mean']
        s['energy_per_image_mj_mean'] = float(np.mean([rs['energy_per_image_mj'] for rs in powered_runs]))
        s['energy_per_image_mj_std'] = float(np.std([rs['energy_per_image_mj'] for rs in powered_runs]))

        print(f"\nSummary:")
        if s.get('idle_power_w'):
            print(f"  Idle power:    {s['idle_power_w']:.3f} W")
        print(f"  Active power:  {s['avg_power_w_mean']:.3f} +/- {s['avg_power_w_std']:.3f} W")
        if s.get('dynamic_power_w') is not None:
            print(f"  Dynamic power: {s['dynamic_power_w']:.3f} W")
        print(f"  Energy/image:  {s['energy_per_image_mj_mean']:.4f} +/- {s['energy_per_image_mj_std']:.4f} mJ")

    output_path = args.output or args.benchmark
    with open(output_path, 'w') as f:
        json.dump(bench, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # ---- Optional power timeline plot ----
    if args.plot is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            if args.plot == 'auto':
                plot_path = output_path.rsplit('.', 1)[0] + '_power.png'
            else:
                plot_path = args.plot

            # Convert to relative time from first sample
            t0 = power_samples[0][0]
            times = np.array([s[0] - t0 for s in power_samples])
            powers = np.array([s[3] for s in power_samples])

            # Rolling average (1s window at ~100Hz)
            window = min(100, len(powers) // 4)
            if window > 1:
                kernel = np.ones(window) / window
                powers_smooth = np.convolve(powers, kernel, mode='same')
            else:
                powers_smooth = powers

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(times, powers, color='#2563eb', linewidth=0.3, alpha=0.2)
            ax.plot(times, powers_smooth, color='#2563eb', linewidth=1.5, alpha=0.9, label='Power (1s avg)')

            # Shade idle window
            if idle_t_start and idle_t_end:
                ax.axvspan(idle_t_start + offset - t0, idle_t_end + offset - t0,
                           alpha=0.15, color='#6b7280', label='Idle')

            # Shade each run
            colors = ['#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2']
            for i, r in enumerate(runs):
                ts = r.get('t_start')
                te = r.get('t_end')
                if ts and te:
                    ax.axvspan(ts + offset - t0, te + offset - t0,
                               alpha=0.15, color=colors[i % len(colors)],
                               label=f'Run {i+1}')

            # Idle power reference line
            if idle_stats:
                ax.axhline(y=idle_stats['power_w_mean'], color='#6b7280',
                           linestyle='--', linewidth=1, alpha=0.5, label=f"Idle ({idle_stats['power_w_mean']:.2f}W)")

            # Active power reference line
            if powered_runs:
                active_mean = float(np.mean([rs['power_w_mean'] for rs in powered_runs]))
                ax.axhline(y=active_mean, color='#dc2626',
                           linestyle='--', linewidth=1, alpha=0.5, label=f"Active ({active_mean:.2f}W)")

            toolchain = bench.get('config', {}).get('toolchain', '?')
            model = bench.get('config', {}).get('model_dir',
                    bench.get('config', {}).get('deploy_dir',
                    bench.get('config', {}).get('model_path', '?')))
            model = os.path.basename(str(model).rstrip('/'))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power (W)')
            ax.set_title(f'Power Timeline: {toolchain} / {model}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Plot saved to: {plot_path}")

        except ImportError:
            print("WARNING: matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
