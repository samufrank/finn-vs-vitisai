"""
Estimate FINN resource usage without full synthesis.
Run inside the FINN Docker container.

Usage:
  python estimate_resources.py --model mlp_mnist_tiny.onnx --fps 1000

NOTE: Pre-HLS estimates can significantly underestimate actual resource usage.
The MLP estimated 49 BRAM18 pre-HLS but required 255+ after HLS synthesis.
Use this for quick sanity checks, not final verification.
"""
import argparse
import json
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to ONNX model')
parser.add_argument('--fps', type=int, default=1000, help='Target FPS')
parser.add_argument('--output', default=None, help='Output directory')
args = parser.parse_args()

output_dir = args.output or f"estimate_{args.model.replace('.onnx', '')}"

cfg = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    target_fps=args.fps,
    synth_clk_period_ns=10.0,
    board="KV260_SOM",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ],
)

build.build_dataflow_cfg(args.model, cfg)

for report in ['estimate_layer_resources.json', 'estimate_network_performance.json']:
    path = f"{output_dir}/report/{report}"
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"\n=== {report} ===")
        print(json.dumps(data, indent=2))
    except:
        pass

print(f"\nKV260 limits: 288 BRAM18, 117120 LUTs, 1248 DSPs")
print("WARNING: These are pre-HLS estimates. Actual usage may be significantly higher.")
