"""
Compile an ONNX model for the KV260 using FINN.
Run inside the FINN Docker container.

Usage:
  python compile.py --model mlp_mnist_tiny.onnx --fps 1000
  python compile.py --model cnn_cifar10_tiny.onnx --fps 100
"""
import argparse
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to ONNX model')
parser.add_argument('--fps', type=int, default=1000, help='Target FPS')
parser.add_argument('--output', default=None, help='Output directory')
parser.add_argument('--board', default='Ultra96', help='Target board')
parser.add_argument('--fpga-part', default=None, help='Override FPGA part string (e.g. xczu3eg-sbva484-1-e)')
args = parser.parse_args()

output_dir = args.output or f"output_{args.model.replace('.onnx', '')}"

cfg = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    target_fps=args.fps,
    synth_clk_period_ns=10.0,
    board=args.board if args.fpga_part is None else None,
    fpga_part=args.fpga_part,
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)

build.build_dataflow_cfg(args.model, cfg)
print(f"--- Build complete: {output_dir} ---")
