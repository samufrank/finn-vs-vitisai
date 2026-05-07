"""
Compile ResNet-8 QONNX through finn-plus 1.4.0 — experimental.

Mainline FINN failed at step_create_dataflow_partition with:
    AssertionError: cycle-free graph violated: partition depends on itself
because residual fork-join produces a cyclic super-node when the partitioner
tries to bundle the residual block's nodes (Im2Col warnings on conv inputs
that come from the float-typed Add output were the precursor).

finn-plus 1.4.0 has additional steps that handle fork/join patterns for
transformer attention:
  - step_streamline:                       unified exhaustive streamliner
                                           with explicit residual support
  - step_convert_elementwise_binary_to_hw: makes residual Add HLS-able
                                           (InferElementwiseBinaryOperation)
  - step_replicate_streams:                handles the fork pattern (X feeds
                                           both branches of the residual)

This script tests whether those primitives let CNN residuals compile.
Stops at estimate reports per gate scope. Bitstream and OOC synthesis
deferred until the user authorizes.

Run from inside the finn-plus venv:
    source ~/.venvs/finn-t-env/bin/activate
    python finn/compile_resnet8_finnplus.py
"""
import os
import sys

# finn-plus must have global settings initialized when running outside
# its `finn` CLI entry point.
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

# finn-plus's build_steps.py was written against a slightly older finn API where
# execute_parent lived under finn.util.test; in current finn-plus it's at
# finn.util.execution. Alias before importing build_steps so its top-level
# import resolves. Touching only sys.modules — does not modify finn-t/.
from finn.util import execution as _finn_util_execution
sys.modules.setdefault("finn.util.test", _finn_util_execution)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

# Pull the residual-aware build steps from the FINN-T workspace.
sys.path.insert(0, os.path.expanduser("~/dev/CEN571-final/finn-t"))
from build_steps import (
    prepare_graph,
    step_streamline,
    step_convert_elementwise_binary_to_hw,
    step_replicate_streams,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX = os.path.join(REPO_ROOT, "finn", "resnet8_cifar10_int8.onnx")
OUTPUT_DIR = os.path.join(REPO_ROOT, "finn", "output_resnet8_cifar10_finnplus")

cfg = build_cfg.DataflowBuildConfig(
    output_dir=OUTPUT_DIR,
    target_fps=100,
    synth_clk_period_ns=10.0,
    board="Ultra96",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    save_intermediate_models=True,
    auto_fifo_depths=False,
    verbose=True,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ],
    steps=[
        # finn-plus's residual-aware QONNX prep + unified streamliner.
        prepare_graph(range_info=None),
        step_streamline,
        # Make residual Add HLS-able BEFORE generic convert_to_hw so the
        # downstream partitioner sees an HW node rather than a float Add.
        step_convert_elementwise_binary_to_hw,
        # Convert fork-input pattern to ReplicateStream nodes so the
        # partitioner can form acyclic super-nodes around the branches.
        step_replicate_streams,
        # Standard FINN tail through estimate reports.
        "step_convert_to_hw",
        "step_specialize_layers",
        "step_create_dataflow_partition",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
    ],
)

if __name__ == "__main__":
    print(f"Input ONNX:  {ONNX}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"Board:       Ultra96  (target_fps=100)")
    print(f"Stop after:  step_generate_estimate_reports")
    print("")

    rc = build.build_dataflow_cfg(ONNX, cfg)
    if rc == 0:
        print(f"--- Build complete: {OUTPUT_DIR} ---")
    else:
        print(f"--- Build FAILED (rc={rc}): {OUTPUT_DIR} ---")
    sys.exit(rc if rc == 0 else 1)
