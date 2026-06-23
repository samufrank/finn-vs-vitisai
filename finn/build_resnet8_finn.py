"""
Parallel-track driver: matched-INT8 ResNet-8 through MAINLINE FINN @ 8ac41e46.

Modes (run inside finn/run_finn_docker.sh):
  --mode export   : build ResNet8_Brevitas_FINN, calibrate activation scales on random
                    batches (>collect_stats_steps), export QONNX. Accuracy irrelevant.
  --mode inspect --onnx <p> : report op-counts, residual-add operand datatypes/scales,
                    AddStreams/DuplicateStreams counts. Works on QONNX or FINN-ONNX.
  --mode compile  : run a CUSTOM step list (adapted from finn-examples ResNet-50
                    step_resnet50_*, mainline transforms only) to step_create_dataflow_
                    partition and STOP. enable_build_pdb_debug=False.

Custom flow assembles EXISTING mainline transforms (no new transforms written):
  streamline: 4x[linear streamline + MoveLinearPastEltwiseAdd + MoveLinearPastFork],
              then LowerConvsToMatMul + transpose cleanup;
  convert_to_hw: ...QuantizedMVAU, Thresholding, ConvInpGen, DuplicateStreams(fork),
              AddStreams(join), LabelSelect, pooling.
"""
import argparse
import collections
import os
import sys

PROJ = "/workspace/project"
ONNX_DEFAULT = os.path.join(PROJ, "finn", "resnet8_finn_int8.onnx")
OUTPUT_DIR = os.path.join(PROJ, "finn", "output_resnet8_finn")
OUTPUT_DIR_SYNTH = os.path.join(PROJ, "finn", "output_resnet8_finn_synth")


# ----------------------------------------------------------------------------- export
def do_export(onnx_path):
    import torch
    from brevitas.export import export_qonnx
    sys.path.insert(0, os.path.join(PROJ, "models"))
    from resnet_finn import ResNet8_Brevitas_FINN

    torch.manual_seed(0)
    model = ResNet8_Brevitas_FINN(in_channels=3, num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet8_Brevitas_FINN parameters: {n_params:,}")

    # calibrate activation observers (>30 steps so ParameterFromRuntimeStats finalizes)
    model.train()
    with torch.no_grad():
        for _ in range(40):
            model(torch.rand(64, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        y = model(torch.rand(2, 3, 32, 32))
    print(f"forward OK, output shape {tuple(y.shape)}")

    export_qonnx(model, torch.randn(1, 3, 32, 32), onnx_path)
    print(f"exported QONNX -> {onnx_path}")


# ---------------------------------------------------------------------------- inspect
def do_inspect(onnx_path):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    g = m.graph
    counts = collections.Counter(n.op_type for n in g.node)
    print("OPCOUNTS:", dict(sorted(counts.items())))
    for op in ["AddStreams", "DuplicateStreams", "StreamingEltwise", "Add",
               "MultiThreshold", "Quant", "MatMul", "MVAU", "Im2Col",
               "ConvolutionInputGenerator"]:
        if op in counts:
            print(f"  {op}: {counts[op]}")

    inits = {i.name for i in g.initializer}
    init_by_name = {i.name: i for i in g.initializer}
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n

    def arr(name):
        return numpy_helper.to_array(init_by_name[name]) if name in init_by_name else None

    print("--- Add nodes (JOIN = two dynamic inputs) ---")
    for n in g.node:
        if n.op_type != "Add":
            continue
        dyn = [i for i in n.input if i not in inits]
        tag = "JOIN" if len(dyn) == 2 else "affine"
        print(f"  {n.name or '<add>'} [{tag}] in={list(n.input)}")
        if len(dyn) == 2:
            scales = []
            for di in dyn:
                p = prod.get(di)
                if p is not None and p.op_type == "Quant":
                    s = arr(p.input[1]); bw = arr(p.input[3])
                    sval = None if s is None else float(np.array(s).flatten()[0])
                    scales.append(sval)
                    print(f"     <- Quant {p.name} scale={sval} bitwidth={None if bw is None else float(np.array(bw).flatten()[0])}")
                else:
                    print(f"     <- producer={'None' if p is None else p.op_type}")
            if len(scales) == 2 and None not in scales:
                same = np.allclose(scales[0], scales[1], rtol=1e-4, atol=1e-9)
                print(f"     SHARED-SCALE: {'YES' if same else 'NO'} ({scales[0]} vs {scales[1]})")

    # FINN datatype view (post step_qonnx_to_finn / streamline / convert)
    try:
        from qonnx.core.modelwrapper import ModelWrapper
        mw = ModelWrapper(onnx_path)
        printed = False
        for n in mw.graph.node:
            if n.op_type in ("Add", "AddStreams"):
                dts = [str(mw.get_tensor_datatype(i)) for i in n.input]
                print(f"   FINN-dt {n.op_type} {n.name}: in={dts} out={mw.get_tensor_datatype(n.output[0])}")
                printed = True
        if not printed:
            print("   (no Add/AddStreams nodes for FINN-dt view)")
    except Exception as e:
        print("   (qonnx datatype read skipped:", repr(e), ")")


# ---------------------------------------------------------------------------- compile
def _streamline_linear(model):
    from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
    from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd, GiveUniqueNodeNames
    from qonnx.transformation.remove import RemoveIdentityOps
    from finn.transformation.streamline.absorb import (
        Absorb1BitMulIntoConv, Absorb1BitMulIntoMatMul, AbsorbAddIntoMultiThreshold,
        AbsorbMulIntoMultiThreshold, AbsorbScalarMulAddIntoTopK, FactorOutMulSignMagnitude)
    from finn.transformation.streamline.collapse_repeated import CollapseRepeatedAdd, CollapseRepeatedMul
    from finn.transformation.streamline.reorder import (
        MoveAddPastConv, MoveAddPastMul, MoveMaxPoolPastMultiThreshold, MoveScalarAddPastMatMul,
        MoveScalarLinearPastInvariants, MoveScalarMulPastConv, MoveScalarMulPastMatMul)
    from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
    from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
    for trn in [
        AbsorbScalarMulAddIntoTopK(), ConvertSubToAdd(), ConvertDivToMul(), RemoveIdentityOps(),
        CollapseRepeatedMul(), BatchNormToAffine(), ConvertSignToThres(), MoveAddPastMul(),
        MoveScalarAddPastMatMul(), MoveAddPastConv(), MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(), MoveScalarLinearPastInvariants(), MoveAddPastMul(),
        CollapseRepeatedAdd(), CollapseRepeatedMul(), AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(), MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(), Absorb1BitMulIntoMatMul(), Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet_streamline(model, cfg):
    from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
    from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveUnusedTensors, SortGraph
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
    from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes, AbsorbTransposeIntoMultiThreshold
    from finn.transformation.streamline.reorder import (
        MoveLinearPastEltwiseAdd, MoveLinearPastFork, MoveTransposePastFork, MoveTransposePastJoinAdd)

    for _ in range(4):
        model = _streamline_linear(model)
        # nonlinear (residual) streamlining: fold scales across fork & join
        model = model.transform(MoveLinearPastEltwiseAdd())
        model = model.transform(MoveLinearPastFork())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())
    model = model.transform(LowerConvsToMatMul())
    # transpose cleanup (mainline lacks MoveTransposePastEltwise; loop to fixpoint)
    for _ in range(4):
        for trn in [MoveTransposePastJoinAdd(), MoveTransposePastFork(),
                    AbsorbConsecutiveTransposes(), AbsorbTransposeIntoMultiThreshold()]:
            model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())
    return model


def step_resnet_convert_to_hw(model, cfg):
    from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveUnusedTensors, SortGraph
    from qonnx.transformation.infer_data_layouts import InferDataLayouts
    from qonnx.transformation.infer_datatypes import InferDataTypes
    import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
    from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
    from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
    from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
    for trn in [
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferPool,
        AbsorbConsecutiveTransposes,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,   # residual fork (fanout-2)
        to_hw.InferAddStreamsLayer,         # residual join (Add -> AddStreams)
        to_hw.InferLabelSelectLayer,
        to_hw.InferGlobalAccPoolLayer,
    ]:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


PARTITION_STEPS = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    step_resnet_streamline,
    step_resnet_convert_to_hw,
    "step_create_dataflow_partition",
]
# Full synthesis tail. step_measure_rtlsim_performance is DELIBERATELY omitted: it is
# perf-only, sits before step_synthesize_bitfile in the default order, and needs rtlsim
# (a hang/slow risk) — omitting it de-risks reaching the LUT fit/bust verdict. Resources
# are unaffected.
SYNTH_TAIL = [
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]


def do_compile(onnx_path, full=False, synthonly=False, fps=1000, output_dir=None, folding=None, auto_fifo=False):
    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg
    out = output_dir or (OUTPUT_DIR_SYNTH if (full or synthonly) else OUTPUT_DIR)
    if synthonly:
        # synth-only: stop after step_out_of_context_synthesis (no bitfile/driver/deploy).
        # OOC synth runs synth+opt+place → LUT fit/bust verdict fast (for the fold sweep).
        steps = PARTITION_STEPS + SYNTH_TAIL[:SYNTH_TAIL.index("step_synthesize_bitfile")]
        gen = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,   # OOC synth asserts this is present
            build_cfg.DataflowOutputType.OOC_SYNTH,
        ]
    elif full:
        steps = PARTITION_STEPS + SYNTH_TAIL
        gen = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.OOC_SYNTH,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ]
    else:
        steps = PARTITION_STEPS
        gen = []
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=out,
        target_fps=fps,
        synth_clk_period_ns=10.0,
        board="Ultra96",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        enable_build_pdb_debug=False,
        # FIFO sizing: default False = minimal depth-2 FIFOs (fast, reaches synth reliably).
        # auto_fifo=True = realistic rtlsim-sized FIFOs (LARGEFIFO_RTLSIM, adds BRAM, slow).
        auto_fifo_depths=auto_fifo,
        save_intermediate_models=True,
        verbose=True,
        generate_outputs=gen,
        steps=steps,
        # Optional: freeze folding from a JSON (step_apply_folding_config overrides the
        # target_fps auto-folding). Used to test single-variable folding tweaks (e.g. MVAU
        # weight ram_style remap) starting from an exact prior auto_folding_config.json.
        **({"folding_config_file": folding} if folding else {}),
    )
    print(f"Input ONNX: {onnx_path}")
    print(f"Output dir: {out}")
    print(f"Mode: {'synthonly' if synthonly else ('full' if full else 'partition')}  target_fps={fps}")
    if folding:
        print(f"Folding override: {folding}")
    rc = build.build_dataflow_cfg(onnx_path, cfg)
    print(f"--- build_dataflow_cfg rc={rc} ---")
    im = os.path.join(out, "intermediate_models")
    if os.path.isdir(im):
        print("intermediate_models:", sorted(os.listdir(im)))
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["export", "inspect", "compile", "synth", "synthonly"])
    ap.add_argument("--onnx", default=ONNX_DEFAULT)
    ap.add_argument("--fps", type=int, default=1000)
    ap.add_argument("--output", default=None)
    ap.add_argument("--folding", default=None, help="folding_config_file (JSON) override")
    ap.add_argument("--auto-fifo-depths", dest="auto_fifo", action="store_true",
                    help="realistic rtlsim-sized FIFOs (default: minimal depth-2)")
    args = ap.parse_args()
    if args.mode == "export":
        do_export(args.onnx)
    elif args.mode == "inspect":
        do_inspect(args.onnx)
    elif args.mode == "compile":
        do_compile(args.onnx, full=False)
    elif args.mode == "synth":
        do_compile(args.onnx, full=True, fps=args.fps, output_dir=args.output, folding=args.folding, auto_fifo=args.auto_fifo)
    else:  # synthonly — stop after OOC synth (fold sweep)
        do_compile(args.onnx, synthonly=True, fps=args.fps, output_dir=args.output, folding=args.folding, auto_fifo=args.auto_fifo)
