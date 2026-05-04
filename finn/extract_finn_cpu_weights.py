#!/usr/bin/env python3
"""extract_finn_cpu_weights.py — extract CPU-side weights from a FINN deploy.

FINN's deployment package leaves the FPGA partition as a bitstream and the
CPU-resident layers as ONNX initializers in
intermediate_models/dataflow_parent.onnx. The board-side runner needs those
initializers as .npy files and a small JSON config describing the topology.

Two model kinds are supported, auto-detected from the graph structure:

  CNN (looks for Im2Col): Im2Col → MatMul → MultiThreshold → Streaming →
    [optional MaxPoolNHWC] → Transpose → GAP → Flatten → MatMul → Mul → Add.
    Emits cnn_*.npy + cpu_config.json with model_kind='cnn'.

  MLP (looks for Flatten before MatMul, no Im2Col): Flatten → MatMul →
    MultiThreshold → Streaming → Mul → Add.
    Emits mlp_*.npy + cpu_config.json with model_kind='mlp'.

Output filenames track source initializer names. CNN emits:
  cnn_MatMul_0_param0.npy           — Conv1 weights, shape (kH*kW*C_in, C_out)
  cnn_MultiThreshold_0_param0.npy   — Conv1 thresholds, shape (C_out, T)
  cnn_MatMul_<j>_param0.npy         — classifier weights, shape (gap_in_c, num_classes)
  cnn_Mul_0_param0.npy / cnn_Add_0_param0.npy — output dequant
MLP emits:
  mlp_MatMul_0_param0.npy           — input layer weights, shape (in_dim, hid0)
  mlp_MultiThreshold_0_param0.npy   — input thresholds, shape (hid0, T)
  mlp_Mul_0_param0.npy / mlp_Add_0_param0.npy — output dequant

cpu_config.json fields:
  schema_version: 1
  model_kind: 'cnn' | 'mlp'
  precision: 8 or 4
  ishape_normal / oshape_normal (CNN: NHWC tuples; MLP: 2D)
  cpu_post_maxpool_k, gap_h, gap_w (CNN only)
  weight_files: { W_conv|W_in, thres, W_cls?, mul, add }

The board-side benchmark.py reads cpu_config.json to drive the C runner.

Usage:
  python3 finn/extract_finn_cpu_weights.py \\
      --run-dir finn/size_sweep_runs/cnn_int8_deep_3
  python3 finn/extract_finn_cpu_weights.py \\
      --run-dir finn/size_sweep_runs/mlp_int8_small
"""
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_driver_py(driver_py_path):
    """Extract io_shape_dict subset from a FINN-emitted driver.py.

    Returns dict with idt, odt (str like 'UINT8'), ishape_normal, oshape_normal
    (tuples). Avoids importing the FINN driver (which depends on PYNQ).
    """
    text = Path(driver_py_path).read_text()

    def grab_shape(key):
        m = re.search(rf'"{key}"\s*:\s*\[(\([^)]+\))\]', text)
        if not m:
            raise ValueError(f"missing {key} in {driver_py_path}")
        return ast.literal_eval(m.group(1))

    def grab_dtype(key):
        m = re.search(rf"\"{key}\"\s*:\s*\[DataType\['([A-Z0-9]+)'\]\]", text)
        if not m:
            raise ValueError(f"missing {key} in {driver_py_path}")
        return m.group(1)

    return {
        'idt':           grab_dtype('idt'),
        'odt':           grab_dtype('odt'),
        'ishape_normal': grab_shape('ishape_normal'),
        'oshape_normal': grab_shape('oshape_normal'),
    }


def build_producer_map(graph):
    """tensor_name -> producing node."""
    return {out: node for node in graph.node for out in node.output}


def build_consumer_map(graph):
    """tensor_name -> list of consuming nodes."""
    cons = {}
    for node in graph.node:
        for inp in node.input:
            cons.setdefault(inp, []).append(node)
    return cons


def first_node_of_type(graph, op_type):
    for n in graph.node:
        if n.op_type == op_type:
            return n
    return None


def attr_ints(node, name):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints)
    return None


def extract_topology(graph):
    """Trace the graph and return a dict describing the CPU partition.

    Identifies:
      - W_conv initializer (Conv1's MatMul second input)
      - thres initializer (first MultiThreshold's second input)
      - W_cls initializer (classifier MatMul's second input — the one feeding Mul/Add)
      - mul, add initializers
      - cpu_post_maxpool_k: 0 or 2
      - The classifier MatMul's index name (e.g. 'MatMul_2' or 'MatMul_3')

    Raises ValueError if the topology doesn't match the expected FINN-CNN
    pattern (Im2Col → MatMul → MultiThreshold → StreamingDataflowPartition →
    [optional MaxPoolNHWC] → Transpose → GAP → Flatten → MatMul → Mul → Add).
    """
    producer = build_producer_map(graph)
    consumer = build_consumer_map(graph)

    # ---- Conv1 path: Im2Col → MatMul → MultiThreshold → Streaming
    im2col = first_node_of_type(graph, 'Im2Col')
    if im2col is None:
        raise ValueError("no Im2Col node found — not a FINN-CNN deploy")
    matmul_conv1 = consumer.get(im2col.output[0], [None])[0]
    if matmul_conv1 is None or matmul_conv1.op_type != 'MatMul':
        raise ValueError(f"Im2Col not consumed by MatMul (got {matmul_conv1.op_type if matmul_conv1 else None})")
    if len(matmul_conv1.input) < 2:
        raise ValueError("Conv1 MatMul missing weight input")
    w_conv_init_name = matmul_conv1.input[1]

    mt_conv1 = consumer.get(matmul_conv1.output[0], [None])[0]
    if mt_conv1 is None or mt_conv1.op_type != 'MultiThreshold':
        raise ValueError(f"Conv1 MatMul not consumed by MultiThreshold (got {mt_conv1.op_type if mt_conv1 else None})")
    thres_init_name = mt_conv1.input[1]

    streaming = consumer.get(mt_conv1.output[0], [None])[0]
    if streaming is None or streaming.op_type != 'StreamingDataflowPartition':
        raise ValueError(
            f"MultiThreshold not consumed by StreamingDataflowPartition "
            f"(got {streaming.op_type if streaming else None})")

    # ---- After FPGA partition: optional MaxPoolNHWC, then Transpose → GAP
    cur = consumer.get(streaming.output[0], [None])[0]
    if cur is None:
        raise ValueError("StreamingDataflowPartition output has no consumer")

    cpu_post_maxpool_k = 0
    if cur.op_type == 'MaxPoolNHWC':
        k = attr_ints(cur, 'kernel_shape')
        s = attr_ints(cur, 'strides')
        p = attr_ints(cur, 'pads')
        if k != [2, 2] or s != [2, 2] or p != [0, 0, 0, 0]:
            raise ValueError(
                f"unsupported MaxPoolNHWC config: kernel={k}, strides={s}, pads={p}. "
                f"Only 2x2 stride-2 valid-pad is implemented.")
        cpu_post_maxpool_k = 2
        cur = consumer.get(cur.output[0], [None])[0]
        if cur is None:
            raise ValueError("MaxPoolNHWC output has no consumer")

    if cur.op_type != 'Transpose':
        raise ValueError(f"expected Transpose after FPGA partition (or after CPU MaxPool), got {cur.op_type}")
    cur = consumer.get(cur.output[0], [None])[0]

    if cur is None or cur.op_type != 'GlobalAveragePool':
        raise ValueError(f"expected GlobalAveragePool after Transpose, got {cur.op_type if cur else None}")
    gap_node = cur

    # ---- Classifier path: GAP → Flatten → MatMul → Mul → Add
    cur = consumer.get(gap_node.output[0], [None])[0]
    if cur is None or cur.op_type != 'Flatten':
        raise ValueError(f"expected Flatten after GAP, got {cur.op_type if cur else None}")

    matmul_cls = consumer.get(cur.output[0], [None])[0]
    if matmul_cls is None or matmul_cls.op_type != 'MatMul':
        raise ValueError(f"expected classifier MatMul after Flatten, got {matmul_cls.op_type if matmul_cls else None}")
    w_cls_init_name = matmul_cls.input[1]
    cls_matmul_node_name = matmul_cls.name  # e.g. 'MatMul_1'

    mul_node = consumer.get(matmul_cls.output[0], [None])[0]
    if mul_node is None or mul_node.op_type != 'Mul':
        raise ValueError(f"expected Mul after classifier MatMul, got {mul_node.op_type if mul_node else None}")
    mul_init_name = mul_node.input[1]

    add_node = consumer.get(mul_node.output[0], [None])[0]
    if add_node is None or add_node.op_type != 'Add':
        raise ValueError(f"expected Add after Mul, got {add_node.op_type if add_node else None}")
    add_init_name = add_node.input[1]

    return {
        'w_conv_init':         w_conv_init_name,
        'thres_init':          thres_init_name,
        'w_cls_init':          w_cls_init_name,
        'mul_init':            mul_init_name,
        'add_init':            add_init_name,
        'cpu_post_maxpool_k':  cpu_post_maxpool_k,
        'cls_matmul_node_name': cls_matmul_node_name,
    }


def extract_topology_qi_cnn(graph):
    """QI CNN partition: input QuantIdentity moves Conv1 onto the FPGA.
    Parent graph shape: MultiThreshold (input quant) -> Transpose ->
    StreamingDataflowPartition -> [optional MaxPoolNHWC] -> Transpose ->
    GAP -> Flatten -> MatMul (classifier) -> Mul -> Add.
    Distinct from extract_topology() because there is no Im2Col / no Conv1
    MatMul on the CPU side — those moved into the FPGA partition.
    """
    consumer = build_consumer_map(graph)

    mt_in = first_node_of_type(graph, 'MultiThreshold')
    if mt_in is None:
        raise ValueError("no MultiThreshold node found — not a FINN-CNN-QI deploy")
    input_thres_name = mt_in.input[1]

    cur = consumer.get(mt_in.output[0], [None])[0]
    if cur is None or cur.op_type != 'Transpose':
        raise ValueError(
            f"input MultiThreshold not consumed by Transpose "
            f"(got {cur.op_type if cur else None})")

    streaming = consumer.get(cur.output[0], [None])[0]
    if streaming is None or streaming.op_type != 'StreamingDataflowPartition':
        raise ValueError(
            f"Transpose not consumed by StreamingDataflowPartition "
            f"(got {streaming.op_type if streaming else None})")

    # After FPGA partition: optional MaxPoolNHWC, then Transpose -> GAP.
    cur = consumer.get(streaming.output[0], [None])[0]
    if cur is None:
        raise ValueError("FPGA partition output has no consumer")

    cpu_post_maxpool_k = 0
    if cur.op_type == 'MaxPoolNHWC':
        k = attr_ints(cur, 'kernel_shape')
        s = attr_ints(cur, 'strides')
        p = attr_ints(cur, 'pads')
        if k != [2, 2] or s != [2, 2] or p != [0, 0, 0, 0]:
            raise ValueError(
                f"unsupported MaxPoolNHWC config: kernel={k}, strides={s}, pads={p}. "
                f"Only 2x2 stride-2 valid-pad is implemented.")
        cpu_post_maxpool_k = 2
        cur = consumer.get(cur.output[0], [None])[0]

    if cur is None or cur.op_type != 'Transpose':
        raise ValueError(f"expected Transpose after FPGA partition (or after CPU MaxPool), got {cur.op_type if cur else None}")
    cur = consumer.get(cur.output[0], [None])[0]

    if cur is None or cur.op_type != 'GlobalAveragePool':
        raise ValueError(f"expected GlobalAveragePool after Transpose, got {cur.op_type if cur else None}")
    gap_node = cur

    cur = consumer.get(gap_node.output[0], [None])[0]
    if cur is None or cur.op_type != 'Flatten':
        raise ValueError(f"expected Flatten after GAP, got {cur.op_type if cur else None}")

    matmul_cls = consumer.get(cur.output[0], [None])[0]
    if matmul_cls is None or matmul_cls.op_type != 'MatMul':
        raise ValueError(f"expected classifier MatMul after Flatten, got {matmul_cls.op_type if matmul_cls else None}")
    w_cls_init_name = matmul_cls.input[1]

    mul_node = consumer.get(matmul_cls.output[0], [None])[0]
    if mul_node is None or mul_node.op_type != 'Mul':
        raise ValueError(f"expected Mul after classifier MatMul, got {mul_node.op_type if mul_node else None}")
    mul_init_name = mul_node.input[1]

    add_node = consumer.get(mul_node.output[0], [None])[0]
    if add_node is None or add_node.op_type != 'Add':
        raise ValueError(f"expected Add after Mul, got {add_node.op_type if add_node else None}")
    add_init_name = add_node.input[1]

    return {
        'input_thres_init':   input_thres_name,
        'w_cls_init':         w_cls_init_name,
        'mul_init':           mul_init_name,
        'add_init':           add_init_name,
        'cpu_post_maxpool_k': cpu_post_maxpool_k,
    }


def extract_topology_qi_mlp(graph):
    """QI MLP partition: input QuantIdentity moves Linear1 onto the FPGA.
    Parent graph shape: MultiThreshold -> Flatten -> StreamingDataflowPartition
    -> Mul -> Add. No first Linear MatMul on CPU.
    """
    consumer = build_consumer_map(graph)

    mt_in = first_node_of_type(graph, 'MultiThreshold')
    if mt_in is None:
        raise ValueError("no MultiThreshold node found — not a FINN-MLP-QI deploy")
    input_thres_name = mt_in.input[1]

    cur = consumer.get(mt_in.output[0], [None])[0]
    if cur is None or cur.op_type != 'Flatten':
        raise ValueError(
            f"input MultiThreshold not consumed by Flatten "
            f"(got {cur.op_type if cur else None})")

    streaming = consumer.get(cur.output[0], [None])[0]
    if streaming is None or streaming.op_type != 'StreamingDataflowPartition':
        raise ValueError(
            f"Flatten not consumed by StreamingDataflowPartition "
            f"(got {streaming.op_type if streaming else None})")

    mul_node = consumer.get(streaming.output[0], [None])[0]
    if mul_node is None or mul_node.op_type != 'Mul':
        raise ValueError(
            f"expected Mul after FPGA partition, got {mul_node.op_type if mul_node else None}")
    mul_init_name = mul_node.input[1]

    add_node = consumer.get(mul_node.output[0], [None])[0]
    if add_node is None or add_node.op_type != 'Add':
        raise ValueError(
            f"expected Add after Mul, got {add_node.op_type if add_node else None}")
    add_init_name = add_node.input[1]

    return {
        'input_thres_init': input_thres_name,
        'mul_init':         mul_init_name,
        'add_init':         add_init_name,
    }


def extract_topology_mlp(graph):
    """Trace the MLP graph and return a dict describing the CPU partition.

    MLP pattern: Flatten → MatMul (input layer, CPU) → MultiThreshold (CPU)
                 → StreamingDataflowPartition → Mul → Add.
    No Im2Col (it's an MLP), no GAP, no classifier MatMul.

    Raises ValueError if the topology doesn't match.
    """
    consumer = build_consumer_map(graph)

    flatten = first_node_of_type(graph, 'Flatten')
    if flatten is None:
        raise ValueError("no Flatten node found — not a FINN-MLP deploy")
    matmul_in = consumer.get(flatten.output[0], [None])[0]
    if matmul_in is None or matmul_in.op_type != 'MatMul':
        raise ValueError(
            f"Flatten not consumed by MatMul (got {matmul_in.op_type if matmul_in else None})")
    if len(matmul_in.input) < 2:
        raise ValueError("input MatMul missing weight input")
    w_in_init_name = matmul_in.input[1]

    mt_in = consumer.get(matmul_in.output[0], [None])[0]
    if mt_in is None or mt_in.op_type != 'MultiThreshold':
        raise ValueError(
            f"input MatMul not consumed by MultiThreshold (got {mt_in.op_type if mt_in else None})")
    thres_init_name = mt_in.input[1]

    streaming = consumer.get(mt_in.output[0], [None])[0]
    if streaming is None or streaming.op_type != 'StreamingDataflowPartition':
        raise ValueError(
            f"MultiThreshold not consumed by StreamingDataflowPartition "
            f"(got {streaming.op_type if streaming else None})")

    mul_node = consumer.get(streaming.output[0], [None])[0]
    if mul_node is None or mul_node.op_type != 'Mul':
        raise ValueError(
            f"expected Mul after FPGA partition, got {mul_node.op_type if mul_node else None}")
    mul_init_name = mul_node.input[1]

    add_node = consumer.get(mul_node.output[0], [None])[0]
    if add_node is None or add_node.op_type != 'Add':
        raise ValueError(
            f"expected Add after Mul, got {add_node.op_type if add_node else None}")
    add_init_name = add_node.input[1]

    return {
        'w_in_init':    w_in_init_name,
        'thres_init':   thres_init_name,
        'mul_init':     mul_init_name,
        'add_init':     add_init_name,
    }


def get_initializer(graph, name):
    for ini in graph.initializer:
        if ini.name == name:
            return numpy_helper.to_array(ini)
    raise ValueError(f"initializer {name!r} not found in graph")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True,
                    help='FINN run directory (contains intermediate_models/ and deploy/)')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing cnn_*.npy / cpu_config.json')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    onnx_path = run_dir / 'intermediate_models' / 'dataflow_parent.onnx'
    deploy_dir = run_dir / 'deploy'
    driver_py = deploy_dir / 'driver' / 'driver.py'

    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found", file=sys.stderr)
        return 2
    if not deploy_dir.exists():
        print(f"ERROR: {deploy_dir} not found", file=sys.stderr)
        return 2
    if not driver_py.exists():
        print(f"ERROR: {driver_py} not found", file=sys.stderr)
        return 2

    print(f"Loading {onnx_path}")
    model = onnx.load(str(onnx_path))
    io = parse_driver_py(driver_py)

    # Auto-detect partition shape:
    #   - classic CNN: has Im2Col before first MatMul (Conv1 on CPU)
    #   - classic MLP: has Flatten then MatMul (Linear1 on CPU)
    #   - QI CNN/MLP: input MultiThreshold then Transpose/Flatten -> Streaming
    #                 (no Im2Col, no first MatMul on CPU)
    has_im2col = first_node_of_type(model.graph, 'Im2Col') is not None
    first_mm = first_node_of_type(model.graph, 'MatMul')
    first_sdp = first_node_of_type(model.graph, 'StreamingDataflowPartition')
    # QI variants: SDP precedes (or no) any MatMul. Detect by node order.
    sdp_before_matmul = False
    if first_sdp is not None:
        for n in model.graph.node:
            if n is first_sdp:
                sdp_before_matmul = True
                break
            if n is first_mm:
                break
    if has_im2col:
        partition = 'classic'
        model_kind = 'cnn'
    elif sdp_before_matmul:
        # QI variant. CNN vs MLP: CNN has Transpose between MT and SDP, MLP has Flatten.
        consumer = build_consumer_map(model.graph)
        mt = first_node_of_type(model.graph, 'MultiThreshold')
        partition = 'qi'
        next_op = consumer.get(mt.output[0], [None])[0].op_type if mt else None
        model_kind = 'cnn' if next_op == 'Transpose' else 'mlp'
    else:
        partition = 'classic'
        model_kind = 'mlp'
    print(f"  model_kind: {model_kind}, partition: {partition}")

    # Precision detection from idt name: 'UINT8' -> 8, 'UINT4' -> 4.
    idt = io['idt']
    m = re.match(r'(?:U?INT)(\d+)$', idt)
    if not m:
        raise ValueError(f"could not parse precision from idt={idt!r}")
    precision = int(m.group(1))

    if model_kind == 'cnn' and partition == 'classic':
        topo = extract_topology(model.graph)

        fpga_out_h = io['oshape_normal'][1]
        fpga_out_w = io['oshape_normal'][2]
        fpga_out_c = io['oshape_normal'][3]
        k = topo['cpu_post_maxpool_k']
        if k == 0:
            gap_h, gap_w = fpga_out_h, fpga_out_w
        else:
            if fpga_out_h < k or fpga_out_w < k:
                raise ValueError(f"fpga_out spatial {fpga_out_h}x{fpga_out_w} too small for {k}x{k} MaxPool")
            gap_h = (fpga_out_h - k) // k + 1
            gap_w = (fpga_out_w - k) // k + 1

        files = {
            'W_conv': f"cnn_{topo['w_conv_init']}.npy",
            'thres':  f"cnn_{topo['thres_init']}.npy",
            'W_cls':  f"cnn_{topo['w_cls_init']}.npy",
            'mul':    f"cnn_{topo['mul_init']}.npy",
            'add':    f"cnn_{topo['add_init']}.npy",
        }
        arrays = {
            'W_conv': get_initializer(model.graph, topo['w_conv_init']),
            'thres':  get_initializer(model.graph, topo['thres_init']),
            'W_cls':  get_initializer(model.graph, topo['w_cls_init']),
            'mul':    get_initializer(model.graph, topo['mul_init']),
            'add':    get_initializer(model.graph, topo['add_init']),
        }
        cfg_extras = {
            'ishape_normal':      list(io['ishape_normal']),
            'oshape_normal':      list(io['oshape_normal']),
            'cpu_post_maxpool_k': k,
            'gap_h':              gap_h,
            'gap_w':              gap_w,
        }
        cfg_log_extra = (f"k={k}, gap={gap_h}x{gap_w}, "
                         f"fpga_out={fpga_out_h}x{fpga_out_w}x{fpga_out_c}")
    elif model_kind == 'cnn' and partition == 'qi':
        topo = extract_topology_qi_cnn(model.graph)

        fpga_out_h = io['oshape_normal'][1]
        fpga_out_w = io['oshape_normal'][2]
        fpga_out_c = io['oshape_normal'][3]
        k = topo['cpu_post_maxpool_k']
        if k == 0:
            gap_h, gap_w = fpga_out_h, fpga_out_w
        else:
            if fpga_out_h < k or fpga_out_w < k:
                raise ValueError(f"fpga_out spatial {fpga_out_h}x{fpga_out_w} too small for {k}x{k} MaxPool")
            gap_h = (fpga_out_h - k) // k + 1
            gap_w = (fpga_out_w - k) // k + 1

        files = {
            'input_thres': f"cnn_qi_{topo['input_thres_init']}.npy",
            'W_cls':       f"cnn_qi_{topo['w_cls_init']}.npy",
            'mul':         f"cnn_qi_{topo['mul_init']}.npy",
            'add':         f"cnn_qi_{topo['add_init']}.npy",
        }
        arrays = {
            'input_thres': get_initializer(model.graph, topo['input_thres_init']),
            'W_cls':       get_initializer(model.graph, topo['w_cls_init']),
            'mul':         get_initializer(model.graph, topo['mul_init']),
            'add':         get_initializer(model.graph, topo['add_init']),
        }
        cfg_extras = {
            'ishape_normal':      list(io['ishape_normal']),
            'oshape_normal':      list(io['oshape_normal']),
            'cpu_post_maxpool_k': k,
            'gap_h':              gap_h,
            'gap_w':              gap_w,
        }
        cfg_log_extra = (f"k={k}, gap={gap_h}x{gap_w}, "
                         f"fpga_out={fpga_out_h}x{fpga_out_w}x{fpga_out_c}, "
                         f"input_thres={list(arrays['input_thres'].shape)}")
    elif model_kind == 'mlp' and partition == 'qi':
        topo = extract_topology_qi_mlp(model.graph)
        files = {
            'input_thres': f"mlp_qi_{topo['input_thres_init']}.npy",
            'mul':         f"mlp_qi_{topo['mul_init']}.npy",
            'add':         f"mlp_qi_{topo['add_init']}.npy",
        }
        arrays = {
            'input_thres': get_initializer(model.graph, topo['input_thres_init']),
            'mul':         get_initializer(model.graph, topo['mul_init']),
            'add':         get_initializer(model.graph, topo['add_init']),
        }
        cfg_extras = {
            'ishape_normal': list(io['ishape_normal']),
            'oshape_normal': list(io['oshape_normal']),
        }
        cfg_log_extra = (f"input_thres={list(arrays['input_thres'].shape)}, "
                         f"out_dim={arrays['add'].shape[0]}")
    else:
        # classic MLP
        topo = extract_topology_mlp(model.graph)
        files = {
            'W_in':  f"mlp_{topo['w_in_init']}.npy",
            'thres': f"mlp_{topo['thres_init']}.npy",
            'mul':   f"mlp_{topo['mul_init']}.npy",
            'add':   f"mlp_{topo['add_init']}.npy",
        }
        arrays = {
            'W_in':  get_initializer(model.graph, topo['w_in_init']),
            'thres': get_initializer(model.graph, topo['thres_init']),
            'mul':   get_initializer(model.graph, topo['mul_init']),
            'add':   get_initializer(model.graph, topo['add_init']),
        }
        cfg_extras = {
            'ishape_normal': list(io['ishape_normal']),
            'oshape_normal': list(io['oshape_normal']),
        }
        cfg_log_extra = (f"in_dim={arrays['W_in'].shape[0]}, "
                         f"hid0={arrays['W_in'].shape[1]}, "
                         f"out_dim={arrays['add'].shape[0]}")

    # Refuse silent overwrite (--force opts in).
    for fn in files.values():
        path = deploy_dir / fn
        if path.exists() and not args.force:
            print(f"ERROR: {path} exists. Use --force to overwrite.", file=sys.stderr)
            return 2
    cfg_path = deploy_dir / 'cpu_config.json'
    if cfg_path.exists() and not args.force:
        print(f"ERROR: {cfg_path} exists. Use --force to overwrite.", file=sys.stderr)
        return 2

    for key, fn in files.items():
        path = deploy_dir / fn
        np.save(str(path), arrays[key])
        print(f"  wrote {fn}: shape={list(arrays[key].shape)} dtype={arrays[key].dtype}")

    cfg = {
        'schema_version':     1,
        'model_kind':         model_kind,
        'partition':          partition,
        'precision':          precision,
        'weight_files':       files,
    }
    cfg.update(cfg_extras)
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(f"  wrote cpu_config.json: {cfg_log_extra}, INT{precision}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
