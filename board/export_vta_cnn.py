#!/usr/bin/env python3
"""Export pre-compiled VTA CNN modules for board-side inference.

Loads a Brevitas CNN checkpoint, folds BatchNorm into conv weights,
compiles VTA GEMM+shift+clip modules (cross-compiled for aarch64),
and saves everything as a self-contained model directory.

CNN inference on VTA uses im2col: each conv layer is converted to a
matrix multiply, then executed via the same GEMM+shift+clip schedule
as the MLP. MaxPool, GlobalAvgPool, and ReLU run on CPU between
VTA GEMM calls.

Two architectures supported (--arch flag, default 'cnn' for legacy):

  cnn:     CNN tiny/small/medium/deep_3/large on MNIST (28x28, 1 channel)
           Conv->BN->ReLU->MaxPool blocks under model.features Sequential,
           single QuantLinear under model.classifier. No skip connections.

  resnet8: MLPerf Tiny ResNet-8 on CIFAR-10 (32x32, 3 channels). Stem +
           three residual blocks (stages 1-3) + GAP + FC. Skip-add is in
           CPU float-space post-dequant, pre-ReLU. Layers carry optional
           skip metadata: 'consume_input_from' (override default chain),
           'skip_add_from' (post-dequant float add).

Usage (from Ubuntu host):
    cd ~/dev/CEN571-final/tvm-v0.12.0
    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 export_vta_cnn.py \
            --arch cnn --dataset mnist \
            --checkpoint ../finn-vs-vitisai/finn/cnn_mnist_tiny.pth \
            --output-dir ./vta_export/cnn_mnist_tiny/

    PYTHONPATH=$(pwd)/python:$(pwd)/vta/python TVM_HOME=$(pwd) \
        python3 export_vta_cnn.py \
            --arch resnet8 --dataset cifar10 \
            --checkpoint ../finn-vs-vitisai/finn/resnet8_cifar10_int8.pth \
            --output-dir ./vta_export/resnet8_cifar10_int8/

Then copy to board:
    scp -r ./vta_export/<dir>/ xilinx@192.168.3.1:/home/xilinx/models/vta/<dir>/
"""
import numpy as np
import tvm
from tvm import te
from tvm.contrib import utils
import vta
import json
import math
import os
import sys
import argparse
import struct
import gzip
import torch


# ---- MNIST loading ----

MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]

def download_mnist(data_dir='./mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    filenames = {
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }
    paths = {}
    for key, filename in filenames.items():
        fname = os.path.join(data_dir, filename)
        if not os.path.exists(fname):
            from urllib.request import urlretrieve
            for mirror in MNIST_MIRRORS:
                try:
                    print(f"  Downloading {key} from {mirror}...", end=" ", flush=True)
                    urlretrieve(mirror + filename, fname)
                    print("OK")
                    break
                except Exception as e:
                    print(f"failed ({e})")
            else:
                raise RuntimeError(f"Could not download {filename}")
        paths[key] = fname
    return paths

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---- CIFAR-10 loading (for ResNet-8 calibration/verify) ----

def load_cifar10_test(data_dir):
    """Load CIFAR-10 test set from torchvision pickle batches.

    Expects ``cifar-10-batches-py/`` under ``data_dir`` (already present in
    finn-vs-vitisai/data/). Returns (images, labels) where:
      images: float32 (N, 3, 32, 32) in [0, 1]
      labels: uint8 (N,)
    """
    import pickle
    test_path = os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch')
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"CIFAR-10 test_batch not found at {test_path}. "
            f"Expected torchvision-extracted batches under {data_dir}/cifar-10-batches-py/")
    with open(test_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    raw = batch[b'data']  # (10000, 3072) uint8, RGB row-major
    labels = np.array(batch[b'labels'], dtype=np.uint8)
    images = raw.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return images, labels


# ---- BN folding ----

def fold_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into preceding Conv2d weights.

    Conv: y = W * x (no bias in Brevitas conv)
    BN:   z = gamma * (y - mean) / sqrt(var + eps) + beta

    Folded: z = W_folded * x + b_folded
      W_folded = gamma / sqrt(var + eps) * W   (per output channel)
      b_folded = -gamma * mean / sqrt(var + eps) + beta
    """
    # conv_weight: (C_out, C_in, kH, kW)
    C_out = conv_weight.shape[0]
    scale = bn_weight / np.sqrt(bn_var + eps)  # (C_out,)

    # Scale each output channel's conv filter
    W_folded = conv_weight * scale.reshape(C_out, 1, 1, 1)
    b_folded = -bn_mean * scale + bn_bias

    return W_folded, b_folded


# ---- im2col ----

def im2col(x, kH, kW, pad=0, stride=1):
    """Extract sliding window patches.

    x: (H, W, C) -> output: (H_out * W_out, kH * kW * C)
    """
    H, W, C = x.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    if pad > 0:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    patches = np.zeros((H_out * W_out, kH * kW * C), dtype=x.dtype)
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+kH, j*stride:j*stride+kW, :]
            patches[idx] = patch.flatten()
            idx += 1
    return patches, H_out, W_out


def maxpool2d(x, pool_size=2):
    """Max pooling on (H, W, C) tensor."""
    H, W, C = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    out = np.zeros((H_out, W_out, C), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            out[i, j] = x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size].max(axis=(0, 1))
    return out


# ---- VTA helpers ----

def tile_weights_2d(W_flat_int8, env):
    """Tile a 2D weight matrix (out_f, in_f) for VTA.
    Both dims must be multiples of BLOCK_OUT and BLOCK_IN respectively.
    """
    out_f, in_f = W_flat_int8.shape
    m = out_f // env.BLOCK_OUT
    n = in_f // env.BLOCK_IN
    return W_flat_int8.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose(0, 2, 1, 3)


def pad_to_block(x, block_size, axis):
    """Pad a dimension to a multiple of block_size."""
    current = x.shape[axis]
    if current % block_size == 0:
        return x, current
    pad_amount = block_size - (current % block_size)
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (0, pad_amount)
    return np.pad(x, pad_widths, mode='constant'), current


def compute_shift_bits(activations_int8, W_int8):
    """Determine right-shift from calibration data.
    activations_int8: (N, in_f), W_int8: (out_f, in_f)
    """
    acc = activations_int8.astype(np.int32) @ W_int8.T.astype(np.int32)
    max_abs = np.max(np.abs(acc))
    if max_abs <= 127:
        return 0
    return int(math.ceil(math.log2(max_abs / 127.0)))


# ---- VTA module compilation ----

def compile_gemm_with_shift(env, o, n, m, shift_bits):
    """Compile VTA GEMM + ALU shift + clip. Returns TVM Module."""
    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
    A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
    B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf")

    ko = te.reduce_axis((0, n), name="ko")
    ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
    C_buf = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda bo, co, bi, ci: te.sum(
            A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
            B_buf[co, ko, ci, ki].astype(env.acc_dtype),
            axis=[ko, ki]),
        name="C_buf")

    shr_const = tvm.tir.const(shift_bits, env.acc_dtype)
    C_shr = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_buf(*i) >> shr_const, name="C_shr")

    C_clip_hi = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.min(C_shr(*i), tvm.tir.const(127, env.acc_dtype)),
        name="C_clip_hi")

    C_clip_lo = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: tvm.te.max(C_clip_hi(*i), tvm.tir.const(-128, env.acc_dtype)),
        name="C_clip_lo")

    C = te.compute(
        (o, m, env.BATCH, env.BLOCK_OUT),
        lambda *i: C_clip_lo(*i).astype(env.inp_dtype), name="C")

    s = te.create_schedule(C.op)
    s[A_buf].set_scope(env.inp_scope)
    s[B_buf].set_scope(env.wgt_scope)
    s[C_buf].set_scope(env.acc_scope)
    s[C_shr].set_scope(env.acc_scope)
    s[C_clip_hi].set_scope(env.acc_scope)
    s[C_clip_lo].set_scope(env.acc_scope)

    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1],
        s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki)
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)
    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

    s[C_shr].pragma(s[C_shr].op.axis[0], env.alu)
    s[C_clip_hi].pragma(s[C_clip_hi].op.axis[0], env.alu)
    s[C_clip_lo].pragma(s[C_clip_lo].op.axis[0], env.alu)
    s[C].pragma(s[C].op.axis[0], env.dma_copy)

    mod = vta.build(s, [A, B, C],
                    tvm.target.vta(),
                    tvm.target.arm_cpu("ultra96"),
                    name="my_gemm")
    return mod


# ---- Weight extraction and preparation ----

def load_brevitas_cnn(checkpoint_path, size='tiny'):
    """Load Brevitas CNN checkpoint, fold BN, return layer list.

    Walks model.features Sequential dynamically to find every QuantConv2d
    block, so 2-conv (tiny/small/medium) and 3-conv (deep_3/large) topologies
    both work without code changes.

    Returns list of dicts with keys:
      'type': 'conv' or 'dense'
      'W': weight array (for conv: flattened to 2D after im2col reshape)
      'b': bias array
      'kernel_size', 'padding', 'in_channels', 'out_channels', 'pool' (conv only)
    """
    # Need brevitas to introspect QuantConv2d/QuantLinear modules.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(repo_root, 'models'))
    import torch.nn as nn
    import brevitas.nn as qnn
    from cnn import CNN_Brevitas, get_cnn_config

    channels = get_cnn_config(size)
    model = CNN_Brevitas(in_channels=1, channels=channels)
    sd = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()

    layers = []

    # Walk features for QuantConv2d → BatchNorm2d → QuantReLU → MaxPool2d blocks.
    # Standard CNN_Brevitas block is 4 modules; we don't assume the offset and
    # instead scan forward from each conv until the next conv (or end).
    feats = list(model.features)
    i = 0
    while i < len(feats):
        m = feats[i]
        if isinstance(m, qnn.QuantConv2d):
            conv = m
            bn = None
            pool = 0
            j = i + 1
            while j < len(feats) and not isinstance(feats[j], qnn.QuantConv2d):
                if isinstance(feats[j], nn.BatchNorm2d):
                    bn = feats[j]
                elif isinstance(feats[j], nn.MaxPool2d):
                    p = feats[j].kernel_size
                    pool = p[0] if isinstance(p, tuple) else int(p)
                j += 1

            conv_w = conv.weight.detach().numpy()
            if bn is not None:
                W_folded, b_folded = fold_bn_into_conv(
                    conv_w,
                    bn.weight.detach().numpy(),
                    bn.bias.detach().numpy(),
                    bn.running_mean.detach().numpy(),
                    bn.running_var.detach().numpy(),
                    bn.eps,
                )
            else:
                W_folded = conv_w
                b_folded = (conv.bias.detach().numpy() if conv.bias is not None
                            else np.zeros(conv_w.shape[0], dtype=np.float32))

            C_out, C_in, kH, kW = W_folded.shape
            pad = conv.padding
            if isinstance(pad, tuple):
                pad = pad[0]

            layers.append({
                'type':         'conv',
                'W':            W_folded.transpose(0, 2, 3, 1).reshape(C_out, -1),
                'b':            b_folded,
                'kernel_size':  int(kH),
                'padding':      int(pad),
                'in_channels':  int(C_in),
                'out_channels': int(C_out),
                'pool':         int(pool),
            })
            i = j
        else:
            i += 1

    # Classifier: single QuantLinear inside the Sequential.
    cls_lins = [m for m in model.classifier if isinstance(m, qnn.QuantLinear)]
    if len(cls_lins) != 1:
        raise RuntimeError(
            f'expected exactly one QuantLinear in classifier; got {len(cls_lins)}')
    cls = cls_lins[0]
    layers.append({
        'type': 'dense',
        'W':    cls.weight.detach().numpy(),
        'b':    cls.bias.detach().numpy(),
    })

    return layers


def _extract_conv_layer(conv, bn, kernel_h, kernel_w, padding, stride):
    """Build a layer dict from a Brevitas QuantConv2d (+ optional BatchNorm).

    Returns a dict with the standard schema used downstream
    (W reshaped to (C_out, kH*kW*C_in), b folded, kernel/padding/in/out).
    Caller fills in skip metadata (consume_input_from, skip_add_from,
    branch_only) and the 'pool' field as appropriate.
    """
    conv_w = conv.weight.detach().numpy()
    if bn is not None:
        W_folded, b_folded = fold_bn_into_conv(
            conv_w,
            bn.weight.detach().numpy(),
            bn.bias.detach().numpy(),
            bn.running_mean.detach().numpy(),
            bn.running_var.detach().numpy(),
            bn.eps,
        )
    else:
        W_folded = conv_w
        b_folded = (conv.bias.detach().numpy() if conv.bias is not None
                    else np.zeros(conv_w.shape[0], dtype=np.float32))
    C_out, C_in, kH, kW = W_folded.shape
    return {
        'type':         'conv',
        'W':            W_folded.transpose(0, 2, 3, 1).reshape(C_out, -1),
        'b':            b_folded,
        'kernel_size':  int(kH),
        'padding':      int(padding),
        'stride':       int(stride),
        'in_channels':  int(C_in),
        'out_channels': int(C_out),
        'pool':         0,
    }


def load_brevitas_resnet8(checkpoint_path):
    """Load Brevitas ResNet-8 checkpoint, fold BN, return layer list with skip metadata.

    Schema additions vs load_brevitas_cnn:
      'consume_input_from': index of an earlier layer whose post-dequant float
                            output is the input to this layer's GEMM (overrides
                            the default chain input from prev layer).
      'skip_add_from':      index of an earlier layer whose post-dequant float
                            output is added to this layer's post-dequant output
                            BEFORE the post-add ReLU (residual add).
      'branch_only':        True if this layer's output feeds only a future
                            skip-add (not the next layer's chain input). The
                            following layer should ignore this layer's output
                            and pull from its own consume_input_from.
      'apply_relu':         True if the layer's post-dequant (and post-skip-add
                            if any) should be passed through ReLU. False on
                            stem-relu placement is captured per ResNet-8's spec
                            (relu after the BN of the FIRST conv in each block;
                            the second conv's relu happens AFTER the skip-add).

    Layer order produced for ResNet-8 (10 layers total, 9 conv + 1 dense):
      0: stem conv    (3->16, stride 1)            relu, no pool
      1: stage1 conv1 (16->16, stride 1)            relu, no pool
      2: stage1 conv2 (16->16, stride 1) skip+=0   relu after skip-add
      3: stage2 down  (16->32, 1x1, stride 2)      branch only, NO relu
      4: stage2 conv1 (16->32, stride 2) input=2   relu, no pool
      5: stage2 conv2 (32->32, stride 1) skip+=3   relu after skip-add
      6: stage3 down  (32->64, 1x1, stride 2)      branch only, NO relu
      7: stage3 conv1 (32->64, stride 2) input=5   relu, no pool
      8: stage3 conv2 (64->64, stride 1) skip+=6   relu after skip-add
      9: dense        (64->10), GAP on input
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(repo_root, 'models'))
    import brevitas.nn as qnn
    from resnet import ResNet8_Brevitas

    model = ResNet8_Brevitas(in_channels=3, num_classes=10)
    sd = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()

    layers = []

    # 0: stem conv (3->16, stride 1) — relu, no pool
    L = _extract_conv_layer(model.stem_conv, model.stem_bn, 3, 3, 1, 1)
    L['apply_relu'] = True
    layers.append(L)

    def add_block(block, prev_chain_idx, fork_input_idx, downsample_present):
        """Append block layers to `layers`, return updated (chain_idx_after_block).

        block:               Brevitas BasicBlock_Brevitas
        prev_chain_idx:      index of the layer whose output feeds this block
                             (= identity skip source, AND = downsample input,
                             AND = main path conv1 input).
        fork_input_idx:      same as prev_chain_idx (only here for clarity)
        downsample_present:  True if the block has a 1x1 downsample
        """
        if downsample_present:
            ds_conv = block.downsample[0]
            ds_bn = block.downsample[1]
            ds = _extract_conv_layer(ds_conv, ds_bn, 1, 1,
                                     0, ds_conv.stride[0])
            ds['consume_input_from'] = fork_input_idx
            ds['branch_only'] = True
            ds['apply_relu'] = False  # downsample is just a skip path
            layers.append(ds)
            skip_src = len(layers) - 1
        else:
            skip_src = fork_input_idx  # identity skip = pre-block activation

        c1 = _extract_conv_layer(block.conv1, block.bn1, 3, 3,
                                 1, block.conv1.stride[0])
        c1['consume_input_from'] = fork_input_idx  # main path also forks here
        c1['apply_relu'] = True
        layers.append(c1)

        c2 = _extract_conv_layer(block.conv2, block.bn2, 3, 3, 1, 1)
        c2['skip_add_from'] = skip_src
        c2['apply_relu'] = True   # ReLU after skip-add (post_add_relu)
        layers.append(c2)

        return len(layers) - 1  # this block's output is the last layer added

    # stage 1: identity skip (16->16, stride 1)
    chain = add_block(model.stage1, prev_chain_idx=0, fork_input_idx=0,
                      downsample_present=False)
    # stage 2: 1x1 downsample (16->32, stride 2)
    chain = add_block(model.stage2, prev_chain_idx=chain, fork_input_idx=chain,
                      downsample_present=True)
    # stage 3: 1x1 downsample (32->64, stride 2)
    chain = add_block(model.stage3, prev_chain_idx=chain, fork_input_idx=chain,
                      downsample_present=True)

    # FC: 64->10 (after GAP). The dense layer's "input layer" is `chain`
    # (the post-stage3 spatial activation), GAP'd to a 64-dim vector.
    layers.append({
        'type':              'dense',
        'W':                 model.fc.weight.detach().numpy(),
        'b':                 model.fc.bias.detach().numpy(),
        'consume_input_from': chain,
    })
    return layers


def prepare_vta_layers(layers, env):
    """Pad weights to VTA alignment, quantize, tile.

    For conv layers, im2col flattens the kernel to (C_out, kH*kW*C_in).
    We pad the inner dim to BLOCK_IN multiples and outer to BLOCK_OUT multiples.

    Returns list of dicts with VTA-ready data.
    """
    vta_layers = []
    for i, layer in enumerate(layers):
        W = layer['W'].astype(np.float32)
        b = layer['b'].astype(np.float32)
        out_f, in_f = W.shape

        # Pad output dim to BLOCK_OUT multiple
        real_out = out_f
        if out_f % env.BLOCK_OUT != 0:
            pad_out = env.BLOCK_OUT - (out_f % env.BLOCK_OUT)
            W = np.pad(W, ((0, pad_out), (0, 0)), mode='constant')
            b = np.pad(b, (0, pad_out), mode='constant')
            out_f = W.shape[0]

        # Pad input dim to BLOCK_IN multiple
        real_in = in_f
        if in_f % env.BLOCK_IN != 0:
            pad_in = env.BLOCK_IN - (in_f % env.BLOCK_IN)
            W = np.pad(W, ((0, 0), (0, pad_in)), mode='constant')
            in_f = W.shape[1]

        # Quantize weights
        w_max = np.max(np.abs(W))
        w_scale = w_max / 127.0 if w_max > 0 else 1e-10
        W_int8 = np.clip(np.round(W / w_scale), -128, 127).astype(np.int8)
        W_tiled = tile_weights_2d(W_int8, env)

        n_tiles = in_f // env.BLOCK_IN
        m_tiles = out_f // env.BLOCK_OUT

        info = {
            'W_int8': W_int8,
            'W_tiled': W_tiled,
            'b_float': b,
            'w_scale': float(w_scale),
            'in_f': in_f,       # padded
            'out_f': out_f,     # padded
            'real_in': real_in,
            'real_out': real_out,
            'n_tiles': n_tiles,
            'm_tiles': m_tiles,
            'layer_type': layer['type'],
        }

        if layer['type'] == 'conv':
            info['kernel_size'] = layer['kernel_size']
            info['padding'] = layer['padding']
            info['in_channels'] = layer['in_channels']
            info['out_channels'] = layer['out_channels']
            info['pool'] = layer.get('pool', 0)
            if 'stride' in layer:
                info['stride'] = layer['stride']

        # Skip-connection metadata (resnet8 path; absent in legacy cnn flow).
        for k in ('consume_input_from', 'skip_add_from', 'branch_only',
                  'apply_relu'):
            if k in layer:
                info[k] = layer[k]

        vta_layers.append(info)
        print(f"  Layer {i} ({layer['type']}): {real_in}->{real_out} "
              f"(padded {in_f}->{out_f}), "
              f"w_scale={w_scale:.6f}, tiles n={n_tiles} m={m_tiles}")

    return vta_layers


# ---- Calibration ----

def _normalize_calibration_image(img):
    """Convert calibration image to spatial format used by im2col.

    MNIST (legacy): img shape (H, W) — leave as 2D so the layer-0 conv hits
                    the existing `h_float.ndim == 2` branch.
    CIFAR-10:       img shape (C, H, W) — transpose to (H, W, C).
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img.transpose(1, 2, 0)
    raise ValueError(f"unsupported calibration image ndim {img.ndim}")


def _save_targets(vta_layers):
    """Set of layer indices whose post-dequant output future layers reference."""
    targets = set()
    for vl in vta_layers:
        for k in ('consume_input_from', 'skip_add_from'):
            if k in vl:
                targets.add(vl[k])
    return targets


def calibrate_cnn(vta_layers, cal_images, env):
    """Run calibration images through CNN pipeline to determine shift amounts.

    cal_images: (N, H, W) for MNIST OR (N, C, H, W) for CIFAR-10. float32 [0,1].
    Returns (shift_amounts, global_x_scale).

    Legacy CNN-on-MNIST path is byte-identical to the pre-refactor logic when
    no layer carries skip metadata. Skip-add layers (resnet8 path) update the
    saved-activations dict and apply the residual add post-dequant, pre-ReLU.
    """
    N = len(cal_images)
    shift_amounts = []

    # Quantize input images
    global_x_scale = np.mean([np.max(np.abs(img)) / 127.0 for img in cal_images])
    if global_x_scale < 1e-10:
        global_x_scale = 1e-10

    save_targets = _save_targets(vta_layers)
    layer_max_acc = [0.0 for _ in vta_layers]

    for img_idx in range(N):
        img = cal_images[img_idx]

        # Quantize input
        x_s = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        current_scale = x_s

        # Process through layers
        h_float = _normalize_calibration_image(img)
        # saved_activations: layer_idx -> (post-dequant float, scale-of-that-output)
        saved_activations = {}

        for i, vl in enumerate(vta_layers):
            # --- Resolve input for this layer ---
            if 'consume_input_from' in vl:
                input_h, input_scale = saved_activations[vl['consume_input_from']]
            else:
                input_h = h_float
                input_scale = current_scale

            if vl['layer_type'] == 'conv':
                # im2col
                if input_h.ndim == 2:
                    h_spatial = input_h[:, :, np.newaxis]
                else:
                    h_spatial = input_h  # (H, W, C)

                patches, H_out, W_out = im2col(
                    h_spatial, vl['kernel_size'], vl['kernel_size'],
                    pad=vl['padding'], stride=vl.get('stride', 1))

                real_patch_dim = patches.shape[1]
                if real_patch_dim < vl['in_f']:
                    patches = np.pad(patches, ((0, 0), (0, vl['in_f'] - real_patch_dim)),
                                     mode='constant')

                p_int8 = np.clip(np.round(patches / input_scale), -128, 127).astype(np.int8)

                acc = p_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                max_abs = np.max(np.abs(acc))
                if max_abs > layer_max_acc[i]:
                    layer_max_acc[i] = max_abs

                shift = int(math.ceil(math.log2(max_abs / 127.0))) if max_abs > 127 else 0
                shifted = acc >> shift
                clipped = np.clip(shifted, -128, 127).astype(np.int8)

                combined = input_scale * vl['w_scale'] * (2 ** shift)
                y_float = clipped.astype(np.float32) * combined + vl['b_float'][:vl['out_f']]

                # Reshape to spatial (real output channels). Skip-add and ReLU
                # operate in spatial layout so saved activations match what
                # consume_input_from layers will see as their input.
                y_spatial = y_float[:, :vl['real_out']].reshape(H_out, W_out, vl['real_out'])

                # Skip-add (residual): saved[X] is the post-dequant spatial
                # activation of layer X. For ResNet-8 the skip source has the
                # same H, W, C as y_spatial (downsamples handle stride/channel
                # change separately as their own VTA layer).
                if 'skip_add_from' in vl:
                    skip_h, _ = saved_activations[vl['skip_add_from']]
                    if skip_h.shape != y_spatial.shape:
                        raise ValueError(
                            f"layer {i} skip-add shape mismatch: "
                            f"main {y_spatial.shape} vs skip {skip_h.shape}")
                    y_spatial = y_spatial + skip_h

                # ReLU. Default for conv is True (preserves legacy behavior);
                # downsample 1x1 layers explicitly set apply_relu=False.
                if vl.get('apply_relu', True):
                    y_spatial = np.maximum(y_spatial, 0)

                # MaxPool
                if vl.get('pool', 0) > 0:
                    y_spatial = maxpool2d(y_spatial, vl['pool'])

                # Save if any future layer references this index
                if i in save_targets:
                    saved_activations[i] = (y_spatial, max(np.max(np.abs(y_spatial)) / 127.0, 1e-10))

                # Update chain (unless this layer is a side-branch)
                if not vl.get('branch_only', False):
                    h_float = y_spatial
                    current_scale = max(np.max(np.abs(h_float)) / 127.0, 1e-10)

            elif vl['layer_type'] == 'dense':
                # GlobalAvgPool: (H, W, C) -> (C,)
                h_vec = input_h.mean(axis=(0, 1))

                # Pad to BLOCK_IN alignment
                if len(h_vec) < vl['in_f']:
                    h_vec = np.pad(h_vec, (0, vl['in_f'] - len(h_vec)), mode='constant')

                h_int8 = np.clip(np.round(h_vec / input_scale), -128, 127).astype(np.int8)

                acc = h_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                max_abs = np.max(np.abs(acc))
                if max_abs > layer_max_acc[i]:
                    layer_max_acc[i] = max_abs

    # Compute shift amounts from accumulated max values
    for i, max_acc in enumerate(layer_max_acc):
        if max_acc <= 127:
            shift = 0
        else:
            shift = int(math.ceil(math.log2(max_acc / 127.0)))
        shift_amounts.append(shift)
        print(f"  Layer {i}: max_acc={max_acc:.0f}, shift={shift}")

    return shift_amounts, global_x_scale


# ---- CPU-side inference verification ----

def verify_cnn(vta_layers, shift_amounts, test_images, test_labels, env, num_verify=100):
    """Run CPU-side VTA-equivalent inference to verify accuracy before compilation.

    test_images: (N, H, W) for MNIST OR (N, C, H, W) for CIFAR-10. float32 [0,1].
    Legacy CNN-on-MNIST path is byte-identical to the pre-refactor logic when
    no layer carries skip metadata.
    """
    save_targets = _save_targets(vta_layers)
    correct = 0
    for img_idx in range(min(num_verify, len(test_labels))):
        img = test_images[img_idx]
        label = test_labels[img_idx]

        x_s = np.max(np.abs(img)) / 127.0 if np.max(np.abs(img)) > 0 else 1e-10
        current_scale = x_s
        h_float = _normalize_calibration_image(img)
        saved_activations = {}

        for i, (vl, shift) in enumerate(zip(vta_layers, shift_amounts)):
            if 'consume_input_from' in vl:
                input_h, input_scale = saved_activations[vl['consume_input_from']]
            else:
                input_h = h_float
                input_scale = current_scale

            if vl['layer_type'] == 'conv':
                if input_h.ndim == 2:
                    h_spatial = input_h[:, :, np.newaxis]
                else:
                    h_spatial = input_h

                patches, H_out, W_out = im2col(
                    h_spatial, vl['kernel_size'], vl['kernel_size'],
                    pad=vl['padding'], stride=vl.get('stride', 1))

                real_patch_dim = patches.shape[1]
                if real_patch_dim < vl['in_f']:
                    patches = np.pad(patches, ((0, 0), (0, vl['in_f'] - real_patch_dim)),
                                     mode='constant')

                p_int8 = np.clip(np.round(patches / input_scale), -128, 127).astype(np.int8)

                # VTA-equivalent: GEMM + shift + clip (truncating, not saturating)
                acc = p_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                shifted = acc >> shift
                clipped = shifted.astype(np.int8)  # VTA truncates

                combined = input_scale * vl['w_scale'] * (2 ** shift)
                y_float = clipped.astype(np.float32) * combined + vl['b_float'][:vl['out_f']]

                y_spatial = y_float[:, :vl['real_out']].reshape(H_out, W_out, vl['real_out'])

                if 'skip_add_from' in vl:
                    skip_h, _ = saved_activations[vl['skip_add_from']]
                    y_spatial = y_spatial + skip_h

                if vl.get('apply_relu', True):
                    y_spatial = np.maximum(y_spatial, 0)

                if vl.get('pool', 0) > 0:
                    y_spatial = maxpool2d(y_spatial, vl['pool'])

                if i in save_targets:
                    saved_activations[i] = (y_spatial, max(np.max(np.abs(y_spatial)) / 127.0, 1e-10))

                if not vl.get('branch_only', False):
                    h_float = y_spatial
                    current_scale = max(np.max(np.abs(h_float)) / 127.0, 1e-10)

            elif vl['layer_type'] == 'dense':
                h_vec = input_h.mean(axis=(0, 1))
                if len(h_vec) < vl['in_f']:
                    h_vec = np.pad(h_vec, (0, vl['in_f'] - len(h_vec)), mode='constant')

                h_int8 = np.clip(np.round(h_vec / input_scale), -128, 127).astype(np.int8)
                acc = h_int8.astype(np.int32) @ vl['W_int8'].T.astype(np.int32)
                shifted = acc >> shift
                clipped = shifted.astype(np.int8)

                combined = input_scale * vl['w_scale'] * (2 ** shift)
                y_float = clipped.astype(np.float32) * combined + vl['b_float'][:vl['out_f']]

                pred = np.argmax(y_float[:vl['real_out']])
                if pred == label:
                    correct += 1

    acc = correct / min(num_verify, len(test_labels))
    return acc


def main():
    parser = argparse.ArgumentParser(description='Export VTA CNN model for board-side inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Brevitas CNN checkpoint (.pth)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for compiled model')
    parser.add_argument('--arch', default='cnn', choices=['cnn', 'resnet8'],
                        help='Architecture family. cnn=Sequential CNN_Brevitas '
                             '(legacy MNIST flow). resnet8=MLPerf Tiny ResNet-8 '
                             'with residual blocks (CIFAR-10).')
    parser.add_argument('--size', default='tiny',
                        help='CNN size config name (tiny, small, medium, '
                             'deep_3, large). Only used for --arch cnn.')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'],
                        help='Calibration/verification dataset.')
    parser.add_argument('--mnist-dir', default='./mnist_data',
                        help='MNIST data directory (used when --dataset mnist)')
    parser.add_argument('--cifar10-dir',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'data'),
                        help='CIFAR-10 data directory (must contain '
                             'cifar-10-batches-py/test_batch).')
    parser.add_argument('--cal-samples', type=int, default=200,
                        help='Number of calibration samples')
    parser.add_argument('--verify-samples', type=int, default=500,
                        help='Number of verification samples (0 to skip)')
    parser.add_argument('--force-m1', action='store_true',
                        help='Compile each VTA module with m=1, regardless of '
                             'the layer\'s m_tiles. Weights are still tiled at '
                             'full m_tiles; the runtime loops over m_chunks. '
                             'Mirrors session 23\'s INT4-o8 transformer fix; '
                             'also needed for INT8 250 MHz CNN deploys whose '
                             'layers have m>1 AND n_chunks>1 (m>1 alone is fine '
                             'on single-call MLPs but fails on tiled-conv calls).')
    args = parser.parse_args()

    env = vta.get_env()
    print(f"VTA env: TARGET={env.TARGET}, BLOCK_IN={env.BLOCK_IN}, BLOCK_OUT={env.BLOCK_OUT}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load and prepare weights ----
    if args.arch == 'cnn':
        print(f"\nLoading Brevitas CNN checkpoint: {args.checkpoint} (size={args.size})")
        raw_layers = load_brevitas_cnn(args.checkpoint, size=args.size)
    elif args.arch == 'resnet8':
        print(f"\nLoading Brevitas ResNet-8 checkpoint: {args.checkpoint}")
        raw_layers = load_brevitas_resnet8(args.checkpoint)
    else:
        raise ValueError(f"unknown --arch {args.arch}")
    for i, l in enumerate(raw_layers):
        skip_info = ''
        if 'skip_add_from' in l:
            skip_info = f" skip_add_from={l['skip_add_from']}"
        if 'consume_input_from' in l:
            skip_info += f" consume_input_from={l['consume_input_from']}"
        if l.get('branch_only'):
            skip_info += ' branch_only'
        print(f"  Layer {i} ({l['type']}): W={l['W'].shape}, b={l['b'].shape}{skip_info}")

    print(f"\nPreparing VTA layers (pad + quantize + tile)...")
    vta_layers = prepare_vta_layers(raw_layers, env)

    # ---- Load calibration dataset ----
    if args.dataset == 'mnist':
        print(f"\nLoading MNIST for calibration...")
        mnist = download_mnist(args.mnist_dir)
        test_images = load_mnist_images(mnist['test_images'])  # (10000, 28, 28)
        test_labels = load_mnist_labels(mnist['test_labels'])
    elif args.dataset == 'cifar10':
        print(f"\nLoading CIFAR-10 test set for calibration...")
        test_images, test_labels = load_cifar10_test(args.cifar10_dir)
        print(f"  Loaded {len(test_images)} test images, shape {test_images.shape}")
    else:
        raise ValueError(f"unknown --dataset {args.dataset}")

    print(f"\nCalibrating shift amounts ({args.cal_samples} samples)...")
    shift_amounts, global_x_scale = calibrate_cnn(
        vta_layers, test_images[:args.cal_samples], env)

    # ---- CPU-side verification ----
    if args.verify_samples > 0:
        print(f"\nCPU-side verification ({args.verify_samples} samples)...")
        acc = verify_cnn(vta_layers, shift_amounts, test_images, test_labels, env,
                         num_verify=args.verify_samples)
        print(f"  VTA-equivalent accuracy: {acc:.4f} ({int(acc * args.verify_samples)}/{args.verify_samples})")
        threshold = 0.85 if args.dataset == 'mnist' else 0.65
        if acc < threshold:
            print(f"  WARNING: Accuracy below {threshold:.2f}. Check weight folding and quantization.")

    # ---- Compile VTA modules ----
    # VTA hardware limitation: when n_tiles > 1, the maximum o dimension is ~96.
    # For larger o (e.g. conv2 with 196 output pixels), we tile o into chunks
    # and call the module multiple times at inference. The module is compiled
    # with o_tile, and the inference code loops over o_total/o_tile chunks.
    MAX_O_WHEN_N_GT1 = 64  # safe margin below empirical ~96 limit

    print(f"\nCompiling VTA modules...")
    module_filenames = []
    # Dynamic spatial tracker for the chain h_float (NOT branch-only side
    # outputs). For legacy CNN paths every conv participates in the chain;
    # for ResNet-8 we also track per-saved-layer spatial dims so that
    # consume_input_from can recover the right (H, W) for downstream layers.
    if args.dataset == 'mnist':
        cur_h, cur_w = 28, 28
    elif args.dataset == 'cifar10':
        cur_h, cur_w = 32, 32
    else:
        raise ValueError(f"unknown --dataset {args.dataset}")
    saved_spatial = {}  # layer_idx -> (H, W) of that layer's output
    for i, vl in enumerate(vta_layers):
        shift = shift_amounts[i]

        if vl['layer_type'] == 'conv':
            # Resolve input spatial dims (chain or fork)
            if 'consume_input_from' in vl:
                in_h, in_w = saved_spatial[vl['consume_input_from']]
            else:
                in_h, in_w = cur_h, cur_w
            stride = vl.get('stride', 1)
            kH = vl['kernel_size']
            pad = vl['padding']
            out_h = (in_h + 2 * pad - kH) // stride + 1
            out_w = (in_w + 2 * pad - kH) // stride + 1
            o_total = out_h * out_w
            # MaxPool divides spatial dims (legacy CNN); ResNet-8 has no pools.
            if vl.get('pool', 0) > 0:
                out_h //= vl['pool']
                out_w //= vl['pool']
            # Save this layer's output spatial dims for any future consumer.
            saved_spatial[i] = (out_h, out_w)
            # Update chain unless side-branch.
            if not vl.get('branch_only', False):
                cur_h, cur_w = out_h, out_w
        else:
            o_total = 1                       # dense

        n_t = vl['n_tiles']
        m_t = vl['m_tiles']
        # Compile-time m: 1 if --force-m1 (runtime loops m_t m-chunks).
        m_compiled = 1 if args.force_m1 else m_t

        # Determine o_tile: tile o when n>1 and o exceeds hardware limit
        if n_t > 1 and o_total > MAX_O_WHEN_N_GT1:
            # Find largest divisor of o_total that's <= MAX_O_WHEN_N_GT1
            o_tile = None
            for candidate in range(MAX_O_WHEN_N_GT1, 0, -1):
                if o_total % candidate == 0:
                    o_tile = candidate
                    break
            assert o_tile is not None
            n_chunks = o_total // o_tile
        else:
            o_tile = o_total
            n_chunks = 1

        # Filename encodes the COMPILED tile shape so different builds with
        # the same model but different --force-m1 settings don't collide.
        fname = f"layer{i}_o{o_tile}_n{n_t}_m{m_compiled}_s{shift}.o"
        m_log = f"m={m_compiled}(of {m_t})" if m_compiled != m_t else f"m={m_t}"
        print(f"  Layer {i} ({vl['layer_type']}, o_total={o_total}, o_tile={o_tile}, "
              f"chunks={n_chunks}, n={n_t}, {m_log}, shift={shift})...",
              end=" ", flush=True)

        mod = compile_gemm_with_shift(env, o_tile, n_t, m_compiled, shift)
        out_path = os.path.join(args.output_dir, fname)
        mod.save(out_path)
        module_filenames.append(fname)
        vl['o_total'] = o_total
        vl['o_tile'] = o_tile
        vl['n_chunks'] = n_chunks
        print(f"OK -> {fname}")

    # ---- Save weights ----
    print(f"\nSaving weights...")
    for i, vl in enumerate(vta_layers):
        np.save(os.path.join(args.output_dir, f'W{i}_tiled.npy'), vl['W_tiled'])
        np.save(os.path.join(args.output_dir, f'b{i}.npy'), vl['b_float'])
        print(f"  W{i}_tiled.npy: {vl['W_tiled'].shape}, b{i}.npy: {vl['b_float'].shape}")

    # ---- Save config ----
    # Preserve the legacy `architecture` string for arch=cnn size=tiny dataset=mnist
    # so byte-exact regression against the existing
    # tvm-v0.12.0/vta_export/cnn_mnist_tiny/ holds. Other variants get
    # arch+size+dataset-derived tags.
    if args.arch == 'cnn' and args.dataset == 'mnist' and args.size == 'tiny':
        config = {
            'model_type': 'cnn',
            'architecture': 'cnn_tiny_8_16_mnist',
            'input_shape': [1, 28, 28],
        }
    elif args.arch == 'cnn':
        config = {
            'model_type': 'cnn',
            'architecture': f'cnn_{args.size}_{args.dataset}',
            'size': args.size,
            'input_shape': [1, 28, 28] if args.dataset == 'mnist' else [3, 32, 32],
        }
    elif args.arch == 'resnet8':
        # ResNet-8 uses the CNN runner path on board (vta_infer.c is_cnn,
        # benchmark.py is_cnn). The residual structure is carried per-layer
        # in skip_add_from / consume_input_from / branch_only / apply_relu.
        # `architecture` retains the family label for human-readable logs.
        config = {
            'model_type': 'cnn',
            'architecture': f'resnet8_{args.dataset}',
            'input_shape': [3, 32, 32],
        }
    else:
        raise ValueError(f"unknown --arch {args.arch}")
    config.update({
        'num_layers': len(vta_layers),
        'layers': [],
        'global_x_scale': float(global_x_scale),
        'vta_config': {
            'BATCH': env.BATCH,
            'BLOCK_IN': env.BLOCK_IN,
            'BLOCK_OUT': env.BLOCK_OUT,
        },
        'bitstream': '1x16_i8w8a32_15_15_18_17.bit',
        'calibration_samples': args.cal_samples,
    })

    for i, vl in enumerate(vta_layers):
        # m_compiled is the m used at TVM/HLS compile time. m_tiles is the
        # number of m-chunks the runtime must loop over. Legacy: equal, and
        # the m_compiled field is OMITTED to keep config.json bit-identical
        # with pre-refactor outputs (regression on cnn-tiny-mnist holds).
        m_compiled = 1 if args.force_m1 else vl['m_tiles']
        layer_config = {
            'index': i,
            'type': vl['layer_type'],
            'in_f': vl['in_f'],
            'out_f': vl['out_f'],
            'real_in': vl['real_in'],
            'real_out': vl['real_out'],
            'n_tiles': vl['n_tiles'],
            'm_tiles': vl['m_tiles'],
            'o_total': vl['o_total'],
            'o_tile': vl['o_tile'],
            'n_chunks': vl['n_chunks'],
            'shift': shift_amounts[i],
            'w_scale': vl['w_scale'],
            'module_file': module_filenames[i],
            'weight_file': f'W{i}_tiled.npy',
            'bias_file': f'b{i}.npy',
        }
        if vl['layer_type'] == 'conv':
            layer_config['kernel_size'] = vl['kernel_size']
            layer_config['padding'] = vl['padding']
            layer_config['in_channels'] = vl['in_channels']
            layer_config['out_channels'] = vl['out_channels']
            layer_config['pool'] = vl.get('pool', 0)
            if 'stride' in vl:
                layer_config['stride'] = vl['stride']
        # Skip metadata (resnet8 path; absent in legacy cnn flow so config
        # bytes for cnn-tiny-mnist remain identical).
        for k in ('consume_input_from', 'skip_add_from', 'branch_only',
                  'apply_relu'):
            if k in vl:
                layer_config[k] = vl[k]
        # Emit m_compiled only when it differs from m_tiles (i.e., --force-m1
        # AND m_tiles > 1). Legacy CNN configs without --force-m1 do NOT
        # carry this field — preserves bit-identical regression.
        if m_compiled != vl['m_tiles']:
            layer_config['m_compiled'] = m_compiled
        config['layers'].append(layer_config)

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Export complete: {args.output_dir}/")
    print(f"  Model: CNN tiny [8, 16] MNIST")
    print(f"  Layers: {len(vta_layers)} ({sum(1 for v in vta_layers if v['layer_type']=='conv')} conv, "
          f"{sum(1 for v in vta_layers if v['layer_type']=='dense')} dense)")
    print(f"  Shift amounts: {shift_amounts}")
    for i, vl in enumerate(vta_layers):
        tile_info = f"o_tile={vl['o_tile']}" if vl['n_chunks'] == 1 else f"o_total={vl['o_total']} o_tile={vl['o_tile']} chunks={vl['n_chunks']}"
        print(f"    Layer {i} ({vl['layer_type']}): "
              f"{tile_info} n={vl['n_tiles']} m={vl['m_tiles']} shift={shift_amounts[i]}")
    print(f"  Files:")
    for f_name in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f_name))
        print(f"    {f_name} ({size} bytes)")
    print(f"\nCopy to board:")
    basename = os.path.basename(args.output_dir.rstrip('/'))
    print(f"  scp -r {args.output_dir} xilinx@192.168.3.1:/home/xilinx/models/vta/{basename}/")
    print(f"\nOn board, link .o -> .so:")
    print(f"  cd /home/xilinx/models/vta/{basename}/")
    for fname in module_filenames:
        so_name = fname.replace('.o', '.so')
        print(f"  gcc -shared -o {so_name} {fname} -ltvm_runtime")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
