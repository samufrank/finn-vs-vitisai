#!/usr/bin/env python3
"""extract_brevitas_weights.py — extract model weights from a Brevitas .pth.

Walks the Brevitas Sequential dynamically (any layer count) and dumps
per-layer .npy files in a format compatible with the VTA export scripts.

Output formats per (model, precision):

  MLP INT8: W{i}.npy + b{i}.npy (float32). Consumed by export_vta_model.py
            via --weights-dir + --architecture.

  MLP INT4: W{i}.npy (int8 in [-7, 7]) + w_scale_{i}.npy + b{i}.npy +
            act_scale_{j}.npy (j=0 synthetic, j=1.. learned from QuantReLU)
            + meta.json. Format matches extract_int4_brevitas.py and is
            consumed by export_vta_model_int4_v2.py.

  CNN INT8: W{i}.npy (BN-folded conv weights, reshaped to (C_out, kH*kW*C_in)
            float32 for im2col GEMM) + b{i}.npy (folded float32) +
            meta.json with per-layer geometry. Consumed by a generalized
            export_vta_cnn.py (Work Item 3).

  CNN INT4: NOT IMPLEMENTED. session 22 explored several (perchan, nobn,
            wide) and that work is in board/archive/cnn_int4_investigation/.
            Out of scope for this extractor.

Usage:
  python3 board/extract_brevitas_weights.py \\
    --model mlp --size tiny --precision 8 \\
    --checkpoint finn/mlp_mnist_tiny.pth

  Output dir defaults to vta_exports/<model>_<size>_int{8,4}/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent          # finn-vs-vitisai/
sys.path.insert(0, str(REPO_ROOT / 'models'))


# ---- helpers ---------------------------------------------------------------

def import_brevitas_classes():
    """Imported lazily so the CNN INT4 path's missing imports don't fail."""
    from mlp import MLP_Brevitas, MLP_Brevitas_INT4, get_mlp_config
    from cnn import (CNN_Brevitas, CNN_Brevitas_INT4, CNN_Brevitas_INT4_PerChan,
                     CNN_Brevitas_INT4_NoBN, CNN_Brevitas_INT4_NoBN_Wide,
                     get_cnn_config)
    import brevitas.nn as qnn
    return {
        'MLP_Brevitas': MLP_Brevitas, 'MLP_Brevitas_INT4': MLP_Brevitas_INT4,
        'CNN_Brevitas': CNN_Brevitas, 'CNN_Brevitas_INT4': CNN_Brevitas_INT4,
        'get_mlp_config': get_mlp_config, 'get_cnn_config': get_cnn_config,
        'qnn': qnn,
    }


def construct_model(model_kind, size, precision, classes):
    """Instantiate the matching Brevitas class with the right size config."""
    if model_kind == 'mlp':
        hidden = classes['get_mlp_config'](size)
        cls = classes['MLP_Brevitas_INT4'] if precision == 4 else classes['MLP_Brevitas']
        return cls(input_size=784, hidden_sizes=hidden), hidden
    else:
        channels = classes['get_cnn_config'](size)
        cls = classes['CNN_Brevitas_INT4'] if precision == 4 else classes['CNN_Brevitas']
        return cls(in_channels=1, channels=channels), channels


def fold_bn_into_conv(conv_w, bn_w, bn_b, bn_m, bn_v, eps):
    """Fold BatchNorm into preceding Conv weights. See export_vta_cnn.py."""
    C_out = conv_w.shape[0]
    scale = bn_w / np.sqrt(bn_v + eps)
    W_folded = conv_w * scale.reshape(C_out, 1, 1, 1)
    b_folded = -bn_m * scale + bn_b
    return W_folded, b_folded


# ---- MLP extraction --------------------------------------------------------

def extract_mlp_int8(model, output_dir, classes):
    qnn = classes['qnn']
    linear_mods = [(i, m) for i, m in enumerate(model.layers)
                   if isinstance(m, qnn.QuantLinear)]
    if not linear_mods:
        raise RuntimeError('No QuantLinear layers found in model.layers')

    arch = []
    for layer_idx, (_mod_idx, layer) in enumerate(linear_mods):
        W = layer.weight.detach().numpy().astype(np.float32)
        b = layer.bias.detach().numpy().astype(np.float32)
        np.save(output_dir / f'W{layer_idx}.npy', W)
        np.save(output_dir / f'b{layer_idx}.npy', b)
        if not arch:
            arch.append(int(W.shape[1]))
        arch.append(int(W.shape[0]))
        print(f'  Layer {layer_idx}: W{tuple(W.shape)} b{tuple(b.shape)} (float32)')

    meta = {
        'model_type': 'mlp',
        'precision': 8,
        'architecture': arch,
        'num_layers': len(linear_mods),
        'format': 'float32 weights for export_vta_model.py',
    }
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def extract_mlp_int4(model, output_dir, classes):
    qnn = classes['qnn']
    linear_mods = [(i, m) for i, m in enumerate(model.layers)
                   if isinstance(m, qnn.QuantLinear)]
    relu_mods = [(i, m) for i, m in enumerate(model.layers)
                 if isinstance(m, qnn.QuantReLU)]

    # Forward pass populates act-quant state (lazy Brevitas Parameters).
    dummy = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        _ = model(dummy)

    arch = []
    meta = {
        'model_type':       'mlp',
        'precision':        4,
        'num_layers':       len(linear_mods),
        'quantizer_config': {
            'weight':     'Int8WeightPerTensorFloat.let(bit_width=4) — signed 4-bit per-tensor',
            'activation': 'Uint8ActPerTensorFloat.let(bit_width=4) — unsigned 4-bit per-tensor',
        },
        'layers':     [],
        'act_scales': [],
        'format':     'matches extract_int4_brevitas.py for export_vta_model_int4_v2.py',
    }

    for layer_idx, (mod_idx, layer) in enumerate(linear_mods):
        qw = layer.quant_weight()
        W_int = qw.int().detach().numpy().astype(np.int8)
        w_scale = float(qw.scale.detach().item())
        assert W_int.min() >= -7 and W_int.max() <= 7, \
            f'Layer {layer_idx}: int weight range [{W_int.min()},{W_int.max()}] outside [-7,7]'
        b = layer.bias.detach().numpy().astype(np.float32)

        np.save(output_dir / f'W{layer_idx}.npy', W_int)
        np.save(output_dir / f'w_scale_{layer_idx}.npy', np.float64(w_scale))
        np.save(output_dir / f'b{layer_idx}.npy', b)

        if not arch:
            arch.append(int(W_int.shape[1]))
        arch.append(int(W_int.shape[0]))

        meta['layers'].append({
            'index': layer_idx, 'module_index': mod_idx,
            'W_shape': list(W_int.shape), 'w_scale': w_scale,
            'w_signed': True, 'w_clip_min': -7, 'w_clip_max': 7,
            'w_bit_width': 4,
        })
        print(f'  Layer {layer_idx} (layers.{mod_idx}): W_int{tuple(W_int.shape)} '
              f'range [{W_int.min()},{W_int.max()}], w_scale={w_scale:.6f}')

    # act_scale_0: synthetic input scale for MNIST [0,1] mapped to [0,15].
    input_act_scale = 1.0 / 15.0
    np.save(output_dir / 'act_scale_0.npy', np.float64(input_act_scale))
    meta['act_scales'].append({
        'index': 0, 'source': 'synthetic (no learned input quantizer)',
        'signed': False, 'clip_min': 0, 'clip_max': 15,
        'bit_width': 4, 'scale': input_act_scale,
    })
    print(f'  act_scale_0 (synthetic input): {input_act_scale:.6f}')

    # act_scale_1..n: learned scales from QuantReLU layers.
    for j, (ri, relu_mod) in enumerate(relu_mods):
        scaling_impl = relu_mod.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
        raw_value = float(scaling_impl.value.detach().item())
        actual_scale = raw_value / 15.0   # int_max for unsigned 4-bit
        scale_idx = j + 1
        np.save(output_dir / f'act_scale_{scale_idx}.npy', np.float64(actual_scale))
        meta['act_scales'].append({
            'index': scale_idx,
            'source': f'layers.{ri}.act_quant',
            'signed': False, 'clip_min': 0, 'clip_max': 15,
            'bit_width': 4, 'scale': actual_scale, 'raw_value': raw_value,
        })
        print(f'  act_scale_{scale_idx} (layers.{ri}, learned): scale={actual_scale:.6f}')

    meta['architecture'] = arch
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


# ---- CNN extraction --------------------------------------------------------

def extract_cnn_int8(model, output_dir, classes):
    qnn = classes['qnn']

    # Walk features for Conv-BN-ReLU-MaxPool blocks.
    feats = list(model.features)
    conv_layer_specs = []
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
                    pool = feats[j].kernel_size
                    if isinstance(pool, tuple):
                        pool = pool[0]
                    break
                j += 1
            conv_layer_specs.append({'conv': conv, 'bn': bn, 'pool': int(pool)})
        i += 1

    if not conv_layer_specs:
        raise RuntimeError('No QuantConv2d layers found in model.features')

    # Classifier (single QuantLinear in model.classifier).
    cls_lins = [m for m in model.classifier if isinstance(m, qnn.QuantLinear)]
    if len(cls_lins) != 1:
        raise RuntimeError(f'expected exactly one classifier QuantLinear; got {len(cls_lins)}')
    cls = cls_lins[0]

    meta = {
        'model_type':      'cnn',
        'precision':       8,
        'num_conv_layers': len(conv_layer_specs),
        'has_classifier':  True,
        'input_shape':     [1, 28, 28],
        'layers':          [],
        'format':          'BN-folded float32 weights, reshaped (C_out, kH*kW*C_in) for VTA im2col GEMM',
    }

    for layer_idx, spec in enumerate(conv_layer_specs):
        conv = spec['conv']
        bn = spec['bn']
        pool = spec['pool']

        conv_w = conv.weight.detach().numpy().astype(np.float32)
        if bn is not None:
            bn_w = bn.weight.detach().numpy().astype(np.float32)
            bn_b = bn.bias.detach().numpy().astype(np.float32)
            bn_m = bn.running_mean.detach().numpy().astype(np.float32)
            bn_v = bn.running_var.detach().numpy().astype(np.float32)
            W_folded, b_folded = fold_bn_into_conv(conv_w, bn_w, bn_b, bn_m, bn_v, bn.eps)
        else:
            W_folded = conv_w
            b_folded = (conv.bias.detach().numpy().astype(np.float32)
                        if conv.bias is not None
                        else np.zeros(W_folded.shape[0], dtype=np.float32))

        # Match export_vta_cnn.py reshape: (C_out, C_in, kH, kW) ->
        # transpose to (C_out, kH, kW, C_in) then flatten -> (C_out, kH*kW*C_in).
        # The HWC im2col patches are in (kH, kW, C_in) order, so weights must too.
        C_out, C_in, kH, kW = W_folded.shape
        W_2d = W_folded.transpose(0, 2, 3, 1).reshape(C_out, -1)

        np.save(output_dir / f'W{layer_idx}.npy', W_2d.astype(np.float32))
        np.save(output_dir / f'b{layer_idx}.npy', b_folded.astype(np.float32))

        # padding from QuantConv2d (typically (1, 1) for 3x3 same).
        pad = conv.padding
        if isinstance(pad, tuple):
            pad = pad[0]

        meta['layers'].append({
            'index': layer_idx, 'type': 'conv',
            'kernel_size': kH, 'padding': int(pad),
            'in_channels': C_in, 'out_channels': C_out,
            'pool': pool,
            'W_shape': list(W_2d.shape),
            'b_shape': list(b_folded.shape),
        })
        print(f'  Conv layer {layer_idx}: W{tuple(W_2d.shape)} b{tuple(b_folded.shape)} '
              f'(C_in={C_in}, C_out={C_out}, k={kH}, pad={pad}, pool={pool}, '
              f'BN-folded={bn is not None})')

    # Classifier
    cls_idx = len(conv_layer_specs)
    W_cls = cls.weight.detach().numpy().astype(np.float32)
    b_cls = cls.bias.detach().numpy().astype(np.float32)
    np.save(output_dir / f'W{cls_idx}.npy', W_cls)
    np.save(output_dir / f'b{cls_idx}.npy', b_cls)
    meta['layers'].append({
        'index': cls_idx, 'type': 'dense',
        'in_features': int(W_cls.shape[1]), 'out_features': int(W_cls.shape[0]),
        'W_shape': list(W_cls.shape), 'b_shape': list(b_cls.shape),
    })
    print(f'  Dense layer {cls_idx}: W{tuple(W_cls.shape)} b{tuple(b_cls.shape)}')

    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',      required=True, choices=['mlp', 'cnn'])
    ap.add_argument('--size',       required=True,
                    help='size config name (e.g. tiny, small, deep_3)')
    ap.add_argument('--precision',  required=True, type=int, choices=[8, 4])
    ap.add_argument('--checkpoint', required=True, help='Brevitas .pth path')
    ap.add_argument('--output-dir', default=None,
                    help='Default: vta_exports/<model>_<size>_int{8,4}/')
    args = ap.parse_args()

    if args.model == 'cnn' and args.precision == 4:
        print('ERROR: CNN INT4 extraction is not implemented in this script. '
              'See board/archive/cnn_int4_investigation/ for the session-22 '
              'PerChan/NoBN variants.', file=sys.stderr)
        return 2

    classes = import_brevitas_classes()
    model, _arch = construct_model(args.model, args.size, args.precision, classes)

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()

    if args.output_dir is None:
        out = (REPO_ROOT.parent / 'finn-vs-vitisai' / 'vta_exports'
               / f'{args.model}_{args.size}_int{args.precision}')
    else:
        out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f'Extracting {args.model.upper()} {args.size} INT{args.precision} '
          f'from {args.checkpoint} -> {out}')
    if args.model == 'mlp':
        if args.precision == 8:
            extract_mlp_int8(model, out, classes)
        else:
            extract_mlp_int4(model, out, classes)
    else:
        extract_cnn_int8(model, out, classes)

    print(f'\nFiles in {out}:')
    for f in sorted(out.iterdir()):
        print(f'  {f.name} ({f.stat().st_size} bytes)')


if __name__ == '__main__':
    sys.exit(main() or 0)
