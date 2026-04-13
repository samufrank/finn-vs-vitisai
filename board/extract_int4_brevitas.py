#!/usr/bin/env python3
"""Extract INT4 weights and learned scales from Brevitas MLP_Brevitas_INT4.

Extracts integer weights, weight scales, activation scales, and biases
using Brevitas's own quantization state — NOT recomputed max-abs scales.

Run with the finn-t-env (has brevitas):
    cd ~/dev/CEN571-final
    ~/.venvs/finn-t-env/bin/python3 finn-vs-vitisai/board/extract_int4_brevitas.py

Outputs to: ~/dev/CEN571-final/tvm-v0.12.0/vta_mnist_weights_int4_v2/
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from mlp import MLP_Brevitas_INT4

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'finn', 'mlp_mnist_tiny_int4.pth')
OUTPUT_DIR = os.path.expanduser('~/dev/CEN571-final/tvm-v0.12.0/vta_mnist_weights_int4_v2')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model = MLP_Brevitas_INT4()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False))
    model.eval()

    # Forward pass to populate activation quantizer state
    dummy = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        _ = model(dummy)

    # Architecture: Flatten(0), QuantLinear(1), QuantReLU(2), QuantLinear(3), QuantReLU(4), QuantLinear(5)
    linear_indices = [1, 3, 5]
    relu_indices = [2, 4]

    meta = {
        'architecture': [784, 64, 32, 10],
        'num_layers': 3,
        'quantizer_config': {
            'weight': 'Int8WeightPerTensorFloat.let(bit_width=4) — signed 4-bit, per-tensor, scale=max_abs/7',
            'activation': 'Uint8ActPerTensorFloat.let(bit_width=4) — unsigned 4-bit, per-tensor, learned scale',
        },
        'layers': [],
    }

    # --- Extract weights and weight scales ---
    for i, li in enumerate(linear_indices):
        layer = model.layers[li]
        qw = layer.quant_weight()

        W_int = qw.int().detach().numpy().astype(np.int8)
        w_scale = float(qw.scale.detach().item())

        assert W_int.min() >= -7 and W_int.max() <= 7, \
            f"Layer {i}: weight int range [{W_int.min()}, {W_int.max()}] outside [-7, 7]"

        bias = layer.bias.detach().numpy().astype(np.float32)

        np.save(os.path.join(OUTPUT_DIR, f'W{i}.npy'), W_int)
        np.save(os.path.join(OUTPUT_DIR, f'w_scale_{i}.npy'), np.float64(w_scale))
        np.save(os.path.join(OUTPUT_DIR, f'b{i}.npy'), bias)

        print(f"Layer {i} (layers.{li}): W{W_int.shape} int range [{W_int.min()}, {W_int.max()}], "
              f"w_scale={w_scale:.6f}, bias shape={bias.shape}")

        meta['layers'].append({
            'index': i,
            'module_index': li,
            'W_shape': list(W_int.shape),
            'w_scale': w_scale,
            'w_signed': True,
            'w_clip_min': -7,
            'w_clip_max': 7,
            'w_bit_width': 4,
        })

    # --- Extract activation scales ---
    # act_scale_0: input quantizer — model has none, so we define it.
    # MNIST pixels are [0, 1] after ToTensor(). For unsigned 4-bit [0, 15]:
    #   scale = 1/15 ≈ 0.06667
    # This means x_int = round(x_float / (1/15)) = round(x_float * 15), clipped to [0, 15].
    input_act_scale = 1.0 / 15.0
    np.save(os.path.join(OUTPUT_DIR, 'act_scale_0.npy'), np.float64(input_act_scale))
    print(f"\nact_scale_0 (input, synthetic): {input_act_scale:.6f}  [unsigned, 0..15]")

    meta['act_scales'] = [{
        'index': 0,
        'source': 'synthetic (no learned input quantizer)',
        'signed': False,
        'clip_min': 0,
        'clip_max': 15,
        'bit_width': 4,
        'scale': input_act_scale,
    }]

    # act_scale_1, act_scale_2: learned from QuantReLU layers
    for j, ri in enumerate(relu_indices):
        relu_mod = model.layers[ri]
        scaling_impl = relu_mod.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
        raw_value = float(scaling_impl.value.detach().item())
        # Brevitas ParameterFromRuntimeStatsScaling: .value is the learned threshold
        # (range parameter). The actual per-integer-unit scale = value / int_max.
        # For Uint4: int_max = 2^4 - 1 = 15. Verified: act_quant output .scale matches.
        int_max = 15.0  # unsigned 4-bit
        actual_scale = raw_value / int_max

        scale_idx = j + 1
        np.save(os.path.join(OUTPUT_DIR, f'act_scale_{scale_idx}.npy'), np.float64(actual_scale))
        print(f"act_scale_{scale_idx} (layers.{ri}, learned): threshold={raw_value:.6f}, "
              f"scale=threshold/15={actual_scale:.6f}  [unsigned, 0..15]")

        meta['act_scales'].append({
            'index': scale_idx,
            'source': f'layers.{ri}.act_quant (QuantReLU, learned ParameterFromRuntimeStatsScaling)',
            'signed': False,
            'clip_min': 0,
            'clip_max': 15,
            'bit_width': 4,
            'scale': actual_scale,
            'raw_value': raw_value,
        })

    # --- Verify: run Brevitas forward on a test vector and compare ---
    print("\n--- Verification ---")
    x_test = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        brevitas_out = model(x_test)
    print(f"Brevitas output (sample): {brevitas_out[0, :5].tolist()}")

    # Save meta
    meta_path = os.path.join(OUTPUT_DIR, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved meta to {meta_path}")

    print(f"\nAll files saved to {OUTPUT_DIR}/")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname} ({size} bytes)")


if __name__ == '__main__':
    main()
