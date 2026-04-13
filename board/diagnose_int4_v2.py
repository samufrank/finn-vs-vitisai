#!/usr/bin/env python3
"""Layer-by-layer VTA INT4 v2 diagnostic for board-side debugging.

Compares actual VTA hardware output against the host-side reference trace
(verification_trace.npz) to localize which layer diverges.

Run on the PYNQ board as root:
    cd /home/xilinx
    python3 diagnose_int4_v2.py 2>&1 | tee diag_int4_v2.log

Expects:
    /home/xilinx/models/vta/mlp_mnist_int4_qat_v2/  (exported model)
    /home/xilinx/models/vta/mlp_mnist_int4_qat_v2/verification_trace.npz
"""
import sys
import os
import json
import ctypes
import numpy as np


# ============================================================
# INT4 nibble packing/unpacking for VTA
# ============================================================

def pack_int4_for_vta(vals_int8):
    """Pack int8 array of int4 values into VTA nibble format (flat contiguous)."""
    vals = np.asarray(vals_int8, dtype=np.int8)
    flat = vals.flatten()
    n = len(flat)
    lo = flat[0::2].view(np.uint8) & 0xF
    hi = flat[1::2].view(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[:n // 2] = packed
    return out.reshape(vals.shape)


def unpack_int4_from_vta(packed_int8):
    """Unpack VTA nibble-packed int4 output to one-value-per-element int8."""
    raw = np.asarray(packed_int8, dtype=np.int8)
    flat = raw.flatten()
    n = len(flat)
    packed_bytes = flat[:n // 2].view(np.uint8)
    lo = (packed_bytes & 0xF).astype(np.int8)
    hi = ((packed_bytes >> 4) & 0xF).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.int8)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.int8)
    out = np.zeros(n, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out.reshape(raw.shape)


# ---- Board-side paths (hard-coded) ----
MODEL_DIR = '/home/xilinx/models/vta/mlp_mnist_int4_qat_v2/'
TRACE_PATH = os.path.join(MODEL_DIR, 'verification_trace.npz')

# ============================================================
# Setup: bitstream, VTA runtime, TVM
# ============================================================

def setup_vta():
    """Load bitstream + VTA runtime. Returns nothing; side-effects only."""
    config_path = os.path.join(MODEL_DIR, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # Clear stale PYNQ state
    stale = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
    try:
        if os.path.exists(stale):
            os.remove(stale)
    except Exception:
        pass

    # Load bitstream
    bitstream_name = config.get('bitstream', '1x16_i4w4a32_15_14_17_17.bit')
    bitstream_path = None
    for candidate in [
        f'/root/.vta_cache/ultra96/0_0_2/{bitstream_name}',
        f'/home/xilinx/.vta_cache/ultra96/0_0_2/{bitstream_name}',
        os.path.join(MODEL_DIR, bitstream_name),
    ]:
        if os.path.exists(candidate):
            bitstream_path = candidate
            break

    if bitstream_path is None:
        print(f"ERROR: Bitstream {bitstream_name} not found")
        sys.exit(1)

    print(f"Loading bitstream: {bitstream_path}")
    try:
        from pynq import Overlay
        overlay = Overlay(bitstream_path)
        print(f"  Overlay loaded, IPs: {list(overlay.ip_dict.keys())}")
    except Exception as e:
        print(f"  Overlay load failed: {e}")
        print(f"  Continuing (bitstream may already be loaded)...")

    # Load VTA runtime
    vta_lib = None
    for candidate in [
        '/home/xilinx/tvm-src/build/libvta.so',
        os.path.join(os.environ.get('TVM_HOME', ''), 'build/libvta.so'),
    ]:
        if os.path.exists(candidate):
            vta_lib = candidate
            break
    if vta_lib is None:
        print("ERROR: libvta.so not found")
        sys.exit(1)
    print(f"Loading VTA runtime: {vta_lib}")
    ctypes.CDLL(vta_lib, ctypes.RTLD_GLOBAL)


# ============================================================
# Model loading (replicates benchmark.py infer_one_vta_native)
# ============================================================

def load_model():
    """Load config, VTA modules, weights, biases. Returns all state."""
    import tvm
    import tvm.runtime
    import vta

    config_path = os.path.join(MODEL_DIR, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    env = vta.get_env()
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT
    ctx = tvm.device("ext_dev", 0)

    num_layers = config['num_layers']
    print(f"Architecture: {config['architecture']}")
    print(f"Layers: {num_layers}, requant_mode: {config.get('requant_mode')}")
    print(f"VTA env: BATCH={env.BATCH}, BLOCK_IN={BLOCK_IN}, BLOCK_OUT={BLOCK_OUT}")

    gemm_modules = []
    W_nds = []
    D_nds = []      # bias VTA tensors (int32) for has_vta_bias layers, else None
    bias_data = []   # raw loaded bias arrays (int32 or float32)
    layer_info = []

    for lc in config['layers']:
        mod_file = lc['module_file']
        so_file = mod_file.replace('.o', '.so') if mod_file.endswith('.o') else mod_file
        mod_path = os.path.join(MODEL_DIR, so_file)
        if not os.path.exists(mod_path):
            mod_path = os.path.join(MODEL_DIR, mod_file)

        print(f"  Layer {lc['index']}: {mod_path}")
        f = tvm.runtime.load_module(mod_path)
        gemm_modules.append(f)

        # Weights -> VTA (pack int4 nibbles)
        W_tiled = np.load(os.path.join(MODEL_DIR, lc['weight_file']))
        W_tiled = pack_int4_for_vta(W_tiled)
        W_nds.append(tvm.nd.array(W_tiled, ctx))

        # Bias
        has_vta_bias = lc.get('has_vta_bias', False)
        b_raw = np.load(os.path.join(MODEL_DIR, lc['bias_file']))
        bias_data.append(b_raw)

        if has_vta_bias:
            D_nds.append(tvm.nd.array(b_raw.astype(np.int32), ctx))
        else:
            D_nds.append(None)

        layer_info.append(lc)

    # Pre-allocate A and C buffers
    A_nds = []
    C_nds = []
    for lc in config['layers']:
        A_nds.append(tvm.nd.array(
            np.zeros((1, lc['n_tiles'], 1, BLOCK_IN), dtype=np.int8), ctx))
        C_nds.append(tvm.nd.array(
            np.zeros((1, lc['m_tiles'], 1, BLOCK_OUT), dtype=np.int8), ctx))

    return {
        'config': config,
        'gemm_modules': gemm_modules,
        'W_nds': W_nds,
        'D_nds': D_nds,
        'A_nds': A_nds,
        'C_nds': C_nds,
        'bias_data': bias_data,
        'layer_info': layer_info,
        'ctx': ctx,
        'BLOCK_IN': BLOCK_IN,
        'BLOCK_OUT': BLOCK_OUT,
    }


# ============================================================
# Diagnostics
# ============================================================

def run_diagnostics(model, trace):
    config = model['config']
    num_layers = config['num_layers']
    BLOCK_IN = model['BLOCK_IN']
    BLOCK_OUT = model['BLOCK_OUT']

    print()
    print("=" * 70)
    print("DIAGNOSTIC A: Bias DMA round-trip verification")
    print("=" * 70)

    for i, lc in enumerate(model['layer_info']):
        if not lc.get('has_vta_bias', False):
            print(f"  Layer {i}: has_vta_bias=false (CPU bias), skipping DMA check")
            continue

        # Read bias back from VTA device
        d_nd = model['D_nds'][i]
        d_readback = d_nd.numpy().flatten()

        # Expected: the .npy file as loaded
        b_expected = model['bias_data'][i].flatten().astype(np.int32)

        match = np.array_equal(d_readback, b_expected)
        max_diff = int(np.max(np.abs(d_readback.astype(np.int64) - b_expected.astype(np.int64))))
        print(f"  Layer {i}: DMA round-trip {'MATCH' if match else 'MISMATCH'} "
              f"(max_diff={max_diff})")
        print(f"    Expected [:16]: {b_expected[:16].tolist()}")
        print(f"    Readback [:16]: {d_readback[:16].tolist()}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC B: Input quantization verification (image 0)")
    print("=" * 70)

    img0 = trace['images_float'][0]
    input_scale = config['input_scale']
    input_clip_max = config['input_clip_max']
    board_input = np.clip(np.round(img0 / input_scale),
                          0, input_clip_max).astype(np.int8)
    trace_input = trace['img0_input_int']

    input_match = np.array_equal(board_input, trace_input)
    input_diff = int(np.max(np.abs(board_input.astype(np.int16) - trace_input.astype(np.int16))))
    print(f"  Input quant {'MATCH' if input_match else 'MISMATCH'} "
          f"(max_diff={input_diff})")
    # Show a non-zero region (MNIST has many leading zeros)
    nz = np.nonzero(board_input)[0]
    if len(nz) > 0:
        start = max(0, nz[0])
        print(f"    Board [{start}:{start+16}]: {board_input[start:start+16].tolist()}")
        print(f"    Trace [{start}:{start+16}]: {trace_input[start:start+16].tolist()}")
    else:
        print(f"    Board [:16]: {board_input[:16].tolist()}")
        print(f"    Trace [:16]: {trace_input[:16].tolist()}")

    print()
    print("=" * 70)
    print("LAYER-BY-LAYER INFERENCE: 10 trace images")
    print("=" * 70)

    first_diverged_layer = None
    board_preds = []
    board_correct = 0

    # Collect per-index divergence data for each layer
    # Key: layer index, Value: dict mapping output index -> count of images where it diverged
    layer_div_indices = {li: {} for li in range(num_layers)}
    layer_div_values = {li: [] for li in range(num_layers)}  # (img, idx, board_val, trace_val)

    for img_idx in range(10):
        img = trace['images_float'][img_idx]
        true_label = int(trace['labels'][img_idx])
        mode_d_pred = int(trace['mode_d_preds'][img_idx])

        print(f"\n--- Image {img_idx} (true={true_label}, mode_d={mode_d_pred}) ---")

        # Quantize input
        h_int8 = np.clip(np.round(img / input_scale),
                         0, input_clip_max).astype(np.int8)

        # Check input matches trace
        trace_x = trace[f'img{img_idx}_input_int']
        if not np.array_equal(h_int8, trace_x):
            print(f"  INPUT MISMATCH (max_diff="
                  f"{int(np.max(np.abs(h_int8.astype(np.int16) - trace_x.astype(np.int16))))})")

        for li, lc in enumerate(model['layer_info']):
            has_vta_bias = lc.get('has_vta_bias', False)
            n_tiles = lc['n_tiles']
            m_tiles = lc['m_tiles']
            in_f = lc['in_f']
            out_f = lc['out_f']
            real_out = lc['real_out']

            # Pad input if needed
            if len(h_int8) < in_f:
                padded = np.zeros(in_f, dtype=np.int8)
                padded[:len(h_int8)] = h_int8
                h_int8 = padded

            # --- Check x_int matches trace ---
            trace_x_int = trace[f'img{img_idx}_layer{li}_x_int']
            x_match = np.array_equal(h_int8[:len(trace_x_int)], trace_x_int)
            if not x_match:
                x_diff = int(np.max(np.abs(
                    h_int8[:len(trace_x_int)].astype(np.int16) -
                    trace_x_int.astype(np.int16))))
                n_mismatch = int(np.sum(h_int8[:len(trace_x_int)] != trace_x_int))
                print(f"  Layer {li} x_int: MISMATCH "
                      f"(max_diff={x_diff}, mismatched={n_mismatch}/{len(trace_x_int)})")

            # --- Run VTA (pack input, unpack output) ---
            x_packed = pack_int4_for_vta(h_int8[:in_f])
            x_tiled = x_packed.reshape(1, n_tiles, 1, BLOCK_IN)
            model['A_nds'][li].copyfrom(x_tiled)
            model['C_nds'][li].copyfrom(
                np.zeros((1, m_tiles, 1, BLOCK_OUT), dtype=np.int8))

            if has_vta_bias:
                # 4-arg: A, W, D, C
                model['gemm_modules'][li](
                    model['A_nds'][li], model['W_nds'][li],
                    model['D_nds'][li], model['C_nds'][li])
            else:
                # 3-arg: A, W, C
                model['gemm_modules'][li](
                    model['A_nds'][li], model['W_nds'][li],
                    model['C_nds'][li])

            vta_out_packed = model['C_nds'][li].numpy().flatten()
            vta_out = unpack_int4_from_vta(vta_out_packed)

            if has_vta_bias:
                # Hidden layer: compare VTA output against trace's clipped
                trace_clipped = trace[f'img{img_idx}_layer{li}_clipped']
                board_clipped = vta_out[:real_out]

                max_abs_diff = int(np.max(np.abs(
                    board_clipped.astype(np.int16) -
                    trace_clipped.astype(np.int16))))
                n_mismatch = int(np.sum(board_clipped != trace_clipped))
                total = len(trace_clipped)

                status = "OK" if max_abs_diff == 0 else "DIVERGED"
                print(f"  Layer {li}: max_abs_diff={max_abs_diff}, "
                      f"mismatched={n_mismatch}/{total}  [{status}]")

                if max_abs_diff > 0:
                    if first_diverged_layer is None:
                        first_diverged_layer = li
                    # Show details
                    diff_idx = np.where(board_clipped != trace_clipped)[0]
                    for di in diff_idx[:5]:
                        print(f"    [{di}] board={board_clipped[di]} "
                              f"trace={trace_clipped[di]}")

                    # Record per-index divergence for heatmap
                    for di in diff_idx:
                        layer_div_indices[li][int(di)] = \
                            layer_div_indices[li].get(int(di), 0) + 1
                        layer_div_values[li].append(
                            (img_idx, int(di),
                             int(board_clipped[di]), int(trace_clipped[di])))

                    # Also show intermediate: what WOULD numpy compute?
                    # Re-derive acc from trace to help isolate
                    trace_acc = trace[f'img{img_idx}_layer{li}_bias_added']
                    trace_shifted = trace[f'img{img_idx}_layer{li}_shifted']
                    print(f"    Trace acc+bias [{diff_idx[0]}]={trace_acc[diff_idx[0]]}, "
                          f"shifted={trace_shifted[diff_idx[0]]}, "
                          f"clipped={trace_clipped[diff_idx[0]]}")

                # Pass VTA output to next layer
                h_int8 = vta_out[:real_out].copy()

            else:
                # Last layer: unpack already sign-extended, dequant, float bias, argmax
                vta_signed = vta_out[:out_f]

                combined = lc['in_scale'] * lc['w_scale'] * (2 ** lc['shift'])
                b_float = model['bias_data'][li]
                y_float = (vta_signed[:real_out].astype(np.float32)
                           * combined + b_float[:real_out])
                board_pred = int(np.argmax(y_float))

                # Compare raw VTA output against what trace expects
                # Trace has acc_raw (no shift/clip for last layer in Mode D)
                # but we DO shift/clip on board, so compare the shifted version
                # by computing it from trace acc_raw
                trace_acc_raw = trace[f'img{img_idx}_layer{li}_acc_raw']
                trace_bias_added = trace[f'img{img_idx}_layer{li}_bias_added']
                trace_pred = int(trace[f'img{img_idx}_layer{li}_pred'])

                print(f"  Layer {li} (last): board_pred={board_pred}, "
                      f"trace_pred={trace_pred}, true={true_label}")
                print(f"    VTA packed[:8]: {vta_out_packed[:8].tolist()}")
                print(f"    VTA sign [:10]: {vta_signed[:real_out].tolist()}")
                print(f"    y_float  [:10]: "
                      f"{[f'{v:.2f}' for v in y_float[:real_out].tolist()]}")
                print(f"    Trace acc+bias: {trace_bias_added[:real_out].tolist()}")
                print(f"    Trace argmax:   {trace_pred} "
                      f"(of {trace_bias_added[:real_out].tolist()})")

                if board_pred != trace_pred:
                    if first_diverged_layer is None:
                        first_diverged_layer = li
                    print(f"    *** PREDICTION MISMATCH ***")

                board_preds.append(board_pred)
                if board_pred == true_label:
                    board_correct += 1

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Board accuracy on 10 trace images: {board_correct}/10")
    print(f"Board predictions: {board_preds}")
    print(f"Mode D predictions: {trace['mode_d_preds'].tolist()}")
    print(f"True labels:        {trace['labels'].tolist()}")

    pred_agree = sum(1 for a, b in zip(board_preds, trace['mode_d_preds'])
                     if a == b)
    print(f"Board vs Mode D agreement: {pred_agree}/10")

    if first_diverged_layer is not None:
        print(f"\nFirst diverging layer across all 10 images: layer {first_diverged_layer}")
    else:
        print(f"\nNo divergence detected — all layers bit-exact with trace.")

    # ---- Diagnostic C: per-index divergence heatmap ----
    for li in range(num_layers):
        div_map = layer_div_indices[li]
        if not div_map:
            continue

        out_size = config['layers'][li]['real_out']
        total_diverged = sum(div_map.values())
        n_indices_hit = len(div_map)

        print()
        print("=" * 70)
        print(f"DIAGNOSTIC C: Layer {li} per-index divergence heatmap "
              f"({out_size} outputs)")
        print("=" * 70)
        print(f"  Total divergent elements: {total_diverged} "
              f"across 10 images")
        print(f"  Distinct indices hit:     {n_indices_hit} / {out_size}")
        if n_indices_hit > 0:
            avg_per_idx = total_diverged / n_indices_hit
            print(f"  Avg divergences/index:    {avg_per_idx:.1f}")

        # Sorted by count descending
        sorted_idx = sorted(div_map.items(), key=lambda x: -x[1])
        print(f"\n  {'Index':>5}  {'Count':>5}  {'Bar'}")
        print(f"  {'-----':>5}  {'-----':>5}  {'---'}")
        for idx, count in sorted_idx:
            bar = '#' * count
            print(f"  {idx:>5}  {count:>5}  {bar}")

        # Interpretation
        print()
        if n_indices_hit < 20 and total_diverged > 20:
            print(f"  >>> STRUCTURAL: {n_indices_hit} indices account for "
                  f"{total_diverged} divergences.")
            print(f"      Likely cause: specific output neurons have "
                  f"accumulator values near shift/clip boundaries,")
            print(f"      or specific weight/bias tiles have DMA/timing "
                  f"issues.")
        elif n_indices_hit >= out_size * 0.6:
            print(f"  >>> COMPUTATIONAL: divergence spread across "
                  f"{n_indices_hit}/{out_size} indices.")
            print(f"      Likely cause: systematic error in GEMM, bias "
                  f"DMA, or shift amount — not index-specific.")
        else:
            print(f"  >>> MIXED: {n_indices_hit} indices hit, neither "
                  f"clearly structural nor purely computational.")

        # Show board vs trace value distribution for diverged elements
        vals = layer_div_values[li]
        if vals:
            board_vals = [v[2] for v in vals]
            trace_vals = [v[3] for v in vals]
            diffs = [v[2] - v[3] for v in vals]
            print(f"\n  Diverged value statistics:")
            print(f"    Board values:  min={min(board_vals)}, "
                  f"max={max(board_vals)}, "
                  f"mean={np.mean(board_vals):.1f}")
            print(f"    Trace values:  min={min(trace_vals)}, "
                  f"max={max(trace_vals)}, "
                  f"mean={np.mean(trace_vals):.1f}")
            print(f"    Diffs (b-t):   min={min(diffs)}, "
                  f"max={max(diffs)}, "
                  f"mean={np.mean(diffs):.1f}, "
                  f"all-same-sign={'yes' if all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs) else 'no'}")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("VTA INT4 v2 Layer-by-Layer Diagnostic")
    print("=" * 70)
    print(f"Model dir: {MODEL_DIR}")
    print(f"Trace:     {TRACE_PATH}")

    # Verify files exist
    if not os.path.isdir(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        sys.exit(1)
    if not os.path.exists(TRACE_PATH):
        print(f"ERROR: Verification trace not found: {TRACE_PATH}")
        sys.exit(1)

    # Load trace
    print("\nLoading verification trace...")
    trace = dict(np.load(TRACE_PATH))
    n_images = trace['images_float'].shape[0]
    print(f"  {n_images} trace images, "
          f"labels={trace['labels'].tolist()}, "
          f"mode_d_preds={trace['mode_d_preds'].tolist()}")

    # Setup VTA hardware
    print()
    setup_vta()

    # Load model
    print("\nLoading model...")
    model = load_model()

    # Run diagnostics
    run_diagnostics(model, trace)


if __name__ == '__main__':
    main()
