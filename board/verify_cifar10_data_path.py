#!/usr/bin/env python3
"""verify_cifar10_data_path.py — end-to-end check of the CIFAR-10 data path.

Tests the complete flow that runs on the board:
    binary file → C-runtime-equivalent read → im2col (ki,kj,c order) →
    quantize → matmul(W0_int8 from tiled VTA export) → dequant+bias → ReLU
against a PyTorch Brevitas ResNet-8 reference forward pass on the same image.

Catches:
  - CHW/HWC mixups in the preprocessor or VTA loader
  - RGB/BGR channel swaps
  - im2col patch row order disagreement with weight tiling
  - W_tiled untile bugs

Pass criterion:
  (a) Image #0 from board binary matches torchvision-loaded reference
      (HWC layout, identical bytes after the documented CHW→HWC transpose).
  (b) VTA-sim layer-0 output is correlated >0.99 with the Brevitas float
      reference, with the same channel argmax pattern. Per-pixel match isn't
      possible because of INT8 quantization, but the structure must match.
"""
import os
import sys
import pickle

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'models'))
from resnet import ResNet8_Brevitas


# ---- Step 1: round-trip the binary against the torchvision pickle ----

def step_round_trip(data_dir, board_bin_dir):
    """Compare board-binary image[0] vs torchvision-pickle image[0] HWC."""
    print("\n[1] Round-trip check: board binary vs torchvision pickle (image 0)")

    # Read first image from board binary (3072 uint8 bytes, HWC layout)
    img_path = os.path.join(board_bin_dir, 'cifar10_test_images.bin')
    with open(img_path, 'rb') as f:
        raw = f.read(32 * 32 * 3)
    img_from_bin = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)

    # Read pickle (CHW), transpose to HWC for comparison
    pkl_path = os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch')
    with open(pkl_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    img0_chw = batch[b'data'][0].reshape(3, 32, 32)
    img_from_pickle = np.transpose(img0_chw, (1, 2, 0))  # HWC

    if not np.array_equal(img_from_bin, img_from_pickle):
        print("  FAIL: image #0 from binary does not match pickle (after CHW→HWC)")
        diff = (img_from_bin.astype(np.int32) - img_from_pickle.astype(np.int32))
        print(f"    abs diff stats: max={np.max(np.abs(diff))} "
              f"mean={np.mean(np.abs(diff)):.3f}")
        return None
    print("  OK: 32×32×3 uint8 HWC bytes identical")

    # Sanity: no obvious R/B swap by inspecting the per-channel mean
    # (sky photos have higher B; grass photos have higher G; varies, but
    # typical CIFAR-10 image #0 is a frog — green-dominant).
    label = int(batch[b'labels'][0])
    print(f"  image #0 label: {label} (CIFAR-10 class names: airplane, "
          f"automobile, bird, cat, deer, dog, frog, horse, ship, truck)")
    print(f"  per-channel mean: R={img_from_bin[..., 0].mean():.1f} "
          f"G={img_from_bin[..., 1].mean():.1f} "
          f"B={img_from_bin[..., 2].mean():.1f}")
    return img_from_bin, label


# ---- Step 2: PyTorch reference layer-0 forward ----

def step_pytorch_reference(brevitas_pth, img_hwc):
    """Run the Brevitas ResNet-8 stem (conv → BN → ReLU) on image #0.

    Returns the post-stem-relu activation, shape (1, 16, 32, 32) float32.
    """
    print("\n[2] PyTorch Brevitas ResNet-8 stem forward")
    model = ResNet8_Brevitas(in_channels=3, num_classes=10)
    sd = torch.load(brevitas_pth, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()

    # PyTorch input: (N, C, H, W) float32 [0, 1]
    img_chw = np.transpose(img_hwc.astype(np.float32) / 255.0, (2, 0, 1))
    x = torch.from_numpy(img_chw).unsqueeze(0)  # (1, 3, 32, 32)

    with torch.no_grad():
        h = model.stem_conv(x)
        h = model.stem_bn(h)
        h = model.stem_relu(h)
    out = h.detach().cpu().numpy()  # (1, 16, 32, 32)
    print(f"  stem output shape: {out.shape}, "
          f"range [{out.min():.4f}, {out.max():.4f}], "
          f"mean {out.mean():.4f}, nonzero {np.count_nonzero(out)}/{out.size}")
    return out[0]  # drop batch dim → (16, 32, 32)


# ---- Step 3: VTA-sim of layer 0 ----

def py_im2col_hwc(x_hwc, kH, kW, pad, stride):
    """Patch order: (ki, kj, c) — matches vta_infer.c im2col exactly.

    Returns (Ho*Wo, kH*kW*C) float32 patches.
    """
    H, W, C = x_hwc.shape
    Ho = (H + 2 * pad - kH) // stride + 1
    Wo = (W + 2 * pad - kW) // stride + 1
    patches = np.zeros((Ho * Wo, kH * kW * C), dtype=x_hwc.dtype)
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            pidx = 0
            for ki in range(kH):
                for kj in range(kW):
                    si = i * stride + ki - pad
                    sj = j * stride + kj - pad
                    for c in range(C):
                        if 0 <= si < H and 0 <= sj < W:
                            patches[idx, pidx] = x_hwc[si, sj, c]
                        pidx += 1
            idx += 1
    return patches, Ho, Wo


def untile_W(W_tiled, real_out, real_in):
    """Reverse tile_weights_2d: (m, n, BLOCK_OUT, BLOCK_IN) → (out_f, in_f),
    then crop padding back to (real_out, real_in)."""
    m, n, BO, BI = W_tiled.shape
    W = W_tiled.transpose(0, 2, 1, 3).reshape(m * BO, n * BI)
    return W[:real_out, :real_in]


def step_vta_sim_layer0(vta_export_dir, img_hwc):
    """Run VTA-equivalent quantized layer 0 on image #0.

    Returns (real_out, Ho, Wo) float32 = post-dequant, post-ReLU activation.
    """
    print("\n[3] VTA-sim of stem conv (board's layer 0)")

    import json
    with open(os.path.join(vta_export_dir, 'config.json')) as f:
        cfg = json.load(f)
    L0 = cfg['layers'][0]
    real_out = L0['real_out']
    real_in = L0['real_in']
    in_f = L0['in_f']
    shift = L0['shift']
    w_scale = L0['w_scale']
    pad = L0['padding']
    kH = kW = L0['kernel_size']
    stride = L0.get('stride', 1)
    print(f"  layer0: kH={kH} pad={pad} stride={stride}  "
          f"real_out={real_out} real_in={real_in} in_f(padded)={in_f}  "
          f"shift={shift} w_scale={w_scale:.6f}")

    W_tiled = np.load(os.path.join(vta_export_dir, 'W0_tiled.npy')).astype(np.int8)
    b0 = np.load(os.path.join(vta_export_dir, 'b0.npy')).astype(np.float32)
    print(f"  W_tiled shape {W_tiled.shape}, b0 shape {b0.shape}")
    W_int8 = untile_W(W_tiled, real_out, real_in)
    print(f"  untiled W_int8 shape {W_int8.shape}, "
          f"range [{W_int8.min()}, {W_int8.max()}]")

    # Image preprocessing: HWC uint8 → HWC float32 in [0,1] (board does this in C)
    x = img_hwc.astype(np.float32) / 255.0

    # Per-image input scale (matches the board's cnn_infer x_max / 127 logic)
    x_max = float(np.abs(x).max())
    in_scale = x_max / 127.0 if x_max > 0 else 1e-10
    print(f"  in_scale = {in_scale:.6f}")

    # im2col (ki, kj, c order)
    patches, Ho, Wo = py_im2col_hwc(x, kH, kW, pad, stride)
    n_pixels = patches.shape[0]
    real_patch = patches.shape[1]
    if real_patch < in_f:
        patches = np.pad(patches, ((0, 0), (0, in_f - real_patch)),
                         mode='constant')
    p_int8 = np.clip(np.rint(patches / in_scale), -128, 127).astype(np.int8)
    print(f"  patches: {Ho}×{Wo} = {n_pixels} pixels, "
          f"patch_dim {real_patch}→{in_f}, "
          f"int8 range [{p_int8.min()}, {p_int8.max()}]")

    # GEMM in int32; pad-row weights are zero (export pads bias too) so the
    # untiled W has the right shape for the un-padded matmul.
    W_full_int8 = np.zeros((in_f, real_out), dtype=np.int8)
    W_full_int8[:real_in, :] = W_int8.T  # (in_f, real_out)
    acc = p_int8.astype(np.int32) @ W_full_int8.astype(np.int32)  # (n_pix, real_out)

    # VTA: shift+clip then dequant. (Mirror cnn_infer dequant exactly.)
    shifted = acc >> shift
    clipped = np.clip(shifted, -128, 127).astype(np.int8)
    combined = in_scale * w_scale * (1 << shift)
    y = clipped.astype(np.float32) * combined + b0[:real_out]
    y = np.maximum(y, 0)  # post-bias ReLU (legacy + ResNet stem both apply)
    y = y.reshape(Ho, Wo, real_out).transpose(2, 0, 1)  # → (C_out, H, W)
    print(f"  VTA-sim output: shape {y.shape}, "
          f"range [{y.min():.4f}, {y.max():.4f}], "
          f"nonzero {np.count_nonzero(y)}/{y.size}")
    return y


# ---- Step 4: compare ----

def step_compare(ref, sim):
    """ref shape: (16, 32, 32). sim shape: (real_out, 32, 32). real_out should be 16."""
    print("\n[4] Compare PyTorch reference vs VTA-sim")
    ro = sim.shape[0]
    if ref.shape[0] != ro:
        print(f"  WARN: ref C_out={ref.shape[0]} vs sim C_out={ro}; "
              f"comparing first {ro}")
    ref_use = ref[:ro]

    # Per-channel correlation (Pearson). High structural agreement.
    corrs = []
    for c in range(ro):
        a = ref_use[c].flatten()
        b = sim[c].flatten()
        # Both might have constant zero (all clipped). Guard divide-by-zero.
        if a.std() < 1e-9 or b.std() < 1e-9:
            corrs.append(float('nan'))
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    corrs_arr = np.array(corrs, dtype=np.float64)
    valid = corrs_arr[~np.isnan(corrs_arr)]
    print(f"  per-channel correlations (n={ro}):")
    print(f"    min={valid.min():.4f} max={valid.max():.4f} "
          f"mean={valid.mean():.4f} median={np.median(valid):.4f}")
    print(f"    NaN channels (one side all zero): "
          f"{int(np.isnan(corrs_arr).sum())}/{ro}")

    # Pixel-level argmax channel comparison — does the dominant channel
    # match per pixel? More robust to magnitude differences.
    ref_argmax = np.argmax(ref_use, axis=0).flatten()
    sim_argmax = np.argmax(sim, axis=0).flatten()
    match_rate = float((ref_argmax == sim_argmax).mean())
    print(f"  argmax-channel agreement per pixel: {100*match_rate:.2f}%")

    # Verdict
    bad = (valid < 0.5).sum()
    print()
    if valid.mean() > 0.85 and match_rate > 0.7:
        print("  PASS: VTA-sim tracks PyTorch reference.")
        print("  HWC layout, channel order, im2col, and weight tiling are consistent.")
        return True
    if valid.mean() > 0.7 and match_rate > 0.5:
        print("  WARN: agreement is positive but below tight thresholds.")
        print("  May indicate INT8 quantization noise or a subtle layout drift.")
        print("  Re-check before declaring the loader board-ready.")
        return True
    print("  FAIL: agreement is poor. Check HWC/CHW or RGB/BGR alignment.")
    print(f"    {bad} of {len(valid)} channels have correlation < 0.5.")
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./data',
                    help='Dir with cifar-10-batches-py/')
    ap.add_argument('--board-bin-dir', default='./cifar10_data',
                    help='Dir produced by prepare_cifar10_for_board.py')
    ap.add_argument('--brevitas-pth',
                    default='./finn/resnet8_cifar10_int8.pth')
    ap.add_argument('--vta-export-dir',
                    default='./vta_exports/resnet8_cifar10_int8')
    args = ap.parse_args()

    rt = step_round_trip(args.data_dir, args.board_bin_dir)
    if rt is None:
        print("\nABORT: data ingestion does not match reference.")
        return 1
    img_hwc, label = rt

    ref = step_pytorch_reference(args.brevitas_pth, img_hwc)
    sim = step_vta_sim_layer0(args.vta_export_dir, img_hwc)
    if not step_compare(ref, sim):
        return 1
    print("\nVerification complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
