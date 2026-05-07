#!/usr/bin/env python3
"""prepare_cifar10_for_board.py — convert CIFAR-10 test set to board-compatible binary.

The vta_infer.c CNN path expects HWC layout (im2col reads x[si*W*C + sj*C + c]),
so this script transposes torchvision's CHW pickle to HWC and writes a flat
uint8 binary. Mirror of MNIST's IDX format philosophy: uint8 source, /255 in
the C runtime.

Output files (default in --out-dir):
  cifar10_test_images.bin  — 10000 × 32 × 32 × 3 uint8, HWC, big-endian-agnostic
                             (each image is a contiguous (H, W, C) record)
  cifar10_test_labels.bin  — 10000 × 1 uint8 labels

Source: torchvision-extracted batches at <data-dir>/cifar-10-batches-py/test_batch
        (Python pickle dict: b'data' is (N, 3072) uint8 CHW row-major)
"""
import argparse
import os
import pickle
import numpy as np


def load_cifar10_test(data_dir):
    test_path = os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"CIFAR-10 test_batch not found at {test_path}")
    with open(test_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    raw = batch[b'data']  # (10000, 3072) uint8, CHW row-major (R then G then B)
    labels = np.array(batch[b'labels'], dtype=np.uint8)
    images_chw = raw.reshape(-1, 3, 32, 32)
    return images_chw, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./data',
                    help='Dir containing cifar-10-batches-py/')
    ap.add_argument('--out-dir', default='./cifar10_data',
                    help='Output dir for the binary files')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading CIFAR-10 test set from {args.data_dir}/cifar-10-batches-py/")
    images_chw, labels = load_cifar10_test(args.data_dir)
    n = images_chw.shape[0]
    print(f"  {n} images, CHW shape {images_chw.shape}")
    print(f"  labels range [{labels.min()}, {labels.max()}]")

    # Transpose CHW → HWC so the board's im2col reads correctly.
    # torchvision/numpy: (N, C, H, W) → (N, H, W, C)
    images_hwc = np.transpose(images_chw, (0, 2, 3, 1))
    print(f"  transposed to HWC: {images_hwc.shape}")
    assert images_hwc.dtype == np.uint8

    # Write contiguous bytes. C runtime will read as flat (H*W*C) per image.
    images_hwc = np.ascontiguousarray(images_hwc, dtype=np.uint8)

    img_path = os.path.join(args.out_dir, 'cifar10_test_images.bin')
    lbl_path = os.path.join(args.out_dir, 'cifar10_test_labels.bin')
    images_hwc.tofile(img_path)
    labels.tofile(lbl_path)
    print(f"  wrote {img_path} ({images_hwc.nbytes:,} bytes)")
    print(f"  wrote {lbl_path} ({labels.nbytes:,} bytes)")

    # Self-check: round-trip the first image and verify it matches the
    # torchvision-loaded test set image #0 after the same CHW→HWC transpose.
    raw_bytes = np.fromfile(img_path, dtype=np.uint8, count=32*32*3)
    img0_from_file = raw_bytes.reshape(32, 32, 3)
    img0_from_pickle = images_hwc[0]
    assert np.array_equal(img0_from_file, img0_from_pickle), "round-trip mismatch"
    print("  round-trip self-check OK (img[0] matches)")

    # Also dump per-image-shape metadata for any consumers that want it.
    meta_path = os.path.join(args.out_dir, 'cifar10_test_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"n_images={n}\n")
        f.write("layout=HWC\n")
        f.write("dtype=uint8\n")
        f.write("shape_per_image=32,32,3\n")
        f.write("byte_stride=3072\n")
    print(f"  wrote {meta_path}")


if __name__ == '__main__':
    main()
