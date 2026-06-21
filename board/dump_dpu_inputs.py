#!/usr/bin/env python3
"""Generate preprocessed float32 NHWC input dumps for the DPU C runner.

Host-side de-risk of the DPU input path. Uses the SAME preprocessing as
run_dpu_benchmark in benchmark.py: the load_mnist / load_cifar10 loaders are
imported directly, and the CHW->HWC transpose mirrors run_dpu_benchmark's
nested _to_dpu_layout() (cast float32 + transpose to NHWC; the loaders already
return NCHW normalized /255).

Writes per-dataset to dpu_data/:
  <name>_test_f32_nhwc.bin    float32, C-contiguous, N*H*W*C, NHWC
  <name>_test_labels.bin      uint8, N
  <name>_test_f32_nhwc.json   sidecar: count,H,W,C,dtype,layout (C runner reads
                              shape without guessing)

A round-trip check reads each binary back and asserts exact float equality with
the in-memory tensor for the first 20 images. On failure it exits non-zero (the
C runner depends on this binary being byte-exact).

This does NOT run benchmark.py's benchmark; it only imports its data loaders.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import load_mnist, load_cifar10  # SAME loaders as the board path

DATA = '/home/samu/dev/CEN571-final/dpu_data'


def to_nhwc(images):
    """Mirror run_dpu_benchmark._to_dpu_layout: float32 + NCHW->NHWC transpose.

    load_mnist/load_cifar10 already return NCHW float32 in [0, 1]. The board
    code transposes (C, H, W) -> (H, W, C) per image; here we do it vectorized
    over the batch. No quantization/normalization (the DPU does the float->fix).
    """
    imgs = images.astype(np.float32)
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
        imgs = np.transpose(imgs, (0, 2, 3, 1))   # NCHW -> NHWC
    return np.ascontiguousarray(imgs)


def dump(name, images, labels):
    nhwc = to_nhwc(images)
    N, H, W, C = nhwc.shape
    bin_path = os.path.join(DATA, f'{name}_test_f32_nhwc.bin')
    lbl_path = os.path.join(DATA, f'{name}_test_labels.bin')
    json_path = os.path.join(DATA, f'{name}_test_f32_nhwc.json')

    nhwc.astype('<f4').tofile(bin_path)            # little-endian float32
    labels.astype(np.uint8).tofile(lbl_path)
    meta = {
        'count': int(N), 'H': int(H), 'W': int(W), 'C': int(C),
        'dtype': 'float32', 'layout': 'NHWC',
        'per_image_elems': int(H * W * C),
        'bin': os.path.basename(bin_path),
        'labels': os.path.basename(lbl_path),
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Round-trip: read back, reshape, exact float equality on first 20 images.
    rb = np.fromfile(bin_path, dtype='<f4').reshape(N, H, W, C)
    ok = np.array_equal(rb[:20], nhwc[:20])
    print(f'  {name:8s} {N}x{H}x{W}x{C}  {os.path.getsize(bin_path)} bytes  '
          f'-> {os.path.basename(bin_path)}  round-trip(first 20)='
          f'{"PASS" if ok else "FAIL"}')
    return ok


def main():
    os.makedirs(DATA, exist_ok=True)
    print(f"Loading datasets from {DATA} ...")
    mnist_imgs, mnist_lbls = load_mnist(path=DATA)
    cifar_imgs, cifar_lbls = load_cifar10(path=os.path.join(DATA, 'test_batch'))

    print("Dumping NHWC float32 tensors + sidecars:")
    ok_mnist = dump('mnist', mnist_imgs, mnist_lbls)
    ok_cifar10 = dump('cifar10', cifar_imgs, cifar_lbls)

    if not (ok_mnist and ok_cifar10):
        print("ROUND-TRIP FAILED — stop. The C runner depends on this binary.")
        sys.exit(1)
    print("All round-trip checks PASS.")


if __name__ == '__main__':
    main()
