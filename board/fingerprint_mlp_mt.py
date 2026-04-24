#!/usr/bin/env python3
"""Fingerprint script: measure prediction-vector difference between
   strict (x > t) and inclusive (x >= t) MultiThreshold semantics for
   the FINN MLP CPU pre-stage.

   For each MNIST test image, computes both act_gt = sum(x > thres) and
   act_ge = sum(x >= thres), runs the FPGA on each, records both
   prediction vectors and accuracies. One board pass = the same evidence
   as a strict before/after sequence, without patch-and-revert.

   Usage:
     python3 fingerprint_mlp_mt.py \
         --deploy /home/xilinx/models/finn/mlp_mnist_tiny/deploy \
         --tag    int8 \
         --output /tmp/fingerprint_int8

     python3 fingerprint_mlp_mt.py \
         --deploy /home/xilinx/models/finn/mlp_mnist_tiny_int4/deploy \
         --tag    int4 \
         --output /tmp/fingerprint_int4
"""

import argparse
import gzip
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np


DEFAULT_MNIST_PATHS = ['/home/xilinx/MNIST/raw', '/home/petalinux/MNIST/raw']


def load_mnist(path=None):
    if path is None:
        for p in DEFAULT_MNIST_PATHS:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            sys.exit(f'ERROR: no MNIST in {DEFAULT_MNIST_PATHS}; pass --mnist <path>')

    def load_images(p):
        with gzip.open(p, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    def load_labels(p):
        with gzip.open(p, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    return (load_images(f'{path}/t10k-images-idx3-ubyte.gz'),
            load_labels(f'{path}/t10k-labels-idx1-ubyte.gz'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deploy', required=True,
                    help='Deploy dir containing driver/, bitfile/, and mlp_*.npy')
    ap.add_argument('--tag', required=True,
                    help='Short tag (e.g. int8, int4) for the output JSON')
    ap.add_argument('--output', default=None,
                    help='Output basename; writes <output>.npz and <output>.json')
    ap.add_argument('--mnist', default=None,
                    help='MNIST raw dir override')
    ap.add_argument('--limit', type=int, default=0,
                    help='If >0, only run on first N test images (for smoke testing)')
    args = ap.parse_args()

    driver_dir = os.path.join(args.deploy, 'driver')
    bitfile = os.path.join(args.deploy, 'bitfile', 'finn-accel.bit')
    sys.path.insert(0, driver_dir)
    from driver import io_shape_dict
    from driver_base import FINNExampleOverlay

    images_u8, labels = load_mnist(args.mnist)
    if args.limit > 0:
        images_u8 = images_u8[:args.limit]
        labels = labels[:args.limit]
    images_flat = images_u8.reshape(len(images_u8), 784)
    print(f'MNIST: {len(images_u8)} images, label distribution: '
          f'{np.bincount(labels).tolist()}')

    W0 = np.load(os.path.join(args.deploy, 'mlp_MatMul_0_param0.npy')).astype(np.float32)
    thres = np.load(os.path.join(args.deploy, 'mlp_MultiThreshold_0_param0.npy')).astype(np.float32)
    mul_out = np.load(os.path.join(args.deploy, 'mlp_Mul_0_param0.npy')).astype(np.float32)
    add_out = np.load(os.path.join(args.deploy, 'mlp_Add_0_param0.npy')).astype(np.float32)
    print(f'Weights: W0 {W0.shape}, thres {thres.shape}, '
          f'mul {mul_out.shape}, add {add_out.shape}')

    ishape_normal = io_shape_dict['ishape_normal'][0]
    print(f'Loading bitfile: {bitfile}')
    ol = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform='zynq-iodma',
        io_shape_dict=io_shape_dict,
        batch_size=1,
        runtime_weight_dir=os.path.join(driver_dir, 'runtime_weights'),
    )

    N = len(images_flat)
    pred_gt = np.zeros(N, dtype=np.int32)
    pred_ge = np.zeros(N, dtype=np.int32)
    tied_image_count = 0
    tied_channel_count = 0

    t0 = time.time()
    for i in range(N):
        acc_pre = (images_flat[i].astype(np.float32) / 255.0) @ W0       # (64,)
        act_gt = np.sum(acc_pre[:, None] > thres, axis=-1).astype(np.uint8)
        act_ge = np.sum(acc_pre[:, None] >= thres, axis=-1).astype(np.uint8)
        diff_chans = int(np.sum(act_gt != act_ge))
        if diff_chans:
            tied_image_count += 1
            tied_channel_count += diff_chans

        hw_gt = ol.execute([act_gt.reshape(ishape_normal)]).flatten().astype(np.float32)
        out_gt = hw_gt * mul_out + add_out
        pred_gt[i] = int(np.argmax(out_gt))

        hw_ge = ol.execute([act_ge.reshape(ishape_normal)]).flatten().astype(np.float32)
        out_ge = hw_ge * mul_out + add_out
        pred_ge[i] = int(np.argmax(out_ge))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) * 2 / elapsed
            print(f'  {i+1}/{N}   {elapsed:.1f}s   {rate:.1f} fpga-calls/s')

    elapsed = time.time() - t0

    correct_gt = int(np.sum(pred_gt == labels))
    correct_ge = int(np.sum(pred_ge == labels))
    n_diff = int(np.sum(pred_gt != pred_ge))
    diff_indices = np.where(pred_gt != pred_ge)[0].tolist()

    summary = {
        'tag': args.tag,
        'deploy': args.deploy,
        'n_images': N,
        'elapsed_seconds': round(elapsed, 2),
        'accuracy_gt': correct_gt / N,
        'accuracy_ge': correct_ge / N,
        'correct_gt': correct_gt,
        'correct_ge': correct_ge,
        'n_pred_diff': n_diff,
        'diff_indices_first50': diff_indices[:50],
        'tied_image_count': tied_image_count,
        'tied_channel_count': tied_channel_count,
        'pred_gt_sha1': hashlib.sha1(pred_gt.tobytes()).hexdigest(),
        'pred_ge_sha1': hashlib.sha1(pred_ge.tobytes()).hexdigest(),
    }
    print('\n===== FINGERPRINT SUMMARY =====')
    print(json.dumps(summary, indent=2))

    if args.output:
        np.savez(args.output, pred_gt=pred_gt, pred_ge=pred_ge, labels=labels[:N])
        with open(args.output + '.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved {args.output}.npz and {args.output}.json')


if __name__ == '__main__':
    main()
