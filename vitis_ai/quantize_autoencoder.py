"""Vitis AI PTQ for the canonical FC autoencoder (ToyCar anomaly detection).

Runs INSIDE the Vitis AI PyTorch Docker. Loads a pre-trained float .pth
from the host, calibrates with standardized ToyCar log-mel features,
evaluates per-recording AUC-ROC for both float and quantized models,
exports xmodel for the DPU target.

This is a separate script from train_and_quantize.py because:
  * autoencoder has no labels (eval metric is AUC, not classification accuracy)
  * loss/eval are MSE on the standardized 640-dim window, not CE on logits
  * dataset is preprocessed .npy features, not torchvision

Usage (inside Docker, after the standard mount + conda-activate dance):
    cd /workspace/project/vitis_ai
    python quantize_autoencoder.py \
        --pth autoencoder_toycar_canonical_float.pth \
        --prep-dir /workspace/project/data/toycar_preprocessed \
        --calib-size 1000 \
        --target DPUCZDX8G_ISA1_B512 \
        --output-dir quantize_result_autoencoder

Then compile xmodel:
    vai_c_xir -x quantize_result_autoencoder/FCAutoencoder_int.xmodel \
              -a arch_zu3_b512.json \
              -o compiled_autoencoder \
              -n autoencoder_toycar_canonical
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pytorch_nndct.apis import torch_quantizer

# Mounted host repo lives at /workspace/project/. models/ is a sibling of vitis_ai/.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'models'))
from autoencoder import FCAutoencoder  # noqa: E402


def load_preprocessed(prep_dir):
    test_x   = np.load(os.path.join(prep_dir, 'test_features.npy'))
    test_rid = np.load(os.path.join(prep_dir, 'test_recording_ids.npy'))
    test_lbl = np.load(os.path.join(prep_dir, 'test_recording_labels.npy'))
    test_mid = np.load(os.path.join(prep_dir, 'test_recording_machine_ids.npy'))
    return test_x, test_rid, test_lbl, test_mid


def evaluate_auc(model, test_x_std, test_rid, test_lbl, test_mid, batch_size=4096):
    model.eval()
    n = len(test_x_std)
    per_window_mse = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(test_x_std[i:i+batch_size].astype(np.float32))
            recon = model(x)
            per_window_mse[i:i+batch_size] = ((recon - x) ** 2).mean(dim=1).cpu().numpy()
    n_rec = len(test_lbl)
    per_rec_mse = np.zeros(n_rec, dtype=np.float64)
    counts = np.zeros(n_rec, dtype=np.int64)
    np.add.at(per_rec_mse, test_rid, per_window_mse)
    np.add.at(counts, test_rid, 1)
    per_rec_mse = per_rec_mse / np.maximum(counts, 1)
    auc_overall = float(roc_auc_score(test_lbl, per_rec_mse))
    aucs_per_machine = {}
    for mid in sorted(np.unique(test_mid).tolist()):
        mask = test_mid == mid
        if (test_lbl[mask] == 0).sum() == 0 or (test_lbl[mask] == 1).sum() == 0:
            continue
        aucs_per_machine[int(mid)] = float(roc_auc_score(test_lbl[mask], per_rec_mse[mask]))
    return auc_overall, aucs_per_machine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', required=True,
                    help='Path to trained float .pth (companion *_input_mean.npy / *_input_std.npy must exist)')
    ap.add_argument('--prep-dir', required=True,
                    help='Path to data/toycar_preprocessed (contains test_features.npy etc.)')
    ap.add_argument('--calib-size', type=int, default=1000)
    ap.add_argument('--target', default='DPUCZDX8G_ISA1_B512')
    ap.add_argument('--output-dir', default='quantize_result_autoencoder')
    args = ap.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    pth_dir = os.path.dirname(os.path.abspath(args.pth)) or '.'
    pth_base = os.path.basename(args.pth)
    if pth_base.endswith('.pth'):
        pth_base = pth_base[:-4]
    mean_path = os.path.join(pth_dir, f'{pth_base}_input_mean.npy')
    std_path  = os.path.join(pth_dir, f'{pth_base}_input_std.npy')
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            f"standardization files missing alongside .pth:\n  {mean_path}\n  {std_path}\n"
            "Train with finn/train_autoencoder.py to produce them.")
    mean = np.load(mean_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)
    print(f"Loaded standardization: mean.shape={mean.shape} std.shape={std.shape}")

    print(f"Loading test data from {args.prep_dir}...")
    test_x, test_rid, test_lbl, test_mid = load_preprocessed(args.prep_dir)
    test_x_std = ((test_x - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    print(f"  test_x: {test_x.shape}  recordings: {len(test_lbl)}  "
          f"({int((test_lbl==0).sum())} normal, {int((test_lbl==1).sum())} anomaly)")

    print(f"Loading model from {args.pth}...")
    model = FCAutoencoder()
    sd = torch.load(args.pth, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    auc_float, aucs_float_pm = evaluate_auc(model, test_x_std, test_rid, test_lbl, test_mid)
    print(f"Float AUC overall: {auc_float:.4f}")
    for mid, a in sorted(aucs_float_pm.items()):
        print(f"  machine {mid}: {a:.4f}")

    # Calibration pass.
    dummy = torch.randn(1, 640)
    print(f"\nCalibrating ({args.calib_size} samples)...")
    quantizer = torch_quantizer('calib', model, (dummy,),
                                 device=torch.device('cpu'), target=args.target)
    qmodel = quantizer.quant_model
    qmodel.eval()
    calib = torch.from_numpy(test_x_std[:args.calib_size])
    with torch.no_grad():
        for i in range(0, args.calib_size, 32):
            qmodel(calib[i:i+32])
    quantizer.export_quant_config()

    # Test pass to evaluate quantized AUC.
    print(f"Evaluating quantized model...")
    quantizer = torch_quantizer('test', model, (dummy,),
                                 device=torch.device('cpu'), target=args.target)
    qmodel = quantizer.quant_model
    qmodel.eval()
    auc_quant, aucs_quant_pm = evaluate_auc(qmodel, test_x_std, test_rid, test_lbl, test_mid)
    print(f"Quantized AUC overall: {auc_quant:.4f}  (Δ vs float = {auc_quant - auc_float:+.4f})")
    for mid, a in sorted(aucs_quant_pm.items()):
        print(f"  machine {mid}: {a:.4f}")

    quantizer.export_xmodel(output_dir=args.output_dir, deploy_check=False)

    summary = {
        'pth':              args.pth,
        'target':           args.target,
        'calib_size':       args.calib_size,
        'n_params':         n_params,
        'auc_float_overall': auc_float,
        'auc_quant_overall': auc_quant,
        'auc_float_per_machine': aucs_float_pm,
        'auc_quant_per_machine': aucs_quant_pm,
    }
    with open(os.path.join(args.output_dir, 'ptq_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {args.output_dir}/ptq_summary.json")
    print(f"\nNext: vai_c_xir -x {args.output_dir}/FCAutoencoder_int.xmodel "
          f"-a arch_zu3_b512.json "
          f"-o compiled_autoencoder "
          f"-n autoencoder_toycar_canonical")


if __name__ == '__main__':
    main()
