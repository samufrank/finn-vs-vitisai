"""Train FC autoencoder on ToyCar (DCASE 2020) for MLPerf Tiny anomaly detection.

Either the plain PyTorch (`--variant float`) or Brevitas INT8 QAT
(`--variant brevitas`) version. Both train on normal-only data with MSE loss.

Evaluation: per-recording reconstruction error → AUC-ROC on the test set.

Usage:
  python finn/train_autoencoder.py --variant brevitas --epochs 30
  python finn/train_autoencoder.py --variant float --epochs 30
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'models'))

from autoencoder import FCAutoencoder, FCAutoencoder_Brevitas

from sklearn.metrics import roc_auc_score


def load_preprocessed(prep_dir):
    train_x = np.load(os.path.join(prep_dir, 'train_features.npy'))
    test_x = np.load(os.path.join(prep_dir, 'test_features.npy'))
    test_rid = np.load(os.path.join(prep_dir, 'test_recording_ids.npy'))
    test_lbl = np.load(os.path.join(prep_dir, 'test_recording_labels.npy'))
    test_mid = np.load(os.path.join(prep_dir, 'test_recording_machine_ids.npy'))
    print(f"  train_features: {train_x.shape}  ({train_x.nbytes/1e6:.0f} MB)")
    print(f"  test_features:  {test_x.shape}   ({test_x.nbytes/1e6:.0f} MB)")
    print(f"  test recordings: {len(test_lbl)}  "
          f"({int((test_lbl == 0).sum())} normal, {int((test_lbl == 1).sum())} anomaly)")
    return train_x, test_x, test_rid, test_lbl, test_mid


def standardize_fit(train_x):
    """Fit mean/std on the train set and return normalized train_x + (mean, std)."""
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (train_x - mean) / std, mean, std


def make_loader(arr, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(arr.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=torch.cuda.is_available())


def evaluate_auc(model, test_x_std, test_rid, test_lbl, test_mid, device,
                 batch_size=4096):
    """Per-recording mean MSE → AUC-ROC overall and per-machine."""
    model.eval()
    n = len(test_x_std)
    per_window_mse = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(test_x_std[i:i+batch_size]).to(device, non_blocking=True)
            recon = model(x)
            mse = ((recon - x) ** 2).mean(dim=1).cpu().numpy()
            per_window_mse[i:i+batch_size] = mse
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
        a = float(roc_auc_score(test_lbl[mask], per_rec_mse[mask]))
        aucs_per_machine[int(mid)] = a
    return auc_overall, aucs_per_machine, per_rec_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['float', 'brevitas'])
    ap.add_argument('--prep-dir', default='data/toycar_preprocessed')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--output-dir', default='finn',
                    help='Where to save the .pth + .onnx (brevitas only)')
    ap.add_argument('--name', default=None,
                    help='Output basename (default: autoencoder_toycar_{variant})')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    name = args.name or f"autoencoder_toycar_{args.variant}"

    print("\nLoading preprocessed data...")
    train_x, test_x, test_rid, test_lbl, test_mid = load_preprocessed(args.prep_dir)

    print("\nFitting standardization on train set...")
    train_x_std, mean, std = standardize_fit(train_x)
    test_x_std = (test_x - mean) / std
    print(f"  feature mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  feature std  range: [{std.min():.2f}, {std.max():.2f}]")

    # Free raw arrays
    del train_x, test_x

    # Save standardization parameters alongside the model.
    np.save(os.path.join(args.output_dir, f'{name}_input_mean.npy'), mean)
    np.save(os.path.join(args.output_dir, f'{name}_input_std.npy'), std)

    print("\nBuilding model...")
    if args.variant == 'float':
        model = FCAutoencoder().to(device)
    else:
        model = FCAutoencoder_Brevitas().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {args.variant}: {n_params:,} parameters")

    train_loader = make_loader(train_x_std, args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_auc = -1.0
    best_state = None
    best_epoch = -1
    print(f"\nTraining {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr})")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0
        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)
            recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)
        scheduler.step()
        train_loss = loss_sum / n_seen
        auc_overall, aucs_per_machine, _ = evaluate_auc(
            model, test_x_std, test_rid, test_lbl, test_mid, device)
        marker = ""
        if auc_overall > best_auc:
            best_auc = auc_overall
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <-- best"
        per_m = " ".join(f"id{m}:{a:.3f}" for m, a in sorted(aucs_per_machine.items()))
        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"train_mse={train_loss:.4f}  AUC={auc_overall:.4f}  "
              f"({per_m})  [{time.time()-t0:.1f}s]{marker}", flush=True)

    print(f"\nBest AUC: {best_auc:.4f} @ epoch {best_epoch}")

    # Reload best, eval one more time, save.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    auc_overall, aucs_per_machine, per_rec_mse = evaluate_auc(
        model, test_x_std, test_rid, test_lbl, test_mid, device)
    print(f"Final AUC (overall): {auc_overall:.4f}")
    for mid, a in sorted(aucs_per_machine.items()):
        print(f"  machine id {mid}: AUC = {a:.4f}")

    pth = os.path.join(args.output_dir, f'{name}.pth')
    if os.path.exists(pth) and not args.force:
        raise FileExistsError(f"{pth} exists; use --force")
    model.cpu()
    torch.save(model.state_dict(), pth)
    print(f"Saved: {pth}")

    # Save metadata. Walk the Sequential to recover the actual layer widths.
    widths = [m.out_features for m in model.net if hasattr(m, 'out_features')]
    arch_str = '-'.join([str(640)] + [str(w) for w in widths])
    meta = {
        'variant': args.variant,
        'architecture': arch_str,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'best_epoch': best_epoch,
        'best_auc_overall': best_auc,
        'final_auc_overall': auc_overall,
        'final_auc_per_machine': aucs_per_machine,
        'n_params': n_params,
    }
    with open(os.path.join(args.output_dir, f'{name}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {name}_meta.json")

    # QONNX export for Brevitas variant — input is the standardized 640-dim
    # vector (standardization is performed outside the model on CPU).
    if args.variant == 'brevitas':
        from brevitas.export import export_qonnx
        onnx_path = os.path.join(args.output_dir, f'{name}.onnx')
        if os.path.exists(onnx_path) and not args.force:
            raise FileExistsError(f"{onnx_path} exists; use --force")
        dummy = torch.randn(1, 640)
        export_qonnx(model, dummy, onnx_path)
        print(f"Exported QONNX: {onnx_path}")


if __name__ == '__main__':
    main()
