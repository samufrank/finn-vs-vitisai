"""
Train a Brevitas model and export to ONNX for FINN compilation.
Run inside the FINN Docker container.

Usage:
  python train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
  python train_and_export.py --model cnn --dataset cifar10 --size tiny --epochs 20
  python train_and_export.py --model cnn --dataset mnist --size tiny --int4
    (INT4 CNN defaults: lr=3e-4, epochs=30; set --lr / --epochs to override.)

Saves the best-val-accuracy checkpoint (not the last-epoch state) as the
.pth/.onnx artifact. Reports both best-val and final-epoch accuracy on test.
"""
import argparse
import copy
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from brevitas.export import export_qonnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from mlp import MLP_Brevitas, MLP_Brevitas_INT4, get_mlp_config
from cnn import (
    CNN_Brevitas, CNN_Brevitas_INT4, CNN_Brevitas_INT4_PerChan,
    CNN_Brevitas_INT4_NoBN, CNN_Brevitas_INT4_NoBN_Wide, get_cnn_config,
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['mlp', 'cnn'])
parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10'])
parser.add_argument('--size', default='tiny', help='Model size config (tiny, small, medium, etc.)')
parser.add_argument('--epochs', type=int, default=None,
                    help='Training epochs. Default: 10, except --int4 --model cnn which defaults to 30.')
parser.add_argument('--lr', type=float, default=None,
                    help='Learning rate. Default: 1e-3, except --int4 --model cnn which defaults to 3e-4.')
parser.add_argument('--output', default=None, help='Output ONNX filename')
parser.add_argument('--int4', action='store_true', help='Use INT4 weights/acts instead of INT8')
parser.add_argument('--per-channel', action='store_true',
                    help='With --int4 --model cnn: use per-channel weight quantization '
                         '(CNN_Brevitas_INT4_PerChan). Required for INT4 CNN with BN fold.')
parser.add_argument('--no-bn', action='store_true',
                    help='With --int4 --model cnn: remove BatchNorm (CNN_Brevitas_INT4_NoBN). '
                         'Required for VTA INT4 deploy via MLP INT4 pattern.')
parser.add_argument('--wide', action='store_true',
                    help='With --int4 --no-bn --model cnn: use wider channels [16,32] '
                         '(CNN_Brevitas_INT4_NoBN_Wide).')
parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
args = parser.parse_args()

# Conditional defaults for INT4 CNN (user-provided --lr / --epochs always win).
if args.epochs is None:
    args.epochs = 30 if (args.int4 and args.model == 'cnn') else 10
if args.lr is None:
    args.lr = 3e-4 if (args.int4 and args.model == 'cnn') else 1e-3

# Dataset
transform = transforms.Compose([transforms.ToTensor()])
if args.dataset == 'mnist':
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    input_size, in_channels, img_size = 784, 1, 28
elif args.dataset == 'cifar10':
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    input_size, in_channels, img_size = 3072, 3, 32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# Model
if args.model == 'mlp':
    hidden = get_mlp_config(args.size)
    ModelClass = MLP_Brevitas_INT4 if args.int4 else MLP_Brevitas
    model = ModelClass(input_size=input_size, hidden_sizes=hidden)
    dummy = torch.randn(1, in_channels, img_size, img_size)
elif args.model == 'cnn':
    channels = get_cnn_config(args.size)
    if args.int4 and args.no_bn and args.wide:
        ModelClass = CNN_Brevitas_INT4_NoBN_Wide
        channels = [16, 32]  # override 'tiny' default for wide
    elif args.int4 and args.no_bn:
        ModelClass = CNN_Brevitas_INT4_NoBN
    elif args.int4 and args.per_channel:
        ModelClass = CNN_Brevitas_INT4_PerChan
    elif args.int4:
        ModelClass = CNN_Brevitas_INT4
    else:
        ModelClass = CNN_Brevitas
    model = ModelClass(in_channels=in_channels, channels=channels)
    dummy = torch.randn(1, in_channels, img_size, img_size)

print(f"Model: {args.model} ({args.size}), Dataset: {args.dataset}, "
      f"INT4: {args.int4}, lr={args.lr}, epochs={args.epochs}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# Train
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

# Best-val tracking (val = test set in this recipe — matches prior MLP INT4
# training convention; no separate val split is carved from train).
best_val_acc = -1.0
best_val_epoch = -1
best_state = None
final_val_acc = None

for epoch in range(args.epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    final_val_acc = val_acc
    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())
        marker = "  <-- best"
    print(f"  Epoch {epoch+1}/{args.epochs}: {100*val_acc:.2f}%{marker}")

print()
print(f"Best val accuracy: {100*best_val_acc:.2f}% @ epoch {best_val_epoch}")
print(f"Final epoch accuracy: {100*final_val_acc:.2f}% @ epoch {args.epochs}")

# Save weights
weight_file = args.output or f"{args.model}_{args.dataset}_{args.size}"
if args.int4:
    weight_file += "_int4"
if args.per_channel:
    weight_file += "_perchan"
if args.no_bn:
    weight_file += "_nobn"
if args.wide:
    weight_file += "_wide"

for ext in ('pth', 'onnx'):
    path = f"{weight_file}.{ext}"
    if os.path.exists(path) and not args.force:
        raise FileExistsError(f"{path} exists. Use --force to overwrite.")

# Reload best-val state before saving and exporting ONNX so both artifacts
# reflect the best-val checkpoint, not the final-epoch state.
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()

torch.save(model.state_dict(), f"{weight_file}.pth")
print(f"Saved: {weight_file}.pth  (best-val checkpoint, "
      f"epoch {best_val_epoch}, {100*best_val_acc:.2f}%)")

export_qonnx(model, dummy, f"{weight_file}.onnx")
print(f"Exported: {weight_file}.onnx")
