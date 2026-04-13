"""
Train a Brevitas model and export to ONNX for FINN compilation.
Run inside the FINN Docker container.

Usage:
  python train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
  python train_and_export.py --model cnn --dataset cifar10 --size tiny --epochs 20
"""
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from brevitas.export import export_qonnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from mlp import MLP_Brevitas, MLP_Brevitas_INT4, get_mlp_config
from cnn import CNN_Brevitas, get_cnn_config

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['mlp', 'cnn'])
parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10'])
parser.add_argument('--size', default='tiny', help='Model size config (tiny, small, medium, etc.)')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--output', default=None, help='Output ONNX filename')
parser.add_argument('--int4', action='store_true', help='Use INT4 weights/acts instead of INT8')
parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
args = parser.parse_args()

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
    model = CNN_Brevitas(in_channels=in_channels, channels=channels)
    dummy = torch.randn(1, in_channels, img_size, img_size)

print(f"Model: {args.model} ({args.size}), Dataset: {args.dataset}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

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
    print(f"  Epoch {epoch+1}/{args.epochs}: {100*correct/total:.2f}%")

# Save weights
weight_file = args.output or f"{args.model}_{args.dataset}_{args.size}"
if args.int4:
    weight_file += "_int4"

for ext in ('pth', 'onnx'):
    path = f"{weight_file}.{ext}"
    if os.path.exists(path) and not args.force:
        raise FileExistsError(f"{path} exists. Use --force to overwrite.")

torch.save(model.state_dict(), f"{weight_file}.pth")

# Export ONNX
model.eval()
export_qonnx(model, dummy, f"{weight_file}.onnx")
print(f"Exported: {weight_file}.onnx")
