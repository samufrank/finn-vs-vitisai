"""
Train a PyTorch model, quantize with Vitis AI, and export xmodel.
Run inside the Vitis AI Docker container.

Usage:
  python train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
  python train_and_quantize.py --model cnn --dataset cifar10 --size tiny --epochs 20
"""
import argparse
import sys
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pytorch_nndct.apis import torch_quantizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from mlp import MLP, get_mlp_config
from cnn import CNN, get_cnn_config
from resnet import ResNet8

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['mlp', 'cnn', 'resnet8'])
parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10'])
parser.add_argument('--size', default='tiny', help='Model size config (ignored for resnet8)')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for compiled xmodel')
parser.add_argument('--target', default='DPUCZDX8G_ISA1_B512')
parser.add_argument('--calib_size', type=int, default=1000, help='Number of calibration images')
parser.add_argument('--from-pretrained', default=None,
                    help='If set, load this .pth and skip training (use it as the '
                         'float input to PTQ). Useful for cross-toolchain comparison '
                         'where the float weights were trained separately.')
args = parser.parse_args()

# Clean previous quantization
if os.path.exists('quantize_result'):
    shutil.rmtree('quantize_result')

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
    model = MLP(input_size=input_size, hidden_sizes=hidden)
elif args.model == 'cnn':
    channels = get_cnn_config(args.size)
    model = CNN(in_channels=in_channels, channels=channels)
elif args.model == 'resnet8':
    model = ResNet8(in_channels=in_channels, num_classes=10)

name = f"{args.model}_{args.dataset}" if args.model == 'resnet8' \
       else f"{args.model}_{args.dataset}_{args.size}"
print(f"Model: {name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

if args.from_pretrained:
    # Load externally-trained weights; skip training.
    print(f"Loading pre-trained weights from {args.from_pretrained}")
    sd = torch.load(args.from_pretrained, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()
else:
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

    torch.save(model.state_dict(), f"{name}.pth")
    model.eval()

# Float accuracy
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Float accuracy: {100*correct/total:.2f}%")

# Quantize - calibration
dummy = torch.randn(args.batch_size, in_channels, img_size, img_size)
print(f"Calibrating (batch={args.batch_size})...")
quantizer = torch_quantizer('calib', model, (dummy,), device=torch.device('cpu'), target=args.target)
quant_model = quantizer.quant_model
quant_model.eval()

calib_loader = torch.utils.data.DataLoader(test_data, batch_size=max(args.batch_size, 32), shuffle=False)
count = 0
for images, labels in calib_loader:
    if images.size(0) < args.batch_size:
        continue
    quant_model(images[:args.batch_size])
    count += args.batch_size
    if count >= args.calib_size:
        break
quantizer.export_quant_config()

# Quantize - export and evaluate
print("Exporting...")
quantizer = torch_quantizer('test', model, (dummy,), device=torch.device('cpu'), target=args.target)
quant_model = quantizer.quant_model
quant_model.eval()

correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        if images.size(0) < args.batch_size:
            continue
        preds = quant_model(images[:args.batch_size]).argmax(1)
        correct += (preds == labels[:args.batch_size]).sum().item()
        total += args.batch_size
print(f"Quantized accuracy: {100*correct/total:.2f}%")

quantizer.export_xmodel()
print(f"Exported: quantize_result/MLP_int.xmodel or CNN_int.xmodel")
print(f"Next: vai_c_xir -x quantize_result/*_int.xmodel -a arch_zu3_b512.json -o compiled -n {name}")
