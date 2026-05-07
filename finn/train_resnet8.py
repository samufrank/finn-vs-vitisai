"""
Train ResNet-8 with Brevitas INT8 QAT on CIFAR-10 and export to QONNX for FINN.

Recipe: SGD + momentum + cosine LR, standard CIFAR-10 augmentation, 200 epochs.
Saves best-val checkpoint. GPU if available, falls back to CPU.

Usage:
  python finn/train_resnet8.py --epochs 200
"""
import argparse
import copy
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from brevitas.export import export_qonnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from resnet import ResNet8_Brevitas

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--output', default='resnet8_cifar10_int8',
                    help='Output basename (creates .pth + .onnx)')
parser.add_argument('--data-dir', default='./data')
parser.add_argument('--force', action='store_true')
parser.add_argument('--num-workers', type=int, default=4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_tf = transforms.Compose([transforms.ToTensor()])

train_data = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_tf)
test_data = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_tf)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

model = ResNet8_Brevitas(in_channels=3, num_classes=10).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResNet8_Brevitas, parameters: {n_params:,}")

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                     momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
loss_fn = nn.CrossEntropyLoss()

best_val_acc = -1.0
best_val_epoch = -1
best_state = None
final_val_acc = None

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    n_train = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        n_train += images.size(0)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
    cur_lr = scheduler.get_last_lr()[0]
    print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={train_loss/n_train:.4f} "
          f"val={100*val_acc:.2f}% lr={cur_lr:.2e}{marker}", flush=True)
    scheduler.step()

print()
print(f"Best val accuracy: {100*best_val_acc:.2f}% @ epoch {best_val_epoch}")
print(f"Final epoch accuracy: {100*final_val_acc:.2f}% @ epoch {args.epochs}")

base = os.path.join(os.path.dirname(__file__), args.output)
for ext in ('pth', 'onnx'):
    path = f"{base}.{ext}"
    if os.path.exists(path) and not args.force:
        raise FileExistsError(f"{path} exists. Use --force to overwrite.")

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
model.cpu()

torch.save(model.state_dict(), f"{base}.pth")
print(f"Saved: {base}.pth (best-val checkpoint, epoch {best_val_epoch}, "
      f"{100*best_val_acc:.2f}%)")

dummy = torch.randn(1, 3, 32, 32)
export_qonnx(model, dummy, f"{base}.onnx")
print(f"Exported: {base}.onnx")
