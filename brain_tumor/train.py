"""
train_attention.py

Train an attention-augmented classifier on a local image dataset organized by class folders.

Usage:
    python train_attention.py --data_dir ./dataset --epochs 30 --batch_size 32
"""

import argparse
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------
# Attention Block
# -------------------------
class AttentionBlock(nn.Module):
    """
    Simple combined channel + spatial attention block.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        y = self.avg_pool(x)
        y = self.fc(y)  # (B, C, 1, 1)
        x = x * y

        # Spatial attention
        # compute mean and max across channel
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        cat = torch.cat([max_pool, mean_pool], dim=1)
        s = self.spatial_conv(cat)
        s = self.spatial_sigmoid(s)
        out = x * s
        return out

# -------------------------
# Model with attention
# -------------------------
class AttentionResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # load pretrained resnet18
        backbone = models.resnet18(pretrained=pretrained)
        # remove final fc
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        # Insert attention after layer2 and layer4 for example
        # We'll create small wrappers in forward
        self.att2 = AttentionBlock(in_channels=128)  # after layer2 (resnet18: layer2 outputs 128)
        self.att4 = AttentionBlock(in_channels=512)  # after layer4

        self.avgpool = backbone.avgpool
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        # run through initial blocks until layer2
        x = self.features[0](x)  # conv1
        x = self.features[1](x)  # bn1
        x = self.features[2](x)  # relu
        x = self.features[3](x)  # maxpool

        # layers 1,2,3,4 are in sequence elements 4..7
        x = self.features[4](x)  # layer1
        x = self.features[5](x)  # layer2
        x = self.att2(x)         # attention after layer2

        x = self.features[6](x)  # layer3
        x = self.features[7](x)  # layer4
        x = self.att4(x)         # attention after layer4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------
# Dataset
# -------------------------
class ImageFolderDataset(Dataset):
    """
    Simple dataset reading images from a list of (path, class_idx)
    """
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, class_idx)
        transform: torchvision transform applied to PIL image
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------
# Utility functions
# -------------------------
def list_images_and_labels(data_dir):
    """
    Walk data_dir, assume immediate subfolders are class names.
    Return list of (path, label_idx) and class_names
    """
    data_dir = Path(data_dir)
    classes = [p.name for p in sorted(data_dir.iterdir()) if p.is_dir()]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    samples = []
    for c in classes:
        cdir = data_dir / c
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'):
            for p in cdir.glob(ext):
                samples.append((str(p), class_to_idx[c]))
    return samples, classes

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []
    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds += outputs.detach().argmax(dim=1).cpu().tolist()
        targets += labels.cpu().tolist()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(targets, preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.cpu().tolist()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(targets, preds)
    return epoch_loss, epoch_acc, preds, targets

# -------------------------
# Main
# -------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # prepare samples and classes
    samples, classes = list_images_and_labels(args.data_dir)
    if len(samples) == 0:
        raise RuntimeError("No images found in the data_dir. Check folder structure.")
    print(f"Found {len(samples)} images across {len(classes)} classes: {classes}")

    # split
    train_samples, val_samples = train_test_split(samples, test_size=args.val_split, stratify=[s[1] for s in samples], random_state=42)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = ImageFolderDataset(train_samples, transform=train_tf)
    val_ds = ImageFolderDataset(val_samples, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    model = AttentionResNet18(num_classes=len(classes), pretrained=args.pretrained)
    model = model.to(device)

    # loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # optionally resume
    start_epoch = 0
    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)

    # save classes map
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump(classes, f)

    # training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # save checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"last_epoch.pth"))

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": classes
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved new best model with val_acc: {best_val_acc:.4f}")

    # final evaluation metrics & report
    print("\nTraining finished. Evaluating best model on validation set...")
    best = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best['model_state'])
    _, _, preds, targets = evaluate(model, val_loader, criterion, device)
    print("Classification report:")
    print(classification_report(targets, preds, target_names=classes, digits=4))

    # save history plot
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(args.output_dir, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.savefig(os.path.join(args.output_dir, "acc.png"))
    plt.close()

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved artifacts to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="root dataset folder (class subfolders)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="where to save models and logs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action='store_true', help="use imagenet pretrained backbone")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume")
    args = parser.parse_args()

    main(args)
