import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# Attention U-Net Implementation
class AttentionBlock(nn.Module):
    """Attention Block for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Convolutional Block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    """Up Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)


class AttentionUNet(nn.Module):
    """Attention U-Net Architecture for Brain Tumor Segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge
        self.conv5 = ConvBlock(512, 1024)
        
        # Decoder with Attention
        self.up4 = UpConvBlock(1024, 512)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.conv_up4 = ConvBlock(1024, 512)
        
        self.up3 = UpConvBlock(512, 256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv_up3 = ConvBlock(512, 256)
        
        self.up2 = UpConvBlock(256, 128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv_up2 = ConvBlock(256, 128)
        
        self.up1 = UpConvBlock(128, 64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv_up1 = ConvBlock(128, 64)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        p1 = self.pool1(e1)
        
        e2 = self.conv2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.conv3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.conv4(p3)
        p4 = self.pool4(e4)
        
        # Bridge
        e5 = self.conv5(p4)
        
        # Decoder with Attention
        d4 = self.up4(e5)
        e4 = self.att4(g=d4, x=e4)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.conv_up4(d4)
        
        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.conv_up3(d3)
        
        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.conv_up2(d2)
        
        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.conv_up1(d1)
        
        # Final output
        out = self.final(d1)
        out = self.sigmoid(out)
        
        return out


class SegmentationDataset(Dataset):
    """Dataset class for Brain Tumor Segmentation"""
    def __init__(self, images_dir, masks_dir, transform=None, img_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size
        
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Try different mask extensions
        mask_name = img_name
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # Try with different extension
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                mask_path = os.path.join(self.masks_dir, base_name + ext)
                if os.path.exists(mask_path):
                    break
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        
        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        
        # Threshold mask to binary
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask


class DiceLoss(nn.Module):
    """Dice Loss for Segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate Dice Score and IoU"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    
    iou = (intersection + 1e-7) / (pred_binary.sum() + target_binary.sum() - intersection + 1e-7)
    
    return dice.item(), iou.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice, iou = calculate_metrics(outputs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice, iou = calculate_metrics(outputs, masks)
            
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou


def main():
    # Hyperparameters - Optimized for RTX 5090
    IMG_SIZE = (512, 512)  # Increased resolution for 5090
    BATCH_SIZE = 16  # Increased batch size for 5090
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    TRAIN_IMAGES = BASE_DIR / 'brisc2025' / 'segmentation_task' / 'train' / 'images'
    TRAIN_MASKS = BASE_DIR / 'brisc2025' / 'segmentation_task' / 'train' / 'masks'
    TEST_IMAGES = BASE_DIR / 'brisc2025' / 'segmentation_task' / 'test' / 'images'
    TEST_MASKS = BASE_DIR / 'brisc2025' / 'segmentation_task' / 'test' / 'masks'
    OUTPUT_DIR = BASE_DIR / 'outputs' / 'segmentation_model'
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SegmentationDataset(
        images_dir=str(TRAIN_IMAGES),
        masks_dir=str(TRAIN_MASKS),
        img_size=IMG_SIZE
    )
    
    test_dataset = SegmentationDataset(
        images_dir=str(TEST_IMAGES),
        masks_dir=str(TEST_MASKS),
        img_size=IMG_SIZE
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Increased for RTX 5090
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,  # Increased for RTX 5090
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nInitializing Attention U-Net model...")
    model = AttentionUNet(in_channels=3, out_channels=1)
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    
    best_dice = 0.0
    
    print("\nStarting training...\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, test_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        print()
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_loss': val_loss
            }, OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved best model (Dice: {val_dice:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_loss': val_loss
            }, OUTPUT_DIR / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'val_iou': val_iou,
        'val_loss': val_loss
    }, OUTPUT_DIR / 'final_model.pth')
    
    # Save training history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best Validation Dice Score: {best_dice:.4f}")
    print(f"Models saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
