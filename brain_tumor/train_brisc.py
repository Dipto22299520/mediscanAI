"""
Train attention-augmented classifier on BRISC2025 brain tumor dataset.

Usage:
    python train_brisc.py
"""

import sys
import os

# Add the brain_tumor directory to path to import the training script
sys.path.append(os.path.join(os.path.dirname(__file__), 'brain_tumor'))

from train import main
import argparse

if __name__ == "__main__":
    # Configure arguments for BRISC2025 dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        default=r"D:\mediscanai\brisc2025\classification_task\train",
                        help="root dataset folder (class subfolders)")
    parser.add_argument("--output_dir", type=str, 
                        default=r"D:\mediscanai\outputs\brisc_model",
                        help="where to save models and logs")
    parser.add_argument("--img_size", type=int, default=384)  # Increased for 5090
    parser.add_argument("--batch_size", type=int, default=64)  # Increased for 5090
    parser.add_argument("--epochs", type=int, default=50)  # More epochs for better training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=8)  # More workers for faster data loading
    parser.add_argument("--pretrained", action='store_true', default=True,
                        help="use imagenet pretrained backbone")
    parser.add_argument("--resume", type=str, default=None, 
                        help="path to checkpoint to resume")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Attention-Based Brain Tumor Classifier")
    print("=" * 60)
    print(f"Dataset: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Classes: glioma, meningioma, no_tumor, pituitary")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Validation Split: {args.val_split}")
    print("=" * 60)
    
    main(args)