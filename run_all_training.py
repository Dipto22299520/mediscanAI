"""
Master script to run both classification and segmentation training
Optimized for RTX 5090 GPU

Usage:
    python run_all_training.py [options]
    
Options:
    --classification-only : Run only classification training
    --segmentation-only   : Run only segmentation training
    --skip-classification : Skip classification, run segmentation only
    --skip-segmentation  : Skip segmentation, run classification only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def run_classification_training():
    """Run classification training"""
    print_banner("STARTING CLASSIFICATION TRAINING (Attention-Based Classifier)")
    
    script_path = Path(__file__).parent / "brain_tumor" / "train_brisc.py"
    
    print(f"📍 Script: {script_path}")
    print(f"🎯 Task: Brain Tumor Classification (4 classes)")
    print(f"🖼️  Image Size: 384x384")
    print(f"📦 Batch Size: 64")
    print(f"🔄 Epochs: 50")
    print(f"🧠 Model: ResNet50 + Attention Mechanism\n")
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(Path(__file__).parent)
        )
        
        elapsed = time.time() - start_time
        print_banner(f"✅ CLASSIFICATION TRAINING COMPLETED in {elapsed/60:.2f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        print_banner(f"❌ CLASSIFICATION TRAINING FAILED (Exit code: {e.returncode})")
        return False
    except Exception as e:
        print_banner(f"❌ CLASSIFICATION TRAINING ERROR: {str(e)}")
        return False

def run_segmentation_training():
    """Run segmentation training"""
    print_banner("STARTING SEGMENTATION TRAINING (Attention U-Net)")
    
    script_path = Path(__file__).parent / "brain_tumor" / "train_segmentation.py"
    
    print(f"📍 Script: {script_path}")
    print(f"🎯 Task: Brain Tumor Segmentation (Pixel-wise)")
    print(f"🖼️  Image Size: 512x512")
    print(f"📦 Batch Size: 16")
    print(f"🔄 Epochs: 50")
    print(f"🧠 Model: Attention U-Net\n")
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(Path(__file__).parent)
        )
        
        elapsed = time.time() - start_time
        print_banner(f"✅ SEGMENTATION TRAINING COMPLETED in {elapsed/60:.2f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        print_banner(f"❌ SEGMENTATION TRAINING FAILED (Exit code: {e.returncode})")
        return False
    except Exception as e:
        print_banner(f"❌ SEGMENTATION TRAINING ERROR: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run brain tumor classification and segmentation training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--classification-only', action='store_true',
                        help='Run only classification training')
    parser.add_argument('--segmentation-only', action='store_true',
                        help='Run only segmentation training')
    parser.add_argument('--skip-classification', action='store_true',
                        help='Skip classification training')
    parser.add_argument('--skip-segmentation', action='store_true',
                        help='Skip segmentation training')
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.classification_only and args.segmentation_only:
        print("❌ Error: Cannot specify both --classification-only and --segmentation-only")
        sys.exit(1)
    
    # Determine what to run
    run_classification = True
    run_segmentation = True
    
    if args.classification_only:
        run_segmentation = False
    elif args.segmentation_only:
        run_classification = False
    elif args.skip_classification:
        run_classification = False
    elif args.skip_segmentation:
        run_segmentation = False
    
    # Print overall configuration
    print_banner("BRAIN TUMOR ML TRAINING PIPELINE - RTX 5090 OPTIMIZED")
    
    print("🚀 Training Configuration:")
    print(f"   • Classification: {'✓ Enabled' if run_classification else '✗ Disabled'}")
    print(f"   • Segmentation:   {'✓ Enabled' if run_segmentation else '✗ Disabled'}")
    print(f"   • GPU: NVIDIA RTX 5090")
    print()
    
    # Check if data directories exist
    base_dir = Path(__file__).parent
    class_dir = base_dir / "brisc2025" / "classification_task" / "train"
    seg_dir = base_dir / "brisc2025" / "segmentation_task" / "train"
    
    if run_classification and not class_dir.exists():
        print(f"⚠️  Warning: Classification data not found at {class_dir}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    if run_segmentation and not seg_dir.exists():
        print(f"⚠️  Warning: Segmentation data not found at {seg_dir}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Track results
    results = {}
    overall_start = time.time()
    
    # Run classification training
    if run_classification:
        results['classification'] = run_classification_training()
        if not results['classification']:
            print("\n⚠️  Classification training failed. Continue with segmentation? (y/n): ", end='')
            if input().lower() != 'y':
                sys.exit(1)
    
    # Run segmentation training
    if run_segmentation:
        results['segmentation'] = run_segmentation_training()
    
    # Print final summary
    overall_elapsed = time.time() - overall_start
    
    print_banner("TRAINING PIPELINE SUMMARY")
    
    if run_classification:
        status = "✅ SUCCESS" if results.get('classification') else "❌ FAILED"
        print(f"Classification Training: {status}")
    
    if run_segmentation:
        status = "✅ SUCCESS" if results.get('segmentation') else "❌ FAILED"
        print(f"Segmentation Training:   {status}")
    
    print(f"\n⏱️  Total Time: {overall_elapsed/60:.2f} minutes")
    print(f"📂 Output Location: {base_dir / 'outputs'}")
    
    # Print output directories
    print("\n📁 Model Outputs:")
    if run_classification:
        class_output = base_dir / "outputs" / "brisc_model"
        print(f"   • Classification: {class_output}")
    if run_segmentation:
        seg_output = base_dir / "outputs" / "segmentation_model"
        print(f"   • Segmentation:   {seg_output}")
    
    print("\n" + "=" * 80)
    
    # Exit with appropriate code
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)

if __name__ == "__main__":
    main()
