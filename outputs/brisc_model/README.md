# Model Files

This folder contains trained model checkpoints.

## Files

- `classes.json` - Class names mapping (included in repo)
- `best_model.pth` - Best model checkpoint (128 MB, not included due to GitHub size limits)
- `last_epoch.pth` - Last epoch checkpoint (128 MB, not included due to GitHub size limits)

## Getting the Trained Model

### Option 1: Train Your Own Model

```bash
python brain_tumor/train.py \
  --data_dir ./path/to/dataset/train \
  --output_dir ./outputs/brisc_model \
  --epochs 50 \
  --batch_size 32 \
  --pretrained
```

### Option 2: Download Pre-trained Model

Due to GitHub's 100MB file size limit, trained models are available via:

- **Google Drive**: [Add your link here after uploading]
- **OneDrive**: [Add your link here after uploading]

Place the downloaded `best_model.pth` file in this directory.

## Model Info

- **Architecture**: AttentionResNet18
- **Input Size**: 224×224×3
- **Output Classes**: 4 (glioma, meningioma, no_tumor, pituitary)
- **Framework**: PyTorch
- **File Size**: ~128 MB per checkpoint
