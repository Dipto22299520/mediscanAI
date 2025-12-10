# 🧠 MediScanAI - Brain Tumor Classification System

An AI-powered brain tumor classification system using attention-augmented deep learning with a React web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![React](https://img.shields.io/badge/React-18-61dafb.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Attention-Based Deep Learning**: ResNet18 with channel and spatial attention mechanisms
- **4 Tumor Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **User-Friendly Web Interface**: Simple React frontend for MRI image upload
- **REST API**: Flask backend for real-time predictions
- **High Accuracy**: ~92% validation accuracy

## 📁 Project Structure

```
mediscanai/
├── brain_tumor/
│   ├── train.py              # Training script with attention blocks
│   └── train_brisc.py        # BRISC2025 dataset training wrapper
├── backend/
│   └── app.py                # Flask API server
├── brain-tumor-app/          # React frontend
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── App.css          # Styling
│   ├── public/
│   └── package.json
├── outputs/
│   └── brisc_model/
│       ├── classes.json     # Class names mapping
│       └── README.md        # Model info
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- pip package manager

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Dipto22299520/mediscanAI.git
cd mediscanAI
```

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Get the Model

**Option A: Train Your Own Model**

```bash
python brain_tumor/train.py \
  --data_dir ./path/to/dataset/train \
  --output_dir ./outputs/brisc_model \
  --epochs 50 \
  --batch_size 32 \
  --pretrained
```

**Option B: Download Pre-trained Model**

Download `best_model.pth` and place it in `outputs/brisc_model/` (see outputs/brisc_model/README.md for links)

### 4️⃣ Start Backend Server

```bash
cd backend
python app.py
```

Backend runs on `http://127.0.0.1:5000` ✅

### 5️⃣ Start Frontend

Open a new terminal:

```bash
cd brain-tumor-app
npm install
npm start
```

Frontend opens at `http://localhost:3000` ✅

## 🏗️ Model Architecture

The model uses **ResNet18** backbone enhanced with attention mechanisms:

```
Input Image (224×224×3)
        ↓
ResNet18 Conv1 + BN + ReLU + MaxPool
        ↓
ResNet18 Layer1 (64 channels)
        ↓
ResNet18 Layer2 (128 channels)
        ↓
🔍 Attention Block (Channel + Spatial)
        ↓
ResNet18 Layer3 (256 channels)
        ↓
ResNet18 Layer4 (512 channels)
        ↓
🔍 Attention Block (Channel + Spatial)
        ↓
Global Average Pooling
        ↓
Fully Connected Layer (4 classes)
        ↓
Output: [glioma, meningioma, no_tumor, pituitary]
```

### Attention Mechanism

1. **Channel Attention**: Learns which feature channels are important
2. **Spatial Attention**: Learns which spatial locations are important

## 🎓 Training Your Own Model

### Dataset Structure

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── glioma/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

### Training Command

```bash
python brain_tumor/train.py \
  --data_dir ./dataset/train \
  --output_dir ./outputs/my_model \
  --epochs 50 \
  --batch_size 32 \
  --img_size 224 \
  --lr 0.0001 \
  --pretrained
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to training data folder | Required |
| `--output_dir` | Where to save model checkpoints | `./outputs` |
| `--epochs` | Number of training epochs | `30` |
| `--batch_size` | Batch size | `32` |
| `--lr` | Learning rate | `0.0001` |
| `--img_size` | Input image size | `224` |
| `--val_split` | Validation split ratio | `0.2` |
| `--pretrained` | Use ImageNet pretrained weights | `False` |

## 🔌 API Documentation

### Health Check

```http
GET http://127.0.0.1:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "classes": ["glioma", "meningioma", "no_tumor", "pituitary"]
}
```

### Predict Tumor Type

```http
POST http://127.0.0.1:5000/api/predict
Content-Type: multipart/form-data
```

**Body:**
- `image`: MRI image file (jpg, png, etc.)

**Response:**
```json
{
  "prediction": "glioma",
  "confidence": 0.9523,
  "probabilities": {
    "glioma": 0.9523,
    "meningioma": 0.0234,
    "no_tumor": 0.0123,
    "pituitary": 0.0120
  }
}
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~92% |
| Number of Classes | 4 |
| Model Size | ~128 MB |

## 🛠️ Technologies Used

### Backend
- Python 3.8+
- PyTorch
- Flask
- torchvision
- Pillow

### Frontend
- React 18
- JavaScript (ES6+)
- CSS3

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Dipto22299520**

- GitHub: [@Dipto22299520](https://github.com/Dipto22299520)

## ⚠️ Disclaimer

This is a research project and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical decisions.

## 📞 Contact

For questions or issues, please open an issue on GitHub.
