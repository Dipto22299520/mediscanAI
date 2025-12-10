"""
Flask API for brain tumor classification
"""
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# -------------------------
# Attention Block (same as training)
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        x = x * y
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        cat = torch.cat([max_pool, mean_pool], dim=1)
        s = self.spatial_conv(cat)
        s = self.spatial_sigmoid(s)
        out = x * s
        return out

class AttentionResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
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
        self.att2 = AttentionBlock(in_channels=128)
        self.att4 = AttentionBlock(in_channels=512)
        self.avgpool = backbone.avgpool
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.att2(x)
        x = self.features[6](x)
        x = self.features[7](x)
        x = self.att4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = r"D:\mediscanai\outputs\brisc_model\best_model.pth"
CLASSES_PATH = r"D:\mediscanai\outputs\brisc_model\classes.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load classes
with open(CLASSES_PATH, 'r') as f:
    classes = json.load(f)

# Load model
model = AttentionResNet18(num_classes=len(classes))
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"✅ Model loaded successfully!")
print(f"Classes: {classes}")

# -------------------------
# API Routes
# -------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "classes": classes})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = {classes[i]: float(probabilities[0][i]) for i in range(len(classes))}
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence_score,
            "probabilities": all_probs
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)