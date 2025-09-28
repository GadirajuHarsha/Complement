#!/usr/bin/env python3
"""
Quick ONNX Model Training Script for Complement
Creates a simple but functional ML model for app recommendations.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class SimpleRecommendationModel(nn.Module):
    """Simplified neural network for app recommendations"""
    
    def __init__(self, input_size, num_classes):
        super(SimpleRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_quick_model():
    """Create and save a quick ONNX model for testing"""
    
    # Common productivity apps for demonstration
    apps = [
        "Visual Studio Code", "Chrome", "Firefox", "Word", "Excel", 
        "PowerPoint", "Teams", "Outlook", "Notepad", "Calculator",
        "Spotify", "Discord", "Steam", "Photoshop", "Blender"
    ]
    
    print(f"🚀 Creating quick model with {len(apps)} app classes...")
    
    # Create model
    model = SimpleRecommendationModel(input_size=8, num_classes=len(apps))
    
    # Generate some simple training data
    num_samples = 1000
    X = torch.randn(num_samples, 8)  # Random features
    y = torch.randint(0, len(apps), (num_samples,))  # Random labels
    
    # Quick training (just a few epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("🎯 Quick training (10 epochs)...")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Export to ONNX
    print("📦 Exporting to ONNX...")
    dummy_input = torch.randn(1, 8)
    
    onnx_path = "models/complement_recommendation_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['context_features'],
        output_names=['app_probabilities'],
        dynamic_axes={
            'context_features': {0: 'batch_size'},
            'app_probabilities': {0: 'batch_size'}
        }
    )
    
    # Save app labels
    labels_path = "models/app_labels.txt"
    with open(labels_path, 'w') as f:
        for app in apps:
            f.write(f"{app}\n")
    
    # Create model info
    info = {
        "model_name": "Complement Quick Recommendation Model",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "num_classes": len(apps),
        "input_features": 8,
        "description": "Quick ONNX model for app recommendations"
    }
    
    with open("models/model_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Quick model created successfully!")
    print(f"📁 Model: {os.path.abspath(onnx_path)}")
    print(f"📁 Labels: {os.path.abspath(labels_path)}")
    print(f"🎯 Model has {len(apps)} app classes")

if __name__ == "__main__":
    create_quick_model()