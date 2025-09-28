#!/usr/bin/env python3
"""
Complement ML Model Training Script
Trains a neural network for app recommendation based on usage patterns.
Exports the model to ONNX format for use in Rust.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import onnx
import os
from datetime import datetime

class ComplementRecommendationModel(nn.Module):
    """
    Neural network for predicting app recommendations based on context features.
    
    Input features (8 dimensions):
    - hour_of_day: 0-1 (normalized hour)
    - day_of_week: 0-1 (normalized day)
    - is_weekend: 0 or 1
    - time_since_last_use: 0-1 (inverted and normalized)
    - usage_frequency: 0-1 (normalized uses per week)
    - recent_apps_similarity: 0-1
    - active_window_context: 0-1
    - search_context: 0-1
    """
    
    def __init__(self, input_size=8, hidden_size=64, num_classes=50):
        super(ComplementRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

def generate_synthetic_training_data(num_samples=10000):
    """
    Generate synthetic training data for the recommendation model.
    In production, this would come from real usage data exported by Complement.
    """
    print("🎲 Generating synthetic training data...")
    
    # Define common app categories
    apps = [
        "Visual Studio Code", "Chrome", "Firefox", "Discord", "Teams", "Slack",
        "Steam", "Spotify", "Notepad++", "File Explorer", "Calculator", "Photos",
        "Word", "Excel", "PowerPoint", "Outlook", "Photoshop", "Figma",
        "Terminal", "Git Bash", "Docker", "VirtualBox", "Zoom", "Notion",
        "Telegram", "WhatsApp", "OBS Studio", "VLC Media Player", "7-Zip",
        "Audacity", "Blender", "Unity", "IntelliJ IDEA", "PyCharm", "Postman",
        "Jupyter", "Anaconda", "R Studio", "MATLAB", "AutoCAD", "SketchUp",
        "Adobe Reader", "Kindle", "Netflix", "YouTube", "Twitch", "Reddit",
        "Twitter", "LinkedIn", "GitHub Desktop", "Sourcetree"
    ]
    
    data = []
    
    for _ in range(num_samples):
        # Random context features
        hour_of_day = np.random.uniform(0, 1)  # 0-1 normalized
        day_of_week = np.random.uniform(0, 1)  # 0-1 normalized
        is_weekend = 1.0 if day_of_week > 5/6 else 0.0
        time_since_last_use = np.random.uniform(0, 1)
        usage_frequency = np.random.exponential(0.3)  # Most apps used infrequently
        usage_frequency = min(usage_frequency, 1.0)
        recent_apps_similarity = np.random.uniform(0, 1)
        active_window_context = np.random.uniform(0, 1)
        search_context = np.random.uniform(0, 1)
        
        # Select target app based on context (simulate realistic patterns)
        app_probabilities = np.ones(len(apps))
        
        # Work hours boost for productivity apps
        if 0.33 < hour_of_day < 0.75 and not is_weekend:  # 8 AM - 6 PM weekdays
            for i, app in enumerate(apps):
                if any(work_term in app.lower() for work_term in 
                      ['code', 'word', 'excel', 'teams', 'outlook', 'notion', 'git']):
                    app_probabilities[i] *= 3.0
        
        # Evening boost for entertainment apps
        if hour_of_day > 0.75:  # After 6 PM
            for i, app in enumerate(apps):
                if any(fun_term in app.lower() for fun_term in 
                      ['steam', 'spotify', 'netflix', 'youtube', 'discord', 'twitch']):
                    app_probabilities[i] *= 2.5
        
        # Weekend boost for creative/personal apps
        if is_weekend:
            for i, app in enumerate(apps):
                if any(weekend_term in app.lower() for weekend_term in 
                      ['photoshop', 'blender', 'steam', 'spotify', 'photos']):
                    app_probabilities[i] *= 2.0
        
        # Frequency boost (higher usage frequency = higher probability)
        app_probabilities *= (1 + usage_frequency * 2)
        
        # Normalize probabilities
        app_probabilities /= app_probabilities.sum()
        
        # Select app
        target_app_idx = np.random.choice(len(apps), p=app_probabilities)
        
        data.append([
            hour_of_day, day_of_week, is_weekend, time_since_last_use,
            usage_frequency, recent_apps_similarity, active_window_context, 
            search_context, target_app_idx
        ])
    
    columns = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'time_since_last_use',
        'usage_frequency', 'recent_apps_similarity', 'active_window_context',
        'search_context', 'target_app'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    print(f"✅ Generated {len(df)} training samples")
    return df, apps

def train_model(df, apps):
    """Train the neural network model."""
    print("🧠 Training neural network model...")
    
    # Prepare features and labels
    X = df[['hour_of_day', 'day_of_week', 'is_weekend', 'time_since_last_use',
            'usage_frequency', 'recent_apps_similarity', 'active_window_context',
            'search_context']].values
    y = df['target_app'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Initialize model
    model = ComplementRecommendationModel(input_size=8, num_classes=len(apps))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 100
    batch_size = 128
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())
        print(f"🎯 Model accuracy: {accuracy:.4f}")
    
    return model

def export_to_onnx(model, apps):
    """Export the trained model to ONNX format."""
    print("📦 Exporting model to ONNX format...")
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 8)  # Batch size 1, 8 features
    
    # Export paths
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    onnx_path = os.path.join(model_dir, "complement_recommendation_model.onnx")
    apps_path = os.path.join(model_dir, "app_labels.txt")
    
    # Export model to ONNX
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
    with open(apps_path, 'w') as f:
        for app in apps:
            f.write(f"{app}\n")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model exported successfully: {onnx_path}")
        print(f"✅ App labels saved: {apps_path}")
        return onnx_path, apps_path
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None, None

def create_model_info(onnx_path, apps_path, apps):
    """Create model information file."""
    info = {
        "model_name": "Complement App Recommendation Model",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "input_features": [
            "hour_of_day (0-1 normalized)",
            "day_of_week (0-1 normalized)", 
            "is_weekend (0 or 1)",
            "time_since_last_use (0-1 inverted/normalized)",
            "usage_frequency (0-1 normalized)",
            "recent_apps_similarity (0-1)",
            "active_window_context (0-1)",
            "search_context (0-1)"
        ],
        "num_classes": len(apps),
        "model_path": onnx_path,
        "labels_path": apps_path,
        "description": "Neural network trained to predict app recommendations based on user context and usage patterns."
    }
    
    info_path = os.path.join("models", "model_info.json")
    import json
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📋 Model info saved: {info_path}")

def main():
    """Main training pipeline."""
    print("🚀 Starting Complement ML Model Training")
    print("=" * 50)
    
    # Generate training data
    df, apps = generate_synthetic_training_data(num_samples=15000)
    
    # Train model
    model = train_model(df, apps)
    
    # Export to ONNX
    onnx_path, apps_path = export_to_onnx(model, apps)
    
    if onnx_path and apps_path:
        # Create model info
        create_model_info(onnx_path, apps_path, apps)
        
        print("\n🎉 Model training and export completed successfully!")
        print(f"📁 Model files created in: {os.path.abspath('models')}")
        print("\nNext steps:")
        print("1. Copy the ONNX model to your Rust project")
        print("2. Update the Rust code to load and use the model")
        print("3. Test the ML-powered recommendations!")
    else:
        print("❌ Model export failed")

if __name__ == "__main__":
    main()