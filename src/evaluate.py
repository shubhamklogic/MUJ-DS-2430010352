import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.transfer_model import RetinopathyResNet50

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # 1. Load Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"
    
    _, _, test_loader = get_dataloaders(
        train_df, val_df, test_df, img_dir,
        batch_size=32, img_size=224, num_workers=2
    )
    
    # 2. Load Model & Weights
    model = RetinopathyResNet50(num_classes=5, freeze_backbone=False).to(device)
    weights_path = "models/best_finetuned_model.pth"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from '{weights_path}'")
    
    # 3. Predict on Test Set
    labels, preds = evaluate_model(model, test_loader, device)
    
    # 4. Metrics Calculation
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    # Quadratic Weighted Kappa (Standard DR metric)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    
    print("\n" + "=" * 50)
    print(f"QUADRATIC WEIGHTED KAPPA (QWK): {qwk:.4f}")
    print("=" * 50)
    
    # Classification Report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    # 5. Plot & Save Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Test Confusion Matrix (QWK: {qwk:.4f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.tight_layout()
    
    cm_path = "data/processed/test_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to '{cm_path}'")

if __name__ == "__main__":
    main()