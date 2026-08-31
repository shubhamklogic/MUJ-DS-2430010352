import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.baseline_model import BaselineRetinopathyCNN

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    # 1. Device configuration (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    
    # 3. Load Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"
    
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df, img_dir, 
        batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=2
    )
    
    # 4. Initialize Model, Loss, Optimizer
    model = BaselineRetinopathyCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Track metrics across epochs
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print("\n--- Starting Baseline Training ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"| Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
    # 5. Save baseline model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_cnn.pth")
    print("\nSaved baseline weights to 'models/baseline_cnn.pth'")
    
    # 6. Plot Learning Curves for Bias vs. Variance Analysis
    os.makedirs("data/processed", exist_ok=True)
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Baseline: Training vs. Validation Loss (Bias / Variance)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Baseline: Training vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = "data/processed/baseline_bias_variance_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Learning curve plot saved to '{plot_path}'")

if __name__ == "__main__":
    main()