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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.transfer_model import RetinopathyResNet50

def train_epoch(model, dataloader, criterion, optimizer, device):
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
        
    return running_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device):
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
            
    return running_loss / total, correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    STAGE1_EPOCHS = 3   # Head training
    STAGE2_EPOCHS = 7   # Fine-tuning layer4 + head
    IMG_SIZE = 224
    
    # Load splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"
    
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df, img_dir, 
        batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=2
    )
    
    # Initialize Model
    model = RetinopathyResNet50(num_classes=5, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    
    # ==========================================
    # STAGE 1: Train Head Only (Feature Extraction)
    # ==========================================
    print("\n--- Stage 1: Training Classification Head (Backbone Frozen) ---")
    optimizer = optim.AdamW(model.backbone.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, STAGE1_EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Stage 1 [Epoch {epoch}/{STAGE1_EPOCHS}] "
              f"| Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.4f} "
              f"| Val Loss: {v_loss:.4f} - Val Acc: {v_acc:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "models/best_finetuned_model.pth")
    
    # ==========================================
    # STAGE 2: Fine-Tuning Top Layers
    # ==========================================
    print("\n--- Stage 2: Fine-Tuning Layer4 + Classification Head ---")
    model.unfreeze_top_layers()
    
    # Use differential learning rates: lower for backbone, slightly higher for head
    optimizer = optim.AdamW([
        {'params': model.backbone.layer4.parameters(), 'lr': 1e-4},
        {'params': model.backbone.fc.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    for epoch in range(STAGE1_EPOCHS + 1, total_epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        
        scheduler.step(v_loss)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Stage 2 [Epoch {epoch}/{total_epochs}] "
              f"| Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.4f} "
              f"| Val Loss: {v_loss:.4f} - Val Acc: {v_acc:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "models/best_finetuned_model.pth")
            print("  --> Best model checkpoint updated!")
            
    print("\nBest fine-tuned weights saved to 'models/best_finetuned_model.pth'")
    
    # ==========================================
    # Step 3: Save Fine-Tuned Learning Curves
    # ==========================================
    epochs_range = range(1, total_epochs + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    plt.axvline(x=STAGE1_EPOCHS + 0.5, color='r', linestyle='--', label='Unfreeze Point')
    plt.title('Fine-Tuned ResNet50: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
    plt.axvline(x=STAGE1_EPOCHS + 0.5, color='r', linestyle='--', label='Unfreeze Point')
    plt.title('Fine-Tuned ResNet50: Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = "data/processed/finetuned_learning_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Fine-tuned curve plot saved to '{plot_path}'")

if __name__ == "__main__":
    main()