import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
import timm
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from src.dataset import RetinopathyDataset, get_transforms

class RetinopathyFoundationPEFT(nn.Module):
    """
    Parameter-Efficient Retinal Foundation Architecture.
    Integrates pre-trained Swin Transformer with Low-Rank Adaptation (LoRA)
    and a 4-cut Ordinal Cumulative Logistic Head.
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=5, lora_r=16, lora_alpha=32):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.ordinal_head = nn.Linear(in_features, num_classes - 1)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.ordinal_head(features)

class CumulativeOrdinalLoss(nn.Module):
    """
    Binary Cross-Entropy over monotonic ordinal threshold indicators.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        cuts = torch.arange(4, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > cuts).float()
        return self.bce(logits, binary_targets)

def evaluate_metrics(model, loader, device):
    """Evaluates discrete predictions against ground truth labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).sum(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return acc, qwk, all_preds, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device Engine] Running training execution on: {device}")
    
    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"
    
    train_tf, val_tf = get_transforms(img_size=224)
    train_loader = DataLoader(RetinopathyDataset(train_df, img_dir, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(RetinopathyDataset(val_df, img_dir, val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(RetinopathyDataset(test_df, img_dir, val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = RetinopathyFoundationPEFT(lora_r=16, lora_alpha=32).to(device)
    criterion = CumulativeOrdinalLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Optional persistent Google Drive path if running in cloud
    drive_dir = "/content/drive/MyDrive/DR_Project_Results"
    has_drive = os.path.exists("/content/drive/MyDrive")
    if has_drive:
        os.makedirs(drive_dir, exist_ok=True)
    
    best_val_qwk = -1.0
    history = []
    
    print("\n--- Commencing Multi-Epoch Convergence (Swin + LoRA Foundation) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        train_loss = running_loss / len(train_df)
        val_acc, val_qwk, _, _ = evaluate_metrics(model, val_loader, device)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_qwk': val_qwk
        })
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(model.state_dict(), "models/best_peft_model.pth")
            if has_drive:
                torch.save(model.state_dict(), os.path.join(drive_dir, "best_peft_model.pth"))
            print(f"  >>> Checkpoint updated! Best Val QWK: {best_val_qwk:.4f}")
            
    # Save training history log
    pd.DataFrame(history).to_csv("results/peft_training_history.csv", index=False)
    
    print("\n=======================================================")
    print("  EVALUATING BEST CHECKPOINT ON 550 HOLD-OUT TEST SCANS")
    print("=======================================================")
    model.load_state_dict(torch.load("models/best_peft_model.pth"))
    test_acc, test_qwk, y_pred, y_true = evaluate_metrics(model, test_loader, device)
    
    print(f"\n[FINAL EMPIRICAL TEST RESULT]")
    print(f"Test Accuracy:                  {test_acc * 100:.2f}%")
    print(f"Quadratic Weighted Kappa (QWK): {test_qwk:.4f}")
    
    # Save discrete test predictions and ground truth
    np.save("results/y_true_test.npy", y_true)
    np.save("results/y_pred_test.npy", y_pred)
    if has_drive:
        np.save(os.path.join(drive_dir, "y_true_test.npy"), y_true)
        np.save(os.path.join(drive_dir, "y_pred_test.npy"), y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'], digits=4)
    print("\nTest Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    
    metrics_str = f"Test Accuracy: {test_acc:.4f}\nTest QWK: {test_qwk:.4f}\n\nConfusion Matrix:\n{cm}\n\nReport:\n{report}"
    with open("results/true_test_metrics.txt", "w") as f:
        f.write(metrics_str)
    if has_drive:
        with open(os.path.join(drive_dir, "true_test_metrics.txt"), "w") as f:
            f.write(metrics_str)
            
    print("\n[✓] Training, checkpointing, and test evaluation pipeline fully executed.")

if __name__ == "__main__":
    main()