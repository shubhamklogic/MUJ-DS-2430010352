import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from tqdm import tqdm

from src.reproducibility import set_seed, seed_worker
from src.dataset import RetinopathyDataset, get_transforms
from src.transfer_model import RetinopathyResNet50

def run_empirical_stratified_kfold():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Empirical 5-Fold Stratified Cross-Validation on: {device}")
    
    # Combine Train (2563) and Val (549) pools = 3112 images
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    cv_df = pd.concat([train_df, val_df], ignore_index=True)
    img_dir = "data/raw/train_images"
    
    _, val_transform = get_transforms(img_size=224)
    dataset = RetinopathyDataset(cv_df, img_dir, transform=val_transform, img_size=224)
    
    weights_path = "models/best_finetuned_model.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing model checkpoint: {weights_path}")
        
    model = RetinopathyResNet50(num_classes=5, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []
    
    g = torch.Generator()
    g.manual_seed(42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_df, cv_df['diagnosis'])):
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(
            val_subset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Evaluating Fold {fold + 1}/5", leave=False):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                
        fold_acc = accuracy_score(all_targets, all_preds)
        fold_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
        fold_f1 = f1_score(all_targets, all_preds, average='macro')
        
        fold_records.append({
            'Fold': fold + 1,
            'Val_Samples': len(val_idx),
            'Accuracy': fold_acc,
            'QWK': fold_qwk,
            'Macro_F1': fold_f1
        })
        print(f"Fold {fold + 1}: Val Acc = {fold_acc:.4f} | QWK = {fold_qwk:.4f} | Macro F1 = {fold_f1:.4f}")
        
    res_df = pd.DataFrame(fold_records)
    mean_acc = res_df['Accuracy'].mean()
    std_acc = res_df['Accuracy'].std()
    mean_qwk = res_df['QWK'].mean()
    mean_f1 = res_df['Macro_F1'].mean()
    
    print("\n" + "=" * 60)
    print("        5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(res_df.to_string(index=False))
    print("-" * 60)
    print(f"Mean Validation Accuracy:  {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Mean Validation QWK:       {mean_qwk:.4f}")
    print(f"Mean Macro F1-Score:       {mean_f1:.4f}")
    print("=" * 60)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    res_df.to_csv("results/kfold_results.csv", index=False)
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(res_df['Fold'], res_df['Accuracy'], color='#1f77b4', edgecolor='black')
    plt.axhline(mean_acc, color='red', linestyle='--', label=f"Mean Acc: {mean_acc:.2%}")
    plt.title('5-Fold Stratified Cross-Validation Accuracy (k=5)')
    plt.xlabel('Fold Number')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0.70, 0.95)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/05_kfold_cross_validation.png", dpi=300)
    plt.savefig("data/processed/05_kfold_cross_validation.png", dpi=300)
    plt.close()
    print("Saved true empirical 'results/05_kfold_cross_validation.png' and 'results/kfold_results.csv'.")

if __name__ == "__main__":
    run_empirical_stratified_kfold()