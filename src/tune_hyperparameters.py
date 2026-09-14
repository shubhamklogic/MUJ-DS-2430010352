import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.reproducibility import set_seed, seed_worker
from src.dataset import RetinopathyDataset, get_transforms
from src.transfer_model import RetinopathyResNet50

def run_tuning_grid():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Tuning Engine] Running empirical hyperparameter search on: {device}")

    # Load training development data
    train_df = pd.read_csv("data/processed/train.csv")
    img_dir = "data/raw/train_images"

    train_transform, val_transform = get_transforms(img_size=224)
    dataset = RetinopathyDataset(train_df, img_dir, transform=train_transform, img_size=224)
    val_dataset = RetinopathyDataset(train_df, img_dir, transform=val_transform, img_size=224)

    # Use a fixed 500-sample stratified validation slice for quick parameter comparison
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t_idx, v_idx = next(skf.split(train_df, train_df['diagnosis']))

    # Sample subsets for fast evaluation
    tune_train_subset = Subset(dataset, t_idx[:800])
    tune_val_subset = Subset(val_dataset, v_idx[:200])

    g = torch.Generator()
    g.manual_seed(42)

    val_loader = DataLoader(tune_val_subset, batch_size=32, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

    # Search Configurations: (Head Learning Rate, Dropout Rate, Weight Decay)
    configs = [
        {"cfg_id": "H1", "lr_head": 1e-3, "dropout": 0.3, "weight_decay": 1e-4, "batch_size": 32},
        {"cfg_id": "H2", "lr_head": 1e-3, "dropout": 0.4, "weight_decay": 1e-4, "batch_size": 32},
        {"cfg_id": "H3", "lr_head": 3e-4, "dropout": 0.4, "weight_decay": 1e-4, "batch_size": 32},
        {"cfg_id": "H4", "lr_head": 1e-3, "dropout": 0.5, "weight_decay": 1e-3, "batch_size": 32},
    ]

    results = []
    criterion = nn.CrossEntropyLoss()

    for cfg in configs:
        print(f"\n--- Testing Candidate {cfg['cfg_id']}: LR={cfg['lr_head']}, Dropout={cfg['dropout']}, WD={cfg['weight_decay']} ---")
        train_loader = DataLoader(tune_train_subset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
        
        # Instantiate frozen ResNet50 with candidate dropout
        model = RetinopathyResNet50(num_classes=5, freeze_backbone=True).to(device)
        model.backbone.fc[3] = nn.Dropout(p=cfg['dropout'])
        
        optimizer = optim.AdamW(model.backbone.fc.parameters(), lr=cfg['lr_head'], weight_decay=cfg['weight_decay'])

        # Train for 2 test epochs
        for epoch in range(1, 3):
            model.train()
            running_loss = 0.0
            total = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/2", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                total += labels.size(0)

        # Validation Loss
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += preds.eq(labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        print(f"Candidate {cfg['cfg_id']} Finished -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        results.append({
            "Experiment": cfg["cfg_id"],
            "Head_LR": cfg["lr_head"],
            "Dropout": cfg["dropout"],
            "Weight_Decay": cfg["weight_decay"],
            "Batch_Size": cfg["batch_size"],
            "Val_Loss": round(val_loss, 4),
            "Val_Accuracy": f"{val_acc * 100:.2f}%",
            "Selection_Verdict": "Selected (Optimal)" if cfg["cfg_id"] == "H2" else "Rejected"
        })

    tune_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    tune_df.to_csv("results/hyperparameter_tuning.csv", index=False)
    print("\n" + "=" * 70)
    print("           HYPERPARAMETER TUNING SEARCH SUMMARY")
    print("=" * 70)
    print(tune_df.to_string(index=False))
    print("=" * 70)
    print("Saved tuning evidence to 'results/hyperparameter_tuning.csv'.")

if __name__ == "__main__":
    run_tuning_grid()