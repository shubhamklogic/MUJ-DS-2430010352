import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from src.reproducibility import set_seed, seed_worker
from src.dataset import RetinopathyDataset, get_transforms
from src.innovation_peft import RetinopathyFoundationPEFT, CumulativeOrdinalLoss

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in tqdm(loader, desc="Training PEFT", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total

def evaluate_peft(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating PEFT", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # Cumulative ordinal stage prediction
            preds = model.ordinal_head.predict_stages(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, acc, qwk

def main():
    # Enforce deterministic seeding
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CO5 PEFT Engine] Initializing Training Loop on: {device}")

    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    IMG_SIZE = 224

    # 1. Load Clinical Splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    img_dir = "data/raw/train_images"

    train_tf, val_tf = get_transforms(img_size=IMG_SIZE)
    train_ds = RetinopathyDataset(train_df, img_dir, transform=train_tf, img_size=IMG_SIZE)
    val_ds = RetinopathyDataset(val_df, img_dir, transform=val_tf, img_size=IMG_SIZE)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )

    # 2. Instantiate PEFT Architecture
    model = RetinopathyFoundationPEFT(num_classes=5, lora_r=16, lora_alpha=32).to(device)
    criterion = CumulativeOrdinalLoss(num_classes=5)

    # Optimize only trainable parameters (LoRA adapters + Ordinal Cumulative Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs("models", exist_ok=True)
    best_val_loss = float("inf")
    save_path = "models/best_peft_model.pth"

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_qwk": []}

    print("\n" + "=" * 65)
    print("      STARTING PARAMETER-EFFICIENT ORDINAL FOUNDATION TRAINING")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_qwk = evaluate_peft(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_qwk"].append(v_qwk)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"| Train Ordinal Loss: {tr_loss:.4f} "
            f"| Val Loss: {v_loss:.4f} "
            f"| Val Acc: {v_acc:.2%} "
            f"| Val QWK: {v_qwk:.4f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best PEFT checkpoint saved: '{save_path}' (Val Loss: {v_loss:.4f})")

    # Export metrics history to results/
    os.makedirs("results", exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv("results/peft_training_history.csv", index=False)

    print("\n" + "=" * 65)
    print(f"PEFT Training Finished Successfully!")
    print(f"Saved Checkpoint: '{save_path}'")
    print(f"Saved History:    'results/peft_training_history.csv'")
    print("=" * 65)

if __name__ == "__main__":
    main()