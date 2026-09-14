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
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score
)
from tqdm import tqdm

from src.reproducibility import set_seed, seed_worker
from src.dataset import get_dataloaders
from src.transfer_model import RetinopathyResNet50
from src.baseline_model import BaselineRetinopathyCNN

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def calculate_specificity_per_class(cm):
    """
    Computes One-vs-Rest Specificity (TNR = TN / (TN + FP)) for each class.
    """
    num_classes = cm.shape[0]
    specificities = []
    total_samples = np.sum(cm)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return specificities

def evaluate_full_medical_suite():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation Engine] Evaluating Test Set on: {device}")

    # 1. Load Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"

    g = torch.Generator()
    g.manual_seed(42)

    _, _, test_loader = get_dataloaders(
        train_df, val_df, test_df, img_dir,
        batch_size=32, img_size=224, num_workers=2
    )

    # 2. Parameter Audit
    baseline = BaselineRetinopathyCNN(num_classes=5)
    b_total, b_train, b_froz = count_parameters(baseline)

    resnet = RetinopathyResNet50(num_classes=5, freeze_backbone=False)
    weights_path = "models/best_finetuned_model.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    resnet.load_state_dict(torch.load(weights_path, map_location=device))
    resnet.to(device)
    resnet.eval()

    r_total, r_train, r_froz = count_parameters(resnet)

    print("\n" + "=" * 60)
    print("            MODEL PARAMETER AUDIT")
    print("=" * 60)
    print(f"Baseline CNN:     Total={b_total:,} | Trainable={b_train:,} | Frozen={b_froz}")
    print(f"ResNet-50 Model:  Total={r_total:,} | Trainable={r_train:,} | Frozen={r_froz}")
    print("=" * 60)

    # 3. Model Inference on 550 Test Images
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing 550 Images", leave=False):
            images = images.to(device)
            outputs = resnet(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    # 4. Compute Metrics
    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative DR (4)']
    acc = accuracy_score(labels, preds)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    cm = confusion_matrix(labels, preds)
    specs = calculate_specificity_per_class(cm)
    macro_spec = np.mean(specs)

    # One-vs-Rest ROC-AUC
    auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    auc_weighted = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')

    # Per-Class Metrics DataFrame
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    per_class_rows = []
    for i, name in enumerate(class_names):
        per_class_rows.append({
            "Class": name,
            "Precision": round(report_dict[name]["precision"], 4),
            "Recall (Sensitivity)": round(report_dict[name]["recall"], 4),
            "Specificity": round(specs[i], 4),
            "F1-Score": round(report_dict[name]["f1-score"], 4),
            "Support": int(report_dict[name]["support"])
        })

    metrics_df = pd.DataFrame(per_class_rows)
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/test_medical_metrics.csv", index=False)

    print("\n" + "=" * 65)
    print("    COMPREHENSIVE MEDICAL EVALUATION METRICS (550 TEST SCANS)")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("-" * 65)
    print(f"Overall Test Accuracy:        {acc * 100:.2f}%")
    print(f"Quadratic Weighted Kappa:     {qwk:.4f}")
    print(f"Macro Average Specificity:    {macro_spec:.4f}")
    print(f"Macro One-vs-Rest ROC-AUC:    {auc_macro:.4f}")
    print(f"Weighted One-vs-Rest ROC-AUC: {auc_weighted:.4f}")
    print("=" * 65)

    # Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4'],
                yticklabels=['0', '1', '2', '3', '4'])
    plt.title(f'Test Confusion Matrix (QWK: {qwk:.4f}, Acc: {acc:.2%})')
    plt.xlabel('Predicted Stage')
    plt.ylabel('Ground Truth Stage')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved 'results/confusion_matrix.png' and 'results/test_medical_metrics.csv'.")

if __name__ == "__main__":
    evaluate_full_medical_suite()