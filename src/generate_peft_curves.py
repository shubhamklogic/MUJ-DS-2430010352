import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

def generate_peft_training_curves():
    csv_path = "results/peft_training_history.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}. Please train PEFT model first.")
        
    df = pd.read_csv(csv_path)
    epochs = range(1, len(df) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    
    # ------------------ Panel A: Loss Progression ------------------
    ax1.plot(epochs, df["train_loss"], label="Train Ordinal Loss", marker="o", color="#1f77b4", linewidth=2.2)
    ax1.plot(epochs, df["val_loss"], label="Val Ordinal Loss", marker="s", color="#ff7f0e", linewidth=2.2)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Ordinal Loss", fontsize=11, fontweight="bold")
    ax1.set_title("(A) PEFT LoRA Convergence Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xticks(list(epochs))
    ax1.set_ylim(0.30, 0.55)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # ------------------ Panel B: Twin-Axis Accuracy & QWK ------------------
    color_acc = "#2ca02c"
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    line1 = ax2.plot(epochs, df["val_acc"] * 100, label="Val Accuracy (%)", marker="o", color=color_acc, linewidth=2.2)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(20, 70)
    ax2.set_xticks(list(epochs))
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    ax2_twin = ax2.twinx()
    color_qwk = "#9467bd"
    ax2_twin.set_ylabel("Validation QWK Score", color=color_qwk, fontsize=11, fontweight="bold")
    line2 = ax2_twin.plot(epochs, df["val_qwk"], label="Val QWK", marker="^", color=color_qwk, linewidth=2.2)
    ax2_twin.tick_params(axis="y", labelcolor=color_qwk)
    ax2_twin.set_ylim(0.0, 0.70)
    
    # Combined legend for Panel B
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower right", framealpha=0.9)
    ax2.set_title("(B) Validation Accuracy & QWK Progression", fontsize=12, fontweight="bold")
    
    plt.suptitle("Parameter-Efficient Foundation Adaptation (LoRA + Ordinal Cumulative Head)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/peft_training_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Visualization Engine] Successfully saved high-clarity curve to: '{out_path}'")

if __name__ == "__main__":
    generate_peft_training_curves()