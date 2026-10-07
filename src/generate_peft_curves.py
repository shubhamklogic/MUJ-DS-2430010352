import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_peft_curves():
    csv_path = "results/peft_training_history.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing training history log: '{csv_path}'")
        
    df = pd.read_csv(csv_path)
    print(f"[Curve Engine] Loaded training history with {len(df)} epochs.")
    
    epochs = df['epoch'].astype(int).tolist()
    train_loss = df['train_loss'].astype(float).tolist()
    val_acc = (df['val_acc'].astype(float) * 100.0).tolist()
    val_qwk = df['val_qwk'].astype(float).tolist()
    
    # Locate peak validation checkpoint
    best_idx = int(np.argmax(val_qwk))
    best_epoch = epochs[best_idx]
    best_qwk = val_qwk[best_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plt.subplots_adjust(wspace=0.28)
    
    # -------------------------------------------------------------
    # Panel A: Cumulative Ordinal BCE Loss Convergence
    # -------------------------------------------------------------
    axes[0].plot(epochs, train_loss, marker='o', linewidth=2.2, color='#1f77b4', label='Train Loss (Cumulative BCE)')
    axes[0].set_title('(a) Training Optimization Loss Convergence', fontsize=11, fontweight='bold', pad=12)
    axes[0].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Cumulative Ordinal BCE Loss', fontsize=10, fontweight='bold')
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right', frameon=True)
    
    # Annotate initial and converged loss
    axes[0].annotate(f'Initial: {train_loss[0]:.4f}', xy=(epochs[0], train_loss[0]),
                     xytext=(epochs[0] + 0.5, train_loss[0] - 0.02),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                     fontsize=9, fontweight='bold')
    axes[0].annotate(f'Converged: {train_loss[-1]:.4f}', xy=(epochs[-1], train_loss[-1]),
                     xytext=(epochs[-1] - 3.8, train_loss[-1] + 0.03),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                     fontsize=9, fontweight='bold')

    # -------------------------------------------------------------
    # Panel B: Validation Metrics (Accuracy & QWK)
    # -------------------------------------------------------------
    axes[1].plot(epochs, val_qwk, marker='s', linewidth=2.2, color='#2ca02c', label='Val Quadratic Weighted Kappa (QWK)')
    axes[1].plot(epochs, [a / 100.0 for a in val_acc], marker='^', linewidth=1.8, linestyle='--', color='#ff7f0e', label='Val Accuracy')
    
    # Highlight best checkpoint epoch
    axes[1].axvline(x=best_epoch, color='#d62728', linestyle=':', linewidth=1.8, label=f'Best Checkpoint (Epoch {best_epoch}: {best_qwk:.4f})')
    axes[1].scatter([best_epoch], [best_qwk], color='#d62728', s=80, zorder=5)
    
    axes[1].set_title('(b) Validation Metric Generalization (549 Scans)', fontsize=11, fontweight='bold', pad=12)
    axes[1].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Metric Score [0.0 - 1.0]', fontsize=10, fontweight='bold')
    axes[1].set_ylim(0.70, 0.94)
    axes[1].set_xticks(epochs)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower right', frameon=True)
    
    axes[1].annotate(f'Peak QWK: {best_qwk:.4f}', xy=(best_epoch, best_qwk),
                     xytext=(best_epoch - 3.2, best_qwk + 0.025),
                     arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
                     fontsize=9, fontweight='bold', color='#d62728')

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/peft_training_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Successfully generated 15-epoch training curve: '{out_path}'")

if __name__ == "__main__":
    generate_peft_curves()