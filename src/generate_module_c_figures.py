import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_doe_ablation_plot():
    csv_path = "results/ablation.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing ablation table: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"[DoE Figure Engine] Loaded ablation matrix with {len(df)} experimental conditions.")
    
    exp_ids = df['Exp_ID'].tolist()
    
    # Clean, single-line configuration labels (NO newlines to prevent diagonal collisions)
    configs = [
        "E1: ResNet-50 (Baseline)",
        "E2: Swin Backbone (Linear)",
        "E3: Full Fine-Tune (88M)",
        "E4: PEFT LoRA (r=8, CE)",
        "E5: PEFT LoRA (r=16, CE)",
        "E6: Proposed PEFT (Ordinal)"
    ]
    
    qwk_scores = df['QWK_Metric'].astype(float).tolist()
    macro_f1 = df['Macro_F1'].astype(float).tolist()
    
    # Color scheme: Baseline (Blue), Intermediates (Grey), Proposed (Green)
    bar_colors = ['#1f77b4', '#7f7f7f', '#7f7f7f', '#7f7f7f', '#7f7f7f', '#2ca02c']
    x_indices = np.arange(len(exp_ids))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))
    plt.subplots_adjust(wspace=0.25, bottom=0.28)
    
    # -------------------------------------------------------------
    # Panel A: Quadratic Weighted Kappa (QWK) Progression
    # -------------------------------------------------------------
    bars_qwk = axes[0].bar(x_indices, qwk_scores, color=bar_colors, edgecolor='black', width=0.52, linewidth=1.2)
    axes[0].set_title('(a) Clinical Agreement Progression (QWK)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel('Quadratic Weighted Kappa (QWK)', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0.80, 0.94)
    
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(configs, fontsize=9.0, fontweight='bold', rotation=30, ha='right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value annotations on top of bars
    for bar, val in zip(bars_qwk, qwk_scores):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, val + 0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold'
        )
        
    axes[0].axhline(y=qwk_scores[0], color='#1f77b4', linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline Reference (0.8616)')
    axes[0].legend(loc='upper left', frameon=True)

    # -------------------------------------------------------------
    # Panel B: Macro-Averaged F1-Score Progression
    # -------------------------------------------------------------
    bars_f1 = axes[1].bar(x_indices, macro_f1, color=bar_colors, edgecolor='black', width=0.52, linewidth=1.2)
    axes[1].set_title('(b) Unweighted Macro F1-Score Progression', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_ylabel('Macro F1-Score', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0.50, 0.66)
    
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(configs, fontsize=9.0, fontweight='bold', rotation=30, ha='right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar, val in zip(bars_f1, macro_f1):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, val + 0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold'
        )
        
    axes[1].axhline(y=macro_f1[0], color='#1f77b4', linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline Reference (0.6091)')
    axes[1].legend(loc='upper left', frameon=True)

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/doe_ablation_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Successfully re-rendered and saved: '{out_path}'")

if __name__ == "__main__":
    generate_doe_ablation_plot()