import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

def generate_ablation_plot():
    csv_path = "results/ablation.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing ablation table: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    exp_ids = df['Exp_ID'].tolist()
    qwk_scores = df['QWK_Metric'].astype(float).tolist()
    macro_f1 = df['Macro_F1'].astype(float).tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    
    # ------------------ Subplot 1: QWK Metric (Panel A) ------------------
    bar_colors = ['#4a7bb0', '#4a7bb0', '#4a7bb0', '#4a7bb0', '#4a7bb0', '#2ca02c']
    bars = ax1.bar(exp_ids, qwk_scores, color=bar_colors, width=0.55, edgecolor='black', alpha=0.9)
    ax1.set_ylim(0.80, 0.96)
    ax1.set_xlabel('Factorial Experiment ID', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Quadratic Weighted Kappa (QWK)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Clinical Agreement Progression (QWK)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, qwk in zip(bars, qwk_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{qwk:.4f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    # ------------------ Subplot 2: Macro F1-Score (Panel B) ------------------
    ax2.plot(exp_ids, macro_f1, color='#1f77b4', marker='o', linewidth=2.5, markersize=8, label='Macro F1')
    ax2.bar(exp_ids, macro_f1, color='#a6c8e0', width=0.55, edgecolor='black', alpha=0.5)
    ax2.set_ylim(0.55, 0.82)
    ax2.set_xlabel('Factorial Experiment ID', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Macro F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Imbalanced Stage Sensitivity (Macro F1)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, f1 in enumerate(macro_f1):
        ax2.text(
            i, f1 + 0.010,
            f"{f1:.4f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0d3d66'
        )
        
    plt.suptitle('Ablation Study: Factorial Progression of Foundation Adaptation and Ordinal Cumulative Head', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/doe_ablation_architecture.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Ablation Engine] Successfully re-generated clean academic figure: '{out_path}'")

if __name__ == "__main__":
    generate_ablation_plot()