import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

def generate_sota_comparison():
    csv_path = Path("results/comparison.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
        
    df = pd.read_csv(csv_path)
    os.makedirs("figures", exist_ok=True)
    
    models = df['Model Architecture'].tolist()
    qwk_scores = [float(x) for x in df['QWK']]
    
    # Highlight our baseline model and proposed innovation
    colors = ['#aec7e8', '#aec7e8', '#1f77b4', '#aec7e8', '#aec7e8', '#c7c7c7', '#2ca02c']
    
    plt.figure(figsize=(11, 5.5))
    bars = plt.barh(models, qwk_scores, color=colors, edgecolor='black', linewidth=0.8)
    
    plt.xlabel('Quadratic Weighted Kappa (QWK) Score', fontsize=11, fontweight='bold')
    plt.title('CO4 Deliverable: Benchmark Comparison on APTOS 2019 Blindness Detection', fontsize=12, fontweight='bold', pad=12)
    plt.xlim(0.60, 0.98)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f'{w:.4f}', va='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    output_path = "figures/sota_comparison_chart.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[CO4] Saved SOTA comparison chart to: {output_path}")

def generate_ablation_figure():
    csv_path = Path("results/ablation.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
        
    df = pd.read_csv(csv_path)
    os.makedirs("figures", exist_ok=True)
    
    exp_ids = df['Exp_ID'].tolist()
    qwk = [float(x) for x in df['QWK_Metric']]
    f1 = [float(x) for x in df['Macro_F1']]
    
    x = range(len(exp_ids))
    width = 0.35
    
    plt.figure(figsize=(10, 5))
    plt.bar([p - width/2 for p in x], qwk, width=width, label='QWK Metric', color='#1f77b4', edgecolor='black')
    plt.bar([p + width/2 for p in x], f1, width=width, label='Macro F1-Score', color='#ff7f0e', edgecolor='black')
    
    plt.xlabel('Design of Experiments (DoE) ID', fontsize=11, fontweight='bold')
    plt.ylabel('Score Value', fontsize=11, fontweight='bold')
    plt.title('CO5 Deliverable: Factorial Design of Experiments (DoE) & Ablation Study', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(x, exp_ids)
    plt.ylim(0.50, 1.00)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate the final proposed innovation
    plt.annotate('Proposed Innovation (E6)\nQWK: 0.9215 | F1: 0.7480',
                 xy=(5, 0.9215), xytext=(3.2, 0.94),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", ec="#28a745"))
    
    plt.tight_layout()
    output_path = "figures/doe_ablation_architecture.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[CO5] Saved DoE ablation chart to: {output_path}")

if __name__ == "__main__":
    print("Generating Module C Deliverables (CO4 SOTA Benchmark & CO5 DoE Ablation)...")
    generate_sota_comparison()
    generate_ablation_figure()
    print("Module C figures generated successfully!")