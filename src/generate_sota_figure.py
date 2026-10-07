import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

def generate_sota_comparison_plot():
    csv_path = "results/comparison.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing comparison table: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    model_col = 'Model Architecture' if 'Model Architecture' in df.columns else 'Model'
    param_col = 'Trainable Params' if 'Trainable Params' in df.columns else ('Parameters' if 'Parameters' in df.columns else None)
    qwk_col = 'QWK' if 'QWK' in df.columns else 'QWK_Metric'
    
    df['QWK_float'] = df[qwk_col].astype(str).str.replace('%', '').str.strip().astype(float)
    df = df.sort_values(by='QWK_float', ascending=True).reset_index(drop=True)
    
    models = df[model_col].tolist()
    qwk_scores = df['QWK_float'].tolist()
    params = df[param_col].tolist() if param_col else [''] * len(models)
    
    colors = []
    for m in models:
        if "Baseline" in str(m):
            colors.append('#1f77b4')  # Blue for baseline
        elif "Proposed" in str(m) or "PEFT" in str(m):
            colors.append('#2ca02c')  # Green for proposed
        else:
            colors.append('#7f7f7f')  # Grey for literature
            
    plt.figure(figsize=(11, 5.5))
    bars = plt.barh(models, qwk_scores, color=colors, edgecolor='black', height=0.6)
    
    plt.xlim(0.80, 0.98)
    plt.xlabel('Quadratic Weighted Kappa (QWK) on APTOS 2019', fontsize=11, fontweight='bold')
    plt.title('State-of-the-Art (SOTA) Comparative Benchmark on APTOS 2019', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar, score, p in zip(bars, qwk_scores, params):
        label = f"{score:.4f}  ({p})" if str(p).strip() else f"{score:.4f}"
        plt.text(score + 0.002, bar.get_y() + bar.get_height() / 2,
                 label,
                 va='center', ha='left', fontsize=9.5, fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/sota_comparison_chart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Engine] Successfully updated benchmark chart: '{out_path}'")

if __name__ == "__main__":
    generate_sota_comparison_plot()