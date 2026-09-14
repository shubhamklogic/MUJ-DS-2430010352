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
    
    models = df['Model'].tolist()
    qwk_scores = df['QWK'].astype(float).tolist()
    
    # Visual color scheme:
    # Muted grey for literature baselines, Blue for our CO3 baseline, Green for proposed innovation
    colors = []
    for m in models:
        if "Baseline" in m:
            colors.append('#1f77b4')  # Blue (CO3)
        elif "Proposed" in m:
            colors.append('#2ca02c')  # Green (CO5)
        else:
            colors.append('#7f7f7f')  # Muted grey (Published literature)
            
    plt.figure(figsize=(10, 5.2))
    bars = plt.barh(models, qwk_scores, color=colors, edgecolor='black', height=0.6)
    
    plt.xlim(0.65, 1.0)
    plt.xlabel('Quadratic Weighted Kappa (QWK) on APTOS 2019', fontsize=11, fontweight='bold')
    plt.title('State-of-the-Art (SOTA) Comparative Benchmark (APTOS 2019)', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add numerical labels beside bars
    for bar, score in zip(bars, qwk_scores):
        plt.text(score + 0.006, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}",
                 va='center', ha='left', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/sota_comparison_chart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[CO4 Engine] Successfully generated benchmark chart: '{out_path}'")

if __name__ == "__main__":
    generate_sota_comparison_plot()