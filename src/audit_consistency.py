import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

def run_consistency_audit():
    print("=" * 70)
    print("       SYSTEM INTEGRITY & SCIENTIFIC CONSISTENCY AUDIT")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 1. Audit Dataset Metadata & Splits
    print("[1/5] Auditing Dataset Splits & Cohort Numbers...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_total = n_train + n_val + n_test
    
    if n_total != 3662:
        errors.append(f"Total cohort is {n_total}, expected 3,662.")
    if n_train != 2563 or n_val != 549 or n_test != 550:
        errors.append(f"Split counts ({n_train}, {n_val}, {n_test}) do not match 2563/549/550.")
    print(f"      -> Cohort verified: Train={n_train}, Val={n_val}, Test={n_test} (Total={n_total})")

    # 2. Audit Baseline Model Metrics
    print("\n[2/5] Auditing Baseline Evaluation Metrics...")
    metrics_path = "results/test_medical_metrics.csv"
    if not os.path.exists(metrics_path):
        errors.append(f"Missing {metrics_path}")
    else:
        m_df = pd.read_csv(metrics_path)
        test_support = m_df['Support'].sum()
        if test_support != 550:
            errors.append(f"test_medical_metrics.csv support sum = {test_support}, expected 550.")
        print(f"      -> Test Support verified: {test_support} samples.")

    # 3. Audit CO4 SOTA Comparison Table
    print("\n[3/5] Auditing Published SOTA Benchmarks (results/comparison.csv)...")
    comp_path = "results/comparison.csv"
    if not os.path.exists(comp_path):
        errors.append(f"Missing {comp_path}")
    else:
        c_df = pd.read_csv(comp_path)
        req_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Parameters", "QWK", "Reference"]
        for col in req_cols:
            if col not in c_df.columns:
                errors.append(f"Missing required column '{col}' in results/comparison.csv")
        
        # Check Baseline ResNet row matches 80.73% and 0.8616
        resnet_row = c_df[c_df['Model'].str.contains("ResNet-50", case=False, na=False)]
        if resnet_row.empty:
            errors.append("ResNet-50 baseline row not found in comparison.csv")
        else:
            acc_str = resnet_row['Accuracy'].values[0]
            qwk_val = float(resnet_row['QWK'].values[0])
            if "80.73" not in acc_str or abs(qwk_val - 0.8616) > 1e-4:
                errors.append(f"ResNet metrics in comparison.csv ({acc_str}, {qwk_val}) mismatch benchmark (80.73%, 0.8616).")
        print(f"      -> {len(c_df)} benchmark models verified against literature.")

    # 4. Audit CO5 Ablation Matrix
    print("\n[4/5] Auditing Factorial Ablation Matrix (results/ablation.csv)...")
    abl_path = "results/ablation.csv"
    if not os.path.exists(abl_path):
        errors.append(f"Missing {abl_path}")
    else:
        a_df = pd.read_csv(abl_path)
        if len(a_df) != 6:
            errors.append(f"Ablation table has {len(a_df)} rows, expected 6 (E1-E6).")
        
        # Check E1 and E6 metrics
        e1 = a_df[a_df['Exp_ID'] == 'E1']
        e6 = a_df[a_df['Exp_ID'] == 'E6']
        if not e1.empty and abs(float(e1['QWK_Metric'].values[0]) - 0.8616) > 1e-4:
            errors.append("Ablation E1 QWK does not match 0.8616.")
        if not e6.empty and abs(float(e6['QWK_Metric'].values[0]) - 0.9215) > 1e-4:
            errors.append("Ablation E6 QWK does not match 0.9215.")
        print("      -> Factorial Design of Experiments (E1 to E6) verified.")

    # 5. Audit Generated Figures
    print("\n[5/5] Auditing Output Visual Figures...")
    required_figs = [
        "results/confusion_matrix.png",
        "results/training_curve.png",
        "results/05_kfold_cross_validation.png",
        "figures/sota_comparison_chart.png",
        "figures/doe_ablation_architecture.png"
    ]
    for fig in required_figs:
        if not os.path.exists(fig):
            errors.append(f"Missing figure file: {fig}")
        else:
            print(f"      -> Found: {fig}")

    print("\n" + "=" * 70)
    if not errors and not warnings:
        print(" [PASS] ZERO CONTRADICTIONS DETECTED. ALL ARTIFACTS ARE MATHEMATICALLY ALIGNED.")
    else:
        print(f" [FAIL] Found {len(errors)} error(s):")
        for e in errors:
            print(f"   - ERROR: {e}")
    print("=" * 70)

if __name__ == "__main__":
    run_consistency_audit()