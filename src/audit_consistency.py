import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

def audit_repository_integrity():
    print("=" * 70)
    print("      SYSTEM INTEGRITY & SCIENTIFIC CONSISTENCY AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------
    # [1/5] Audit Dataset Splits & Cohort Preservation
    # -------------------------------------------------------------
    print("\n[1/5] Auditing Dataset Splits & Cohort Leakage...")
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split CSV: '{p}'")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    total_cohort = n_train + n_val + n_test

    assert n_train == 2563, f"Expected 2563 train scans, got {n_train}"
    assert n_val == 549, f"Expected 549 val scans, got {n_val}"
    assert n_test == 550, f"Expected 550 test scans, got {n_test}"
    assert total_cohort == 3662, f"Expected 3662 total scans, got {total_cohort}"

    # Verify zero leakage across splits
    all_ids = set(train_df['id_code']).union(set(val_df['id_code'])).union(set(test_df['id_code']))
    assert len(all_ids) == 3662, "Data leakage detected: Duplicate scan IDs found across splits!"
    print(f"  [PASS] Cohort preserved: Train={n_train}, Val={n_val}, Test={n_test} (Total={total_cohort}) with Zero Leakage.")

    # -------------------------------------------------------------
    # [2/5] Audit Test Arrays & Empirical Metrics
    # -------------------------------------------------------------
    print("\n[2/5] Auditing Test Vectors & Empirical Metrics...")
    true_arr_path = "results/y_true_test.npy"
    pred_arr_path = "results/y_pred_test.npy"

    assert os.path.exists(true_arr_path), f"Missing '{true_arr_path}'"
    assert os.path.exists(pred_arr_path), f"Missing '{pred_arr_path}'"

    y_true = np.load(true_arr_path)
    y_pred = np.load(pred_arr_path)

    assert len(y_true) == 550, f"Expected 550 true labels, got {len(y_true)}"
    assert len(y_pred) == 550, f"Expected 550 predictions, got {len(y_pred)}"

    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    assert np.isclose(acc, 0.7854545, atol=1e-3), f"Accuracy mismatch: expected ~78.55%, got {acc*100:.2f}%"
    assert np.isclose(qwk, 0.8998, atol=1e-3), f"QWK mismatch: expected ~0.8998, got {qwk:.4f}"
    print(f"  [PASS] Verified Hold-Out Evaluation: Accuracy={acc*100:.2f}%, QWK={qwk:.4f} on N=550 test scans.")

    # -------------------------------------------------------------
    # [3/5] Audit SOTA Comparative Benchmark Table
    # -------------------------------------------------------------
    print("\n[3/5] Auditing SOTA Comparative Benchmark Table...")
    comp_path = "results/comparison.csv"
    assert os.path.exists(comp_path), f"Missing '{comp_path}'"
    comp_df = pd.read_csv(comp_path)

    model_col = 'Model Architecture' if 'Model Architecture' in comp_df.columns else 'Model'
    qwk_col = 'QWK' if 'QWK' in comp_df.columns else 'QWK_Metric'

    proposed_row = comp_df[comp_df[model_col].str.contains("Proposed", case=False, na=False)]
    assert not proposed_row.empty, "Proposed model row missing in comparison.csv!"
    prop_qwk = float(str(proposed_row[qwk_col].values[0]).replace('%', '').strip())
    assert np.isclose(prop_qwk, 0.8998, atol=1e-3), f"comparison.csv Proposed QWK is {prop_qwk}, expected 0.8998"
    print(f"  [PASS] comparison.csv synchronized: Proposed PEFT QWK = {prop_qwk:.4f}.")

    # -------------------------------------------------------------
    # [4/5] Audit Factorial DoE Ablation Matrix
    # -------------------------------------------------------------
    print("\n[4/5] Auditing Factorial DoE Ablation Matrix...")
    ablation_path = "results/ablation.csv"
    assert os.path.exists(ablation_path), f"Missing '{ablation_path}'"
    ablation_df = pd.read_csv(ablation_path)

    e6_row = ablation_df[ablation_df['Exp_ID'] == 'E6']
    assert not e6_row.empty, "Experiment E6 missing in ablation.csv!"
    e6_qwk = float(e6_row['QWK_Metric'].values[0])
    e6_f1 = float(e6_row['Macro_F1'].values[0])

    assert np.isclose(e6_qwk, 0.8998, atol=1e-3), f"ablation.csv E6 QWK is {e6_qwk}, expected 0.8998"
    assert np.isclose(e6_f1, 0.5911, atol=1e-3), f"ablation.csv E6 Macro F1 is {e6_f1}, expected 0.5911"
    print(f"  [PASS] ablation.csv synchronized: E6 QWK = {e6_qwk:.4f}, Macro F1 = {e6_f1:.4f}.")

    # -------------------------------------------------------------
    # [5/5] Audit Deliverables & Figure Files
    # -------------------------------------------------------------
    print("\n[5/5] Auditing Physical Deliverables & Visual Assets...")
    required_assets = [
        "models/best_peft_model.pth",
        "results/true_test_metrics.txt",
        "results/per_stage_clinical_sensitivity.csv",
        "results/statistical_significance.csv",
        "results/proposed_confusion_matrix.png",
        "figures/sota_comparison_chart.png",
        "figures/doe_ablation_architecture.png",
        "figures/peft_training_curves.png",
        "figures/xai_gradcam_stages.png",
        "src/train_peft_full.py"
    ]

    for asset in required_assets:
        assert os.path.exists(asset), f"Missing required deliverable: '{asset}'"
        size = os.path.getsize(asset)
        assert size > 0, f"Deliverable '{asset}' is empty (0 bytes)!"
        print(f"  [✓] {asset} verified ({size:,} bytes)")

    print("\n" + "=" * 70)
    print(" [PASS] ZERO CONTRADICTIONS DETECTED. ALL ARTIFACTS ARE MATHEMATICALLY ALIGNED.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    audit_repository_integrity()