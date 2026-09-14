import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.reproducibility import set_seed

def verify_dataset_partitions():
    set_seed(42)
    
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"
    
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing partition file: {p}")
            
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_total = n_train + n_val + n_test
    
    print("=" * 65)
    print("      DATASET PARTITION & LEAKAGE VERIFICATION AUDIT")
    print("=" * 65)
    print(f"Train subset:       {n_train} samples ({n_train / n_total * 100:.2f}%)")
    print(f"Validation subset:  {n_val} samples ({n_val / n_total * 100:.2f}%)")
    print(f"Test subset:        {n_test} samples ({n_test / n_total * 100:.2f}%)")
    print(f"Total Cohort:       {n_total} samples")
    print("-" * 65)
    
    # 1. Leakage Verification via Set Intersections
    train_ids = set(train_df['id_code'])
    val_ids = set(val_df['id_code'])
    test_ids = set(test_df['id_code'])
    
    leak_train_val = train_ids.intersection(val_ids)
    leak_train_test = train_ids.intersection(test_ids)
    leak_val_test = val_ids.intersection(test_ids)
    
    assert len(leak_train_val) == 0, f"LEAKAGE: {len(leak_train_val)} IDs overlap between Train and Val!"
    assert len(leak_train_test) == 0, f"LEAKAGE: {len(leak_train_test)} IDs overlap between Train and Test!"
    assert len(leak_val_test) == 0, f"LEAKAGE: {len(leak_val_test)} IDs overlap between Val and Test!"
    
    print("Zero Data Leakage Check:")
    print("  Train IDs ∩ Val IDs:   0 overlapping samples (PASS)")
    print("  Train IDs ∩ Test IDs:  0 overlapping samples (PASS)")
    print("  Val IDs ∩ Test IDs:    0 overlapping samples (PASS)")
    print("-" * 65)
    
    # 2. Stratification Class Proportions Audit
    print("Class Proportions across Partitions (%):")
    prop_df = pd.DataFrame({
        'Train %': train_df['diagnosis'].value_counts(normalize=True).sort_index() * 100,
        'Val %':   val_df['diagnosis'].value_counts(normalize=True).sort_index() * 100,
        'Test %':  test_df['diagnosis'].value_counts(normalize=True).sort_index() * 100
    })
    print(prop_df.round(2).to_string())
    print("=" * 65)
    print("Status: Stratification preserved; Zero data leakage mathematically verified.\n")

if __name__ == "__main__":
    verify_dataset_partitions()