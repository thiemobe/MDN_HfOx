"""
Split original_data.csv into train/val and test sets.

Split logic:
  - Rows 0-599: training/validation pool (train_val_samples.csv)
  - Rows 600+: test set (test_samples.csv)

This matches the split used in baseline_comparison.py (N_POOL = 600).
"""

import os
import pandas as pd

DATA_FILE = 'data/original_data.csv'
TRAIN_VAL_OUTPUT = 'data/train_val_samples.csv'
TEST_OUTPUT = 'data/test_samples.csv'
N_POOL = 600  # rows 0..599: train + validation; rows 600..: test

def split_data():
    """Split original_data.csv into train/val and test sets."""
    print(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
    
    print(f"Total rows in original_data.csv: {len(df)}")
    
    # Split into train/val pool (rows 0-599) and test set (rows 600+)
    df_train_val = df.iloc[:N_POOL]
    df_test = df.iloc[N_POOL:]
    
    print(f"Train/Val pool: {len(df_train_val)} rows (rows 0-{N_POOL-1})")
    print(f"Test set: {len(df_test)} rows (rows {N_POOL}+)")
    
    # Save split data
    os.makedirs(os.path.dirname(TRAIN_VAL_OUTPUT) or '.', exist_ok=True)
    
    df_train_val.to_csv(TRAIN_VAL_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"Saved: {TRAIN_VAL_OUTPUT}")
    
    df_test.to_csv(TEST_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"Saved: {TEST_OUTPUT}")
    
    print("\nData split complete!")
    print(f"  {TRAIN_VAL_OUTPUT}: {len(df_train_val)} rows (for training/validation)")
    print(f"  {TEST_OUTPUT}: {len(df_test)} rows (for testing)")


if __name__ == "__main__":
    split_data()
