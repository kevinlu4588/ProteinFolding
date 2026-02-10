"""
Deduplicate patching dataset to unique target + loop combinations.

Input: patching_dataset.csv with one row per (target, loop, donor) combination
Output: Deduplicated dataset with unique (target, loop) combinations

Keeps target chain info and loop/helix locations, removes donor-specific columns.
"""

import os
import sys
import pandas as pd
import numpy as np


# Input/Output paths
INPUT_CSV = "/share/NFS/u/kevin/ProteinFolding-1/data_old/patching_dataset.csv"
OUTPUT_CSV = "data/target_loops_dataset.csv"


def main():
    print("=" * 60)
    print("DEDUPLICATING PATCHING DATASET")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading patching dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Identify target-related columns (keep these)
    target_columns = [
        'target_name',
        'target_sequence', 
        'target_length',
        'loop_idx',
        'loop_start',
        'loop_end', 
        'loop_length',
        'loop_sequence',
        'target_patch_start',
        'target_patch_end',
        'patch_length',
    ]
    
    # Check which columns exist
    existing_target_cols = [col for col in target_columns if col in df.columns]
    missing_cols = [col for col in target_columns if col not in df.columns]
    
    print(f"\n[2] Target columns found: {existing_target_cols}")
    if missing_cols:
        print(f"  Missing columns: {missing_cols}")
    
    # Select only target columns
    df_targets = df[existing_target_cols].copy()
    
    # Deduplicate by target_name + loop_idx (unique target/loop combinations)
    print(f"\n[3] Deduplicating...")
    print(f"  Before dedup: {len(df_targets)} rows")
    
    # Define what makes a unique entry
    dedup_cols = ['target_name', 'loop_idx']
    dedup_cols = [col for col in dedup_cols if col in df_targets.columns]
    
    if not dedup_cols:
        # Fallback: deduplicate by target_name + loop_start + loop_end
        dedup_cols = ['target_name', 'loop_start', 'loop_end']
        dedup_cols = [col for col in dedup_cols if col in df_targets.columns]
    
    print(f"  Deduplicating by: {dedup_cols}")
    
    df_deduped = df_targets.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"  After dedup: {len(df_deduped)} rows")
    
    # Sort by target_name and loop_idx for cleaner output
    sort_cols = [col for col in ['target_name', 'loop_idx', 'loop_start'] if col in df_deduped.columns]
    if sort_cols:
        df_deduped = df_deduped.sort_values(sort_cols).reset_index(drop=True)
    
    # Summary statistics
    print(f"\n[4] Summary:")
    if 'target_name' in df_deduped.columns:
        n_targets = df_deduped['target_name'].nunique()
        print(f"  Unique targets: {n_targets}")
    
    if 'loop_idx' in df_deduped.columns:
        loops_per_target = df_deduped.groupby('target_name')['loop_idx'].count()
        print(f"  Loops per target: min={loops_per_target.min()}, max={loops_per_target.max()}, mean={loops_per_target.mean():.1f}")
    
    if 'target_length' in df_deduped.columns:
        print(f"  Target length range: {df_deduped['target_length'].min()}-{df_deduped['target_length'].max()}")
    
    if 'loop_length' in df_deduped.columns:
        print(f"  Loop length range: {df_deduped['loop_length'].min()}-{df_deduped['loop_length'].max()}")
    
    if 'patch_length' in df_deduped.columns:
        print(f"  Patch length range: {df_deduped['patch_length'].min()}-{df_deduped['patch_length'].max()}")
    
    # Save
    print(f"\n[5] Saving to {OUTPUT_CSV}...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_deduped.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(df_deduped)} rows")
    
    # Show sample
    print(f"\n[6] Sample rows:")
    print(df_deduped.head(10).to_string())
    
    print(f"\n  Done! Output saved to: {OUTPUT_CSV}")
    
    return df_deduped


if __name__ == "__main__":
    main()