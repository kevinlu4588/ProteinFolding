import pandas as pd

df = pd.read_parquet("/share/NFS/u/kevin/ProteinFolding-1/data_old/block_patching_results.parquet")

print(df)
print(df.columns)
import pandas as pd
import pandas as pd

# Define the filter conditions
sequence_block0_mask = (
    (df['patch_mode'] == 'sequence') & 
    (df['patch_mask_mode'] == 'intra') & 
    (df['hairpin_found'] == True) & 
    (df['block_idx'] == 0)
)

pairwise_block30_mask = (
    (df['patch_mode'] == 'pairwise') & 
    (df['patch_mask_mode'] == 'intra') & 
    (df['hairpin_found'] == True) & 
    (df['block_idx'] == 30)
)

# Print population sizes
print(f"Sequence block 0 population: {sequence_block0_mask.sum()}")
print(f"Pairwise block 30 population: {pairwise_block30_mask.sum()}")

# Sample from each population
sequence_block0 = df[sequence_block0_mask].sample(n=400, random_state=42)
pairwise_block30 = df[pairwise_block30_mask].sample(n=200, random_state=42)

# Combine into single dataset
single_block_patching_successes = pd.concat([
    sequence_block0,
    pairwise_block30
]).reset_index(drop=True)

# Verify
print(f"\nTotal rows: {len(single_block_patching_successes)}")
print(f"Sequence block 0 rows: {len(sequence_block0)}")
print(f"Pairwise block 30 rows: {len(pairwise_block30)}")

single_block_patching_successes.to_csv("data/single_block_patching_successes.csv")