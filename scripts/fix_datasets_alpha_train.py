"""
Filter all-alpha protein dataset to get 200 diverse sequences.

Criteria:
- Chain A only
- Length between 100-400 residues
- Low sequence similarity (greedy selection for diversity)
- No overlap with existing test set
- Save to data/alpha_helical_train.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict


# Path to existing test set to exclude
EXCLUDE_CSV = "/share/NFS/u/kevin/ProteinFolding-1/data_old/target_test.csv"


def load_exclude_sequences(csv_path):
    """Load sequences to exclude from existing test set."""
    try:
        df = pd.read_csv(csv_path)
        sequences = set(df['sequence'].tolist())
        pdb_ids = set(df['pdb_id'].str.lower().tolist())
        print(f"  Loaded {len(sequences)} sequences to exclude from {csv_path}")
        return sequences, pdb_ids
    except FileNotFoundError:
        print(f"  Warning: Exclude file not found: {csv_path}")
        return set(), set()
    except Exception as e:
        print(f"  Warning: Could not load exclude file: {e}")
        return set(), set()


def load_from_fasta(fasta_path):
    """Load data from FASTA file."""
    records = []
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous record
                if current_header and current_seq:
                    record = parse_fasta_header(current_header, ''.join(current_seq))
                    if record:
                        records.append(record)
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget last record
        if current_header and current_seq:
            record = parse_fasta_header(current_header, ''.join(current_seq))
            if record:
                records.append(record)
    
    return records


def parse_fasta_header(header, sequence):
    """Parse FASTA header like: 1oaiA|len=59|helix=0.69|1.10.8.10"""
    parts = header.split('|')
    
    pdb_chain = parts[0] if parts else ''
    pdb_id = pdb_chain[:4].lower() if len(pdb_chain) >= 4 else pdb_chain.lower()
    chain = pdb_chain[4].upper() if len(pdb_chain) > 4 else 'A'
    
    length = len(sequence)
    helix_fraction = None
    cath_codes = ''
    
    for part in parts[1:]:
        if part.startswith('len='):
            try:
                length = int(part.split('=')[1])
            except:
                pass
        elif part.startswith('helix='):
            try:
                helix_fraction = float(part.split('=')[1])
            except:
                pass
        elif '.' in part:  # Likely CATH code
            cath_codes = part
    
    return {
        'pdb_id': pdb_id,
        'chain': chain,
        'length': length,
        'helix_fraction': helix_fraction,
        'cath_codes': cath_codes,
        'sequence': sequence
    }


def sequence_identity(seq1, seq2):
    """
    Compute pairwise sequence identity.
    Uses length of shorter sequence as denominator.
    """
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 1.0  # Treat empty as identical to be safe
    
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / min_len


def select_diverse_sequences(records, n_target=200, max_identity=0.25, 
                            exclude_sequences=None, exclude_pdb_ids=None):
    """
    Greedy selection of diverse sequences.
    Selects sequences that have < max_identity similarity to all already selected.
    Excludes sequences that match the exclude set.
    """
    if len(records) == 0:
        return []
    
    exclude_sequences = exclude_sequences or set()
    exclude_pdb_ids = exclude_pdb_ids or set()
    
    selected = []
    
    print(f"  Selecting up to {n_target} sequences with <{max_identity:.0%} identity...")
    print(f"  Excluding {len(exclude_sequences)} sequences and {len(exclude_pdb_ids)} PDB IDs")
    
    excluded_count = 0
    low_identity_rejected = 0
    
    for i, candidate in enumerate(records):
        if len(selected) >= n_target:
            break
        
        # Check if sequence or PDB ID is in exclude set
        if candidate['sequence'] in exclude_sequences:
            excluded_count += 1
            continue
        
        if candidate['pdb_id'].lower() in exclude_pdb_ids:
            excluded_count += 1
            continue
        
        # Also check similarity to excluded sequences (not just exact match)
        too_similar_to_excluded = False
        for excl_seq in exclude_sequences:
            if sequence_identity(candidate['sequence'], excl_seq) >= max_identity:
                too_similar_to_excluded = True
                break
        
        if too_similar_to_excluded:
            excluded_count += 1
            continue
        
        # Check identity against all selected sequences
        is_diverse = True
        for existing in selected:
            identity = sequence_identity(candidate['sequence'], existing['sequence'])
            if identity >= max_identity:
                is_diverse = False
                low_identity_rejected += 1
                break
        
        if is_diverse:
            selected.append(candidate)
            if len(selected) % 20 == 0:
                print(f"    Selected {len(selected)} sequences...")
    
    print(f"  Excluded {excluded_count} sequences due to overlap with test set")
    print(f"  Rejected {low_identity_rejected} sequences due to high similarity to selected")
    
    return selected


def main():
    # Configuration
    INPUT_FASTA = "cath_all_alpha.fasta"
    OUTPUT_CSV = "data/alpha_helical_train.csv"
    
    MIN_LENGTH = 90
    MAX_LENGTH = 400
    TARGET_CHAIN = 'A'
    N_SEQUENCES = 200
    MAX_IDENTITY = 0.25  # 25% sequence identity threshold
    
    # Allow command line override
    if len(sys.argv) > 1:
        INPUT_FASTA = sys.argv[1]
    
    print("=" * 60)
    print("FILTERING ALL-ALPHA DATASET")
    print("=" * 60)
    print(f"  Input: {INPUT_FASTA}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Length filter: {MIN_LENGTH}-{MAX_LENGTH}")
    print(f"  Chain: {TARGET_CHAIN}")
    print(f"  Target sequences: {N_SEQUENCES}")
    print(f"  Max sequence identity: {MAX_IDENTITY:.0%}")
    print()
    
    # Load FASTA
    print("[1] Loading FASTA...")
    records = load_from_fasta(INPUT_FASTA)
    print(f"  Loaded {len(records)} sequences")
    
    # Filter by chain
    print(f"\n[2] Filtering for chain {TARGET_CHAIN}...")
    records = [r for r in records if r['chain'] == TARGET_CHAIN]
    print(f"  After chain filter: {len(records)} sequences")
    
    # Filter by length
    print(f"\n[3] Filtering for length {MIN_LENGTH}-{MAX_LENGTH}...")
    records = [r for r in records if MIN_LENGTH <= r['length'] <= MAX_LENGTH]
    print(f"  After length filter: {len(records)} sequences")
    
    if len(records) == 0:
        print("\nError: No sequences passed filters!")
        sys.exit(1)
    
    # Load sequences to exclude
    print(f"\n[4] Loading sequences to exclude...")
    exclude_sequences, exclude_pdb_ids = load_exclude_sequences(EXCLUDE_CSV)
    
    # Sort by helix fraction (prefer higher helix content)
    records.sort(key=lambda x: x['helix_fraction'] or 0, reverse=True)
    
    # Select diverse sequences
    print(f"\n[5] Selecting diverse sequences...")
    selected = select_diverse_sequences(
        records, 
        n_target=N_SEQUENCES, 
        max_identity=MAX_IDENTITY,
        exclude_sequences=exclude_sequences,
        exclude_pdb_ids=exclude_pdb_ids
    )
    print(f"  Selected {len(selected)} diverse sequences")
    
    if len(selected) < N_SEQUENCES:
        print(f"\n  Warning: Only found {len(selected)} sequences meeting diversity criteria")
        print(f"  You may need to relax the identity threshold or add more input data")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Save to CSV
    print(f"\n[6] Saving to {OUTPUT_CSV}...")
    df = pd.DataFrame(selected)
    
    # Reorder columns
    cols = ['pdb_id', 'chain', 'length', 'helix_fraction', 'cath_codes', 'sequence']
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(df)} sequences")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total sequences: {len(df)}")
    print(f"  Length range: {df['length'].min()}-{df['length'].max()}")
    print(f"  Mean length: {df['length'].mean():.1f}")
    
    if df['helix_fraction'].notna().any():
        hf = df['helix_fraction'].dropna()
        print(f"  Helix fraction: {hf.min():.0%}-{hf.max():.0%} (mean: {hf.mean():.0%})")
    
    # Verify diversity
    print("\n  Verifying diversity...")
    sequences = df['sequence'].tolist()
    max_observed_identity = 0
    total_pairs = 0
    high_identity_pairs = 0
    
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            identity = sequence_identity(sequences[i], sequences[j])
            max_observed_identity = max(max_observed_identity, identity)
            total_pairs += 1
            if identity >= MAX_IDENTITY:
                high_identity_pairs += 1
    
    print(f"  Total pairs checked: {total_pairs}")
    print(f"  Max observed identity: {max_observed_identity:.1%}")
    print(f"  Pairs with identity >= {MAX_IDENTITY:.0%}: {high_identity_pairs}")
    
    print(f"\n  Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()