"""
Single-Block Activation Patching
================================

Identifies which ESMFold trunk blocks encode hairpin structural information by
patching activations one block at a time. For each of the 48 trunk blocks,
transplants the (s, z) representations from a hairpin-containing donor sequence
into a helical acceptor sequence and measures whether a hairpin forms.

This experiment reveals the temporal dynamics of structure formation: early blocks
(0-10) show the strongest patching effects, indicating that hairpin geometry is
established early in the folding trunk's forward pass.

Usage:
    python block_patching.py --parquet patching_dataset.parquet --output_dir results/
"""

import argparse
import os
import types
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import types

# Add project root (parent of src/) to path so `src.*` imports work without PYTHONPATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput,
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
)
from transformers.utils import ContextManagers

from src.utils.trunk_utils import detect_hairpins
from src.utils.representation_utils import CollectedRepresentations, TrunkHooks


# ============================================================================
# PART 1: COLLECTION CONVENIENCE FUNCTION
# ============================================================================

def run_and_collect(
    model,
    tokenizer,
    device: str,
    sequence: str,
    num_recycles: int = 0,
) -> Tuple[EsmForProteinFoldingOutput, CollectedRepresentations]:
    """
    Run model and collect trunk representations using hooks.
    """
    collector = CollectedRepresentations()
    
    trunk_hooks = TrunkHooks(model.trunk, collector)
    trunk_hooks.register(collect_s=True, collect_z=True)
    
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=num_recycles)
    finally:
        trunk_hooks.remove()
    
    return outputs, collector


# ============================================================================
# PART 2: MASKS
# ============================================================================

def create_pairwise_mask(
    donor_start: int,
    donor_end: int,
    donor_len: int,
    target_start: int,
    target_end: int,
    target_len: int,
    mode: str,
) -> torch.Tensor:
    """Create pairwise patch mask."""
    mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    
    if mode == "intra":
        mask[target_start:target_end, target_start:target_end] = True
        
    elif mode in ("touch", "hole"):
        left_extent = min(donor_start, target_start)
        right_extent = min(donor_len - donor_end, target_len - target_end)
        
        transport_start = target_start - left_extent
        transport_end = target_end + right_extent
        
        mask[target_start:target_end, transport_start:transport_end] = True
        mask[transport_start:transport_end, target_start:target_end] = True
        
        if mode == "hole":
            mask[target_start:target_end, target_start:target_end] = False
    
    return mask


# ============================================================================
# PART 3: INTERVENTION - SINGLE BLOCK TRUNK PATCHING
# ============================================================================

def make_trunk_single_block_patch_forward(
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    target_block: int,
):
    """
    Create a trunk forward that patches a SINGLE block with donor representations.
    
    Args:
        donor_s_blocks: Dict mapping block_idx -> [B, patch_len, D_s] sequence repr
        donor_z_blocks: Dict mapping block_idx -> [B, L, L, D_z] FULL pairwise repr
        target_start, target_end: Where to patch in target
        donor_start: Where donor region starts (for pairwise coordinate mapping)
        pairwise_mask: Boolean mask for which (i,j) pairs to patch
        patch_mode: 'sequence', 'pairwise', or 'both'
        target_block: Which block index to patch (only this block gets patched)
    
    Returns:
        A forward function to be bound to model.trunk
    """
    
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            no_recycles += 1

        def apply_patch(block_idx, s, z):
            """Apply donor patch at this block (only if it's the target block)."""
            if block_idx != target_block:
                return s, z
            
            # Patch sequence
            if patch_mode in ('both', 'sequence') and block_idx in donor_s_blocks:
                donor_s = donor_s_blocks[block_idx].to(s.device, dtype=s.dtype)
                s[:, target_start:target_end, :] = donor_s
            
            # Patch pairwise
            if patch_mode in ('both', 'pairwise') and block_idx in donor_z_blocks:
                donor_z = donor_z_blocks[block_idx].to(z.device, dtype=z.dtype)
                mask_dev = pairwise_mask.to(z.device)
                
                indices = torch.where(mask_dev)
                for i in range(len(indices[0])):
                    ti, tj = indices[0][i].item(), indices[1][i].item()
                    di = ti - target_start + donor_start
                    dj = tj - target_start + donor_start
                    if 0 <= di < donor_z.shape[1] and 0 <= dj < donor_z.shape[2]:
                        z[:, ti, tj, :] = donor_z[:, di, dj, :]
            
            return s, z

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            for block_idx, block in enumerate(self.blocks):
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
                s, z = apply_patch(block_idx, s, z)
            return s, z

        s_s, s_z = s_s_0, s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        for recycle_idx in range(no_recycles):
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa, mask.float(),
                )

                recycle_s, recycle_z = s_s, s_z
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375, 21.375, self.recycle_bins,
                )

        structure["s_s"] = s_s
        structure["s_z"] = s_z
        return structure
    
    return forward


@contextmanager
def patch_trunk_single_block(
    model,
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    target_block: int,
):
    """
    Context manager for trunk single-block patching.
    
    Usage:
        with patch_trunk_single_block(model, donor_s, donor_z, ..., target_block=5):
            outputs = model(**inputs)
    """
    original = model.trunk.forward
    
    patched_forward = make_trunk_single_block_patch_forward(
        donor_s_blocks, donor_z_blocks,
        target_start, target_end, donor_start,
        pairwise_mask, patch_mode, target_block,
    )
    model.trunk.forward = types.MethodType(patched_forward, model.trunk)
    
    try:
        yield
    finally:
        model.trunk.forward = original


# ============================================================================
# PART 4: ANALYSIS UTILITIES
# ============================================================================

def evaluate_hairpin(
    outputs: EsmForProteinFoldingOutput,
    model,
    target_start: int,
    target_end: int,
) -> Dict[str, Any]:
    """Evaluate hairpin formation and structure quality."""
    hairpin_found, _ = detect_hairpins(outputs, model)
    
    plddt = outputs.plddt[0].cpu().numpy()
    ptm = outputs.ptm.item() if outputs.ptm is not None else None
    
    return {
        'hairpin_found': hairpin_found,
        'mean_plddt': float(plddt.mean()),
        'patch_region_plddt': float(plddt[target_start:target_end].mean()),
        'ptm': ptm,
    }


# ============================================================================
# PART 5: PLOTTING
# ============================================================================

def generate_summary_plots(results_df: pd.DataFrame, output_dir: str):
    """Generate summary plots from results."""
    try:
        from block_plotting import (
            plot_success_rates,
            plot_success_rates_grouped,
            plot_plddt_comparison,
            plot_summary_table,
        )
        
        output_path = Path(output_dir)
        
        plot_success_rates(results_df, output_path)
        plot_success_rates_grouped(results_df, output_path)
        plot_plddt_comparison(results_df, output_path)
        plot_summary_table(results_df, output_path)
        
        print(f"  Plots saved to {output_dir}")
        
    except ImportError as e:
        print(f"  Warning: Could not import plotting module: {e}")
        print("  Generating basic plots instead...")
        generate_basic_plots(results_df, output_dir)


def generate_basic_plots(results_df: pd.DataFrame, output_dir: str):
    """Generate basic plots if plotting module not available."""
    import matplotlib.pyplot as plt
    
    if len(results_df) == 0:
        return
    
    # Plot: Hairpin detection rate by block
    fig, ax = plt.subplots(figsize=(12, 5))
    
    block_success = results_df.groupby('block_idx')['hairpin_found'].mean()
    ax.bar(block_success.index, block_success.values, color='steelblue')
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Hairpin Detection Rate')
    ax.set_title('Hairpin Detection Rate by Block')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hairpin_by_block.png"), dpi=150)
    plt.close()
    
    # Plot: Hairpin detection by block × patch_mode
    fig, ax = plt.subplots(figsize=(14, 5))
    
    pivot = results_df.pivot_table(
        values='hairpin_found', 
        index='block_idx', 
        columns='patch_mode', 
        aggfunc='mean'
    )
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Hairpin Detection Rate')
    ax.set_title('Hairpin Detection by Block × Patch Mode')
    ax.set_ylim(0, 1)
    ax.legend(title='Patch Mode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hairpin_by_block_mode.png"), dpi=150)
    plt.close()
    
    # Plot: Heatmap of success by block × mask_mode
    fig, axes = plt.subplots(1, len(results_df['patch_mode'].unique()), figsize=(16, 5))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    
    for ax, patch_mode in zip(axes, results_df['patch_mode'].unique()):
        subset = results_df[results_df['patch_mode'] == patch_mode]
        pivot = subset.pivot_table(
            values='hairpin_found',
            index='patch_mask_mode',
            columns='block_idx',
            aggfunc='mean'
        )
        
        import seaborn as sns
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, 
                    annot=True, fmt='.2f', cbar_kws={'label': 'Success Rate'})
        ax.set_title(f'Patch Mode: {patch_mode}')
        ax.set_xlabel('Block Index')
        ax.set_ylabel('Mask Mode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_block_mask.png"), dpi=150)
    plt.close()
    
    print(f"  Basic plots saved to {output_dir}")


# ============================================================================
# PART 6: MAIN EXPERIMENT
# ============================================================================

def run_experiment_on_dataset(
    parquet_path: str,
    output_dir: str,
    patch_modes: List[str] = ["sequence", "pairwise", "both"],
    patch_mask_modes: List[str] = ["intra", "touch"],
    save_pdbs: bool = False,
    n_cases: Optional[int] = None,
    device: Optional[str] = None,
    flush_every: int = 20,
) -> pd.DataFrame:
    """
    Run single-block patching experiments on the patching dataset.
    
    Tests each block individually to identify which blocks matter most.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_pdbs:
        pdb_dir = os.path.join(output_dir, "pdbs")
        os.makedirs(pdb_dir, exist_ok=True)
    else:
        pdb_dir = None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("Model loaded")
    
    # Load dataset
    print(f"Loading patching dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df[(df["patch_module"] == "trunk") & (df["hairpin_found"] == True) & (df['patch_mask_mode'] == 'intra')]
    if n_cases is not None:
        df = df.head(n_cases)
    print(f"Running {len(df)} cases")
    
    # Results storage
    all_results = []
    results_path = os.path.join(output_dir, "single_block_patching_results.parquet")
    
    # Columns to preserve from input
    preserve_cols = [
        "target_name", "target_sequence", "target_length",
        "loop_idx", "loop_start", "loop_end", "loop_length", "loop_sequence",
        "target_patch_start", "target_patch_end", "patch_length",
        "donor_pdb", "donor_sequence", "donor_length",
        "donor_hairpin_start", "donor_hairpin_end", "donor_hairpin_length",
        "donor_hairpin_sequence",
        "donor_strand1_length", "donor_strand2_length",
        "donor_loop_sequence", "donor_loop_length",
        "donor_handedness_magnitude", "loop_similarity",
    ]
    
    for case_idx, row in tqdm(df.iterrows(), total=len(df), desc="Cases"):
        target_seq = row["target_sequence"]
        donor_seq = row["donor_sequence"]
        
        target_start = int(row["target_patch_start"])
        target_end = int(row["target_patch_end"])
        donor_start = int(row["donor_hairpin_start"])
        donor_end = int(row["donor_hairpin_end"])
        
        print(f"\nCase {case_idx}: loop {row.get('loop_idx', '?')} <- {row.get('donor_pdb', '?')}")
        print(f"  Target patch: [{target_start}:{target_end})")
        print(f"  Donor hairpin: [{donor_start}:{donor_end})")
        
        # Preserve metadata
        case_meta = {col: row[col] for col in preserve_cols if col in row.index}
        case_meta["case_idx"] = case_idx
        case_meta["target_start"] = target_start
        case_meta["target_end"] = target_end
        case_meta["donor_start"] = donor_start
        case_meta["donor_end"] = donor_end
        
        # =====================================================================
        # COLLECT DONOR REPRESENTATIONS
        # =====================================================================
        print("  Collecting donor representations...")
        donor_outputs, donor_collected = run_and_collect(
            model, tokenizer, device, donor_seq
        )
        
        # Extract hairpin region for sequence representations
        donor_s_region = {
            k: v[:, donor_start:donor_end, :] 
            for k, v in donor_collected.s_blocks.items()
        }
        # Keep full z_blocks for pairwise patching
        donor_z_full = donor_collected.z_blocks
        
        num_blocks = len(donor_s_region)
        print(f"    Trunk blocks: {num_blocks}")
        
        del donor_outputs
        torch.cuda.empty_cache()
        
        # Create pairwise masks
        pairwise_masks = {}
        for mask_mode in patch_mask_modes:
            pairwise_masks[mask_mode] = create_pairwise_mask(
                donor_start=donor_start,
                donor_end=donor_end,
                donor_len=len(donor_seq),
                target_start=target_start,
                target_end=target_end,
                target_len=len(target_seq),
                mode=mask_mode,
            )
        
        case_results = []
        
        # =====================================================================
        # SINGLE-BLOCK PATCHING EXPERIMENTS
        # =====================================================================
        for patch_mode in patch_modes:
            for mask_mode in patch_mask_modes:
                # Skip redundant mask modes for sequence-only patching
                if patch_mode == "sequence" and mask_mode != patch_mask_modes[0]:
                    continue
                
                for block_idx in range(num_blocks):
                    with patch_trunk_single_block(
                        model, donor_s_region, donor_z_full,
                        target_start, target_end, donor_start,
                        pairwise_masks[mask_mode], patch_mode, block_idx
                    ):
                        with torch.no_grad():
                            inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                            outputs = model(**inputs, num_recycles=0)
                    
                    eval_result = evaluate_hairpin(
                        outputs, model, target_start, target_end
                    )
                    
                    result = {
                        "block_idx": block_idx,
                        "patch_mode": patch_mode,
                        "patch_mask_mode": mask_mode,
                        **eval_result,
                    }
                    result.update(case_meta)
                    
                    if save_pdbs and pdb_dir:
                        pdb_str = model.output_to_pdb(outputs)[0]
                        pdb_filename = f"case{case_idx}_block{block_idx}_{patch_mode}_{mask_mode}.pdb"
                        with open(os.path.join(pdb_dir, pdb_filename), 'w') as f:
                            f.write(pdb_str)
                    
                    case_results.append(result)
                    
                    del outputs
                    torch.cuda.empty_cache()
                
                # Print progress for this mode
                success_rate = sum(r['hairpin_found'] for r in case_results if r['patch_mode'] == patch_mode and r['patch_mask_mode'] == mask_mode) / num_blocks
                print(f"    {patch_mode}/{mask_mode}: {success_rate:.1%} success across {num_blocks} blocks")
        
        all_results.extend(case_results)
        
        # Flush results and regenerate plots every flush_every cases
        cases_completed = case_idx + 1
        if cases_completed % flush_every == 0 or cases_completed == len(df):
            print(f"\n  Flushing results ({cases_completed} cases completed)...")
            interim_df = pd.DataFrame(all_results)
            interim_df.to_parquet(results_path, index=False)
            generate_basic_plots(interim_df, output_dir)
        
        # Clean up
        del donor_collected, donor_s_region, donor_z_full
        torch.cuda.empty_cache()
    
    # Final summary
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(results_path, index=False)
    generate_basic_plots(results_df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Total experiments: {len(results_df)}")
    print(f"\nHairpin detection rate by block_idx:")
    print(results_df.groupby("block_idx")["hairpin_found"].mean())
    print(f"\nHairpin detection rate by patch_mode:")
    print(results_df.groupby("patch_mode")["hairpin_found"].mean())
    
    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run single-block ESMFold patching experiments (refactored)"
    )
    parser.add_argument(
        "--parquet", type=str, default=os.path.join(_PROJECT_ROOT, "data", "all_block_patching_results.parquet"),
        help="Path to patching_dataset.parquet"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(_PROJECT_ROOT, "results", "single_block_v2"),
        help="Output directory"
    )
    parser.add_argument(
        "--patch_modes", nargs="+", default=["sequence", "pairwise", "both"],
        help="Patch modes for trunk"
    )
    parser.add_argument(
        "--mask_modes", nargs="+", default=["intra", "touch"],
        help="Pairwise mask modes"
    )
    parser.add_argument(
        "--n_cases", type=int, default=None,
        help="Number of cases to run"
    )
    parser.add_argument(
        "--save_pdbs", action="store_true",
        help="Save PDB structures"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device"
    )
    parser.add_argument(
        "--flush_every", type=int, default=20,
        help="Save results and regenerate plots every N cases"
    )
    
    args = parser.parse_args()
    
    results_df = run_experiment_on_dataset(
        parquet_path=args.parquet,
        output_dir=args.output_dir,
        patch_modes=args.patch_modes,
        patch_mask_modes=args.mask_modes,
        save_pdbs=args.save_pdbs,
        n_cases=args.n_cases,
        device=args.device,
        flush_every=args.flush_every,
    )
    
    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()