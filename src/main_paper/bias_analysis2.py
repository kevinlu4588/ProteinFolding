#!/usr/bin/env python
"""
Pair-to-Sequence Attention Bias Analysis

Analyzes how pairwise representations influence sequence attention via the 
pair_to_sequence bias term in ESMFold transformer blocks. Computes contact 
prediction AUC metrics across blocks to understand when structural information
emerges in the bias terms.

Usage:
    python pair_to_seq_analysis.py --parquet data.parquet --output results/ --n-cases 30
    python pair_to_seq_analysis.py --parquet data.parquet --n-cases 10 --blocks 0 10 20 30 40 47
"""

import argparse
import os
import types
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput, EsmFoldingTrunk
from transformers.utils import ContextManagers


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze pair-to-sequence attention bias in ESMFold',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--parquet', type=str, default='data_old/all_block_patching_results.parquet',
                        help='Path to input parquet file with sequences')
    parser.add_argument('--output', type=str, default='better_bias_analysis',
                        help='Output directory for results')
    parser.add_argument('--n-cases', type=int, default=10,
                        help='Number of sequences to analyze')
    parser.add_argument('--n-visualize', type=int, default=2,
                        help='Number of sequences to generate visualizations for')
    parser.add_argument('--blocks', type=int, nargs='+', default=None,
                        help='Specific blocks to analyze (default: all 0-47)')
    parser.add_argument('--viz-block', type=int, default=30,
                        help='Block to visualize in detail')
    parser.add_argument('--contact-threshold', type=float, default=8.0,
                        help='Distance threshold for contacts (Angstroms)')
    parser.add_argument('--smoothing-window', type=int, default=5,
                        help='Sliding window for metric smoothing (1=no smoothing)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    parser.add_argument('--save-pdb', action='store_true',
                        help='Save PDB strings in output parquet')
    parser.add_argument('--save-biases', action='store_true',
                        help='Save bias tensors (warning: large files)')
    return parser.parse_args()


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class BiasCollectionOutput:
    biases: Dict[int, torch.Tensor]
    attentions_with_bias: Dict[int, torch.Tensor]
    attentions_without_bias: Dict[int, torch.Tensor]
    structure_output: EsmForProteinFoldingOutput
    ca_positions: torch.Tensor
    contact_map: torch.Tensor
    distances: torch.Tensor
    pdb_string: str


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def compute_attention_components(seq_attention_module, x, bias, mask=None):
    """Compute attention with and without bias."""
    module = seq_attention_module
    t = module.proj(x).view(*x.shape[:2], module.num_heads, -1)
    t = t.permute(0, 2, 1, 3)
    q, k, v = t.chunk(3, dim=-1)
    q = module.rescale_factor * q
    raw_scores = torch.einsum("...qc,...kc->...qk", q, k)
    bias_permuted = bias.permute(0, 3, 1, 2)
    if mask is not None:
        raw_scores = raw_scores.masked_fill(mask[:, None, None, :] == False, -1e9)
    attn_without_bias = F.softmax(raw_scores, dim=-1)
    attn_with_bias = F.softmax(raw_scores + bias_permuted, dim=-1)
    return attn_with_bias.permute(0, 2, 3, 1), attn_without_bias.permute(0, 2, 3, 1)


def collect_all_biases_trunk_forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles,
                                      blocks_to_collect, collected_data):
    """Modified trunk forward that collects biases from all specified blocks in one pass."""
    device = seq_feats.device
    s_s_0, s_z_0 = seq_feats, pair_feats
    
    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles += 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        for block_idx, block in enumerate(self.blocks):
            if block_idx in blocks_to_collect:
                bias = block.pair_to_sequence(z)
                collected_data['biases'][block_idx] = bias.detach().cpu()
                seq_normed = block.layernorm_1(s)
                attn_with, attn_without = compute_attention_components(block.seq_attention, seq_normed, bias, mask)
                collected_data['attn_with_bias'][block_idx] = attn_with.detach().cpu()
                collected_data['attn_without_bias'][block_idx] = attn_without.detach().cpu()
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
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
                structure["positions"][-1][:, :, :3], 3.375, 21.375, self.recycle_bins,
            )

    structure["s_s"], structure["s_z"] = s_s, s_z
    return structure


def collect_biases_single_pass(model, tokenizer, device, sequence, blocks_to_collect, contact_threshold=8.0):
    """Collect all biases in a single forward pass."""
    collected_data = {'biases': {}, 'attn_with_bias': {}, 'attn_without_bias': {}}
    original_trunk_forward = model.trunk.forward
    
    def bound_forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        return collect_all_biases_trunk_forward(
            self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles,
            blocks_to_collect, collected_data
        )
    
    model.trunk.forward = types.MethodType(bound_forward, model.trunk)
    
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(**inputs, num_recycles=0)
    
    model.trunk.forward = original_trunk_forward
    
    # Get structure info
    ca_positions = outputs.positions[-1, 0, :, 1, :]  # CA is atom index 1
    diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1))
    contacts = (distances < contact_threshold).float()
    pdb_string = model.output_to_pdb(outputs)[0]
    
    return BiasCollectionOutput(
        biases=collected_data['biases'],
        attentions_with_bias=collected_data['attn_with_bias'],
        attentions_without_bias=collected_data['attn_without_bias'],
        structure_output=outputs,
        ca_positions=ca_positions.cpu(),
        contact_map=contacts.cpu(),
        distances=distances.cpu(),
        pdb_string=pdb_string,
    )


def compute_metrics(bias, contacts, distances, region_start=0, region_end=None, region_name="global"):
    """Compute correlation and AUC metrics for a region."""
    if region_end is None:
        region_end = bias.shape[1]
    
    bias_region = bias[0, region_start:region_end, region_start:region_end, :].mean(dim=-1)
    contacts_region = contacts[region_start:region_end, region_start:region_end]
    distances_region = distances[region_start:region_end, region_start:region_end]
    region_len = region_end - region_start
    
    triu_idx = np.triu_indices(region_len, k=1)
    bias_flat = bias_region[triu_idx].cpu().numpy()
    contacts_flat = contacts_region[triu_idx].cpu().numpy()
    dist_flat = distances_region[triu_idx].cpu().numpy()
    
    results = {f'{region_name}_n_pairs': len(bias_flat)}
    valid = ~np.isnan(bias_flat) & ~np.isnan(dist_flat)
    if valid.sum() < 3:
        return results
    
    results[f'{region_name}_bias_dist_pearson'], _ = pearsonr(dist_flat[valid], bias_flat[valid])
    
    n_contacts = contacts_flat[valid].sum()
    if n_contacts > 0 and n_contacts < len(contacts_flat[valid]):
        try:
            results[f'{region_name}_auc'] = roc_auc_score(contacts_flat[valid], bias_flat[valid])
        except:
            pass
    
    contact_mask = contacts_flat > 0.5
    if contact_mask.sum() > 0 and (~contact_mask).sum() > 0:
        results[f'{region_name}_mean_bias_at_contacts'] = float(bias_flat[contact_mask].mean())
        results[f'{region_name}_mean_bias_at_noncontacts'] = float(bias_flat[~contact_mask].mean())
        results[f'{region_name}_contact_bias_diff'] = (
            results[f'{region_name}_mean_bias_at_contacts'] - results[f'{region_name}_mean_bias_at_noncontacts']
        )
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def visualize_bias_and_contacts(bias, contacts, distances, block_idx, hairpin_start, hairpin_end, 
                                 sequence_name="", contact_threshold=8.0, output_path=None):
    """Comprehensive visualization of bias and contacts."""
    bias_avg = bias[0].mean(dim=-1).cpu().numpy()
    contacts_np = contacts.cpu().numpy()
    distances_np = distances.cpu().numpy()
    
    L = bias_avg.shape[0]
    hp_len = hairpin_end - hairpin_start
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Global protein
    ax = axes[0, 0]
    im = ax.imshow(contacts_np, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Global Contacts (<{contact_threshold}Å)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    rect = Rectangle((hairpin_start-0.5, hairpin_start-0.5), hp_len, hp_len, lw=2, ec='red', fc='none')
    ax.add_patch(rect)
    
    ax = axes[0, 1]
    vmax = np.percentile(np.abs(bias_avg), 98)
    im = ax.imshow(bias_avg, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Global Bias (Block {block_idx})')
    plt.colorbar(im, ax=ax, fraction=0.046)
    rect = Rectangle((hairpin_start-0.5, hairpin_start-0.5), hp_len, hp_len, lw=2, ec='green', fc='none')
    ax.add_patch(rect)
    
    ax = axes[0, 2]
    im = ax.imshow(bias_avg, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.contour(contacts_np, levels=[0.5], colors='lime', linewidths=1.5)
    ax.set_title('Global: Bias + Contacts')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 3]
    triu_idx = np.triu_indices(L, k=1)
    n_points = min(5000, len(bias_avg[triu_idx]))
    idx = np.random.choice(len(bias_avg[triu_idx]), n_points, replace=False)
    colors = ['green' if contacts_np[triu_idx][i] > 0.5 else 'gray' for i in idx]
    ax.scatter(distances_np[triu_idx][idx], bias_avg[triu_idx][idx], c=colors, alpha=0.3, s=10)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=contact_threshold, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('Bias')
    ax.set_title('Global: Bias vs Distance')
    
    # Row 2: Hairpin region
    hp_contacts = contacts_np[hairpin_start:hairpin_end, hairpin_start:hairpin_end]
    hp_bias = bias_avg[hairpin_start:hairpin_end, hairpin_start:hairpin_end]
    hp_dists = distances_np[hairpin_start:hairpin_end, hairpin_start:hairpin_end]
    
    ax = axes[1, 0]
    im = ax.imshow(hp_contacts, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Hairpin Contacts [{hairpin_start}:{hairpin_end}]')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 1]
    hp_vmax = max(np.percentile(np.abs(hp_bias), 98), 0.01)
    im = ax.imshow(hp_bias, cmap='RdBu_r', aspect='auto', vmin=-hp_vmax, vmax=hp_vmax)
    ax.set_title(f'Hairpin Bias (Block {block_idx})')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 2]
    im = ax.imshow(hp_bias, cmap='RdBu_r', aspect='auto', vmin=-hp_vmax, vmax=hp_vmax)
    ax.contour(hp_contacts, levels=[0.5], colors='lime', linewidths=2)
    ax.set_title('Hairpin: Bias + Contacts')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 3]
    triu_hp = np.triu_indices(hp_len, k=1)
    colors = ['green' if hp_contacts[triu_hp][i] > 0.5 else 'gray' for i in range(len(hp_bias[triu_hp]))]
    ax.scatter(hp_dists[triu_hp], hp_bias[triu_hp], c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=contact_threshold, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('Bias')
    
    # Add metrics
    valid = ~np.isnan(hp_bias[triu_hp]) & ~np.isnan(hp_dists[triu_hp])
    if valid.sum() > 2:
        r, _ = pearsonr(hp_dists[triu_hp][valid], hp_bias[triu_hp][valid])
        try:
            auc = roc_auc_score(hp_contacts[triu_hp][valid], hp_bias[triu_hp][valid])
            ax.set_title(f'Hairpin: r={r:.2f}, AUC={auc:.2f}')
        except:
            ax.set_title(f'Hairpin: r={r:.2f}')
    
    plt.suptitle(f'{sequence_name} - Block {block_idx}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_metrics_summary(all_metrics, blocks, smoothing_window=1, output_path=None):
    """Plot aggregate metrics across blocks with optional sliding window smoothing."""
    df = pd.DataFrame(all_metrics)
    agg = df.groupby('block_idx').agg({
        'global_bias_dist_pearson': 'mean',
        'global_auc': 'mean',
        'global_contact_bias_diff': 'mean',
        'hairpin_bias_dist_pearson': 'mean',
        'hairpin_auc': 'mean',
        'hairpin_contact_bias_diff': 'mean',
    }).reset_index()
    
    # Apply sliding window smoothing if window > 1
    if smoothing_window > 1:
        for col in ['global_bias_dist_pearson', 'global_auc', 'global_contact_bias_diff',
                    'hairpin_bias_dist_pearson', 'hairpin_auc', 'hairpin_contact_bias_diff']:
            agg[f'{col}_smooth'] = agg[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Helper to get smoothed or raw column
    def get_col(name):
        return f'{name}_smooth' if smoothing_window > 1 else name
    
    # Global
    axes[0, 0].plot(agg['block_idx'], agg[get_col('global_bias_dist_pearson')], 'o-', lw=2)
    if smoothing_window > 1:
        axes[0, 0].plot(agg['block_idx'], agg['global_bias_dist_pearson'], 'o', alpha=0.3, color='gray', ms=4)
    axes[0, 0].axhline(y=0, color='gray', ls='--', alpha=0.5)
    axes[0, 0].set_title('Global: Bias vs Distance (r)')
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(agg['block_idx'], agg[get_col('global_auc')], 'o-', color='green', lw=2)
    if smoothing_window > 1:
        axes[0, 1].plot(agg['block_idx'], agg['global_auc'], 'o', alpha=0.3, color='gray', ms=4)
    axes[0, 1].axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    axes[0, 1].set_title('Global: Contact AUC')
    axes[0, 1].set_xlabel('Block')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.3)
    
    axes[0, 2].plot(agg['block_idx'], agg[get_col('global_contact_bias_diff')], 'o-', color='purple', lw=2)
    if smoothing_window > 1:
        axes[0, 2].plot(agg['block_idx'], agg['global_contact_bias_diff'], 'o', alpha=0.3, color='gray', ms=4)
    axes[0, 2].axhline(y=0, color='gray', ls='--', alpha=0.5)
    axes[0, 2].set_title('Global: Contact - Noncontact Bias')
    axes[0, 2].set_xlabel('Block')
    axes[0, 2].grid(alpha=0.3)
    
    # Hairpin
    axes[1, 0].plot(agg['block_idx'], agg[get_col('hairpin_bias_dist_pearson')], 'o-', lw=2)
    if smoothing_window > 1:
        axes[1, 0].plot(agg['block_idx'], agg['hairpin_bias_dist_pearson'], 'o', alpha=0.3, color='gray', ms=4)
    axes[1, 0].axhline(y=0, color='gray', ls='--', alpha=0.5)
    axes[1, 0].set_title('Hairpin: Bias vs Distance (r)')
    axes[1, 0].set_xlabel('Block')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(agg['block_idx'], agg[get_col('hairpin_auc')], 'o-', color='green', lw=2)
    if smoothing_window > 1:
        axes[1, 1].plot(agg['block_idx'], agg['hairpin_auc'], 'o', alpha=0.3, color='gray', ms=4)
    axes[1, 1].axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    axes[1, 1].set_title('Hairpin: Contact AUC')
    axes[1, 1].set_xlabel('Block')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.3)
    
    axes[1, 2].plot(agg['block_idx'], agg[get_col('hairpin_contact_bias_diff')], 'o-', color='purple', lw=2)
    if smoothing_window > 1:
        axes[1, 2].plot(agg['block_idx'], agg['hairpin_contact_bias_diff'], 'o', alpha=0.3, color='gray', ms=4)
    axes[1, 2].axhline(y=0, color='gray', ls='--', alpha=0.5)
    axes[1, 2].set_title('Hairpin: Contact - Noncontact Bias')
    axes[1, 2].set_xlabel('Block')
    axes[1, 2].grid(alpha=0.3)
    
    smooth_label = f' (smoothed, window={smoothing_window})' if smoothing_window > 1 else ''
    plt.suptitle(f'Aggregate Metrics Across All Sequences{smooth_label}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return agg


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()
    
    # Setup
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Determine blocks to analyze
    if args.blocks:
        blocks_to_analyze = args.blocks
    else:
        blocks_to_analyze = list(range(48))
    
    print(f"Analyzing {args.n_cases} sequences across {len(blocks_to_analyze)} blocks")
    print(f"Visualizing first {args.n_visualize} sequences at block {args.viz_block}")
    
    # Load model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print(f"Model loaded. Attention heads: {model.trunk.blocks[0].seq_attention.num_heads}")
    
    # Load data
    print(f"\nLoading data from {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    donors = df[['donor_pdb', 'donor_sequence', 'donor_hairpin_start', 'donor_hairpin_end']].drop_duplicates()
    donors = donors.head(args.n_cases)
    print(f"Found {len(donors)} unique sequences")
    
    # Run analysis
    all_metrics = []
    sequence_info = []
    
    for seq_idx, (idx, row) in enumerate(tqdm(donors.iterrows(), total=len(donors), desc="Sequences")):
        donor_pdb = row['donor_pdb']
        sequence = row['donor_sequence']
        hairpin_start = int(row['donor_hairpin_start'])
        hairpin_end = int(row['donor_hairpin_end'])
        
        should_visualize = seq_idx < args.n_visualize
        
        print(f"\n{'='*60}")
        print(f"{donor_pdb}: hairpin [{hairpin_start}:{hairpin_end}], len={len(sequence)}")
        print('='*60)
        
        # Collect all biases in single forward pass
        data = collect_biases_single_pass(
            model, tokenizer, device, sequence, 
            blocks_to_analyze, args.contact_threshold
        )
        
        # Store sequence info
        seq_info = {
            'seq_idx': seq_idx,
            'donor_pdb': donor_pdb,
            'sequence': sequence,
            'seq_len': len(sequence),
            'hairpin_start': hairpin_start,
            'hairpin_end': hairpin_end,
            'hairpin_len': hairpin_end - hairpin_start,
        }
        if args.save_pdb:
            seq_info['pdb_string'] = data.pdb_string
        sequence_info.append(seq_info)
        
        for block_idx in blocks_to_analyze:
            if block_idx not in data.biases:
                continue
            
            bias = data.biases[block_idx]
            
            # Compute metrics
            global_m = compute_metrics(bias, data.contact_map, data.distances, 0, len(sequence), "global")
            hairpin_m = compute_metrics(bias, data.contact_map, data.distances, hairpin_start, hairpin_end, "hairpin")
            
            metrics = {
                'seq_idx': seq_idx,
                'donor_pdb': donor_pdb,
                'block_idx': block_idx,
                **global_m,
                **hairpin_m
            }
            all_metrics.append(metrics)
            
            # Visualize for first N sequences at specified block
            if should_visualize and block_idx == args.viz_block:
                viz_path = os.path.join(args.output, f'bias_contacts_{donor_pdb}_block{block_idx}.png')
                visualize_bias_and_contacts(
                    bias, data.contact_map, data.distances, block_idx,
                    hairpin_start, hairpin_end, donor_pdb, args.contact_threshold,
                    output_path=viz_path
                )
                print(f"  Saved visualization: {viz_path}")
        
        torch.cuda.empty_cache()
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args.output, 'metrics.parquet')
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")
    
    seq_info_df = pd.DataFrame(sequence_info)
    seq_info_path = os.path.join(args.output, 'sequence_info.parquet')
    seq_info_df.to_parquet(seq_info_path, index=False)
    print(f"Saved sequence info: {seq_info_path}")
    
    # Generate summary plot
    summary_path = os.path.join(args.output, 'metrics_summary.png')
    agg_df = plot_metrics_summary(
        all_metrics, blocks_to_analyze, 
        smoothing_window=args.smoothing_window,
        output_path=summary_path
    )
    print(f"Saved summary plot: {summary_path}")
    
    # Save aggregated metrics
    agg_path = os.path.join(args.output, 'metrics_aggregated.parquet')
    agg_df.to_parquet(agg_path, index=False)
    print(f"Saved aggregated metrics: {agg_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nGlobal AUC by Block (first 10, last 10):")
    print(agg_df[['block_idx', 'global_auc']].head(10).to_string(index=False))
    print("...")
    print(agg_df[['block_idx', 'global_auc']].tail(10).to_string(index=False))
    
    print("\nHairpin AUC by Block (first 10, last 10):")
    print(agg_df[['block_idx', 'hairpin_auc']].head(10).to_string(index=False))
    print("...")
    print(agg_df[['block_idx', 'hairpin_auc']].tail(10).to_string(index=False))
    
    # Key findings
    if 'global_auc' in agg_df.columns:
        best_global = agg_df.loc[agg_df['global_auc'].idxmax()]
        best_hairpin = agg_df.loc[agg_df['hairpin_auc'].idxmax()]
        print(f"\nBest global AUC: {best_global['global_auc']:.3f} at block {int(best_global['block_idx'])}")
        print(f"Best hairpin AUC: {best_hairpin['hairpin_auc']:.3f} at block {int(best_hairpin['block_idx'])}")
    
    print(f"\nResults saved to: {args.output}/")


if __name__ == '__main__':
    main()