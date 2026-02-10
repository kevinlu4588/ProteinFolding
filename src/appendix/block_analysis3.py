"""
Information Flow Analysis in Hairpin Regions
=============================================

Analyzes the relative contribution of pair-to-sequence vs sequence-to-pair
information flow within hairpin regions of ESMFold's folding trunk.

Computes:
- pair2seq bias norms: How much pairwise info influences sequence attention
- seq2pair update norms: How much sequence info updates pairwise representations
- Relative contributions across blocks with optional smoothing

Uses donor sequences (which contain hairpins) to focus on structurally
relevant regions.
"""

import pandas as pd
import numpy as np
import torch
import types
from types import SimpleNamespace
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmForProteinFolding
import os
import gc

# ============================================================================
# Setup
# ============================================================================

OUTPUT_DIR = "information_flow_donor_simplified"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading ESMFold model...")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
print("Model loaded")

NUM_BLOCKS = 48

# Smoothing windows to use
SMOOTHING_WINDOWS = [1, 3, 5]  # 1 = no smoothing


def smooth_series(series, window):
    """Apply sliding window smoothing to a pandas Series."""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ============================================================================
# Monkey patch to collect layer outputs and communication signals
# ============================================================================

def collect_information_flow_trunk(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """
    Collect layer outputs and communication signals.
    """
    from transformers.models.esm.modeling_esmfold import EsmFoldingTrunk
    from transformers.modeling_utils import ContextManagers
    
    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles = no_recycles + 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        
        stats = {
            's_s_list': [],
            's_z_list': [],
            'pair2seq_bias_list': [],
            'seq2pair_update_list': [],
            'seq_attention_output_list': [],
            'triangular_update_list': [],
        }
        
        for block in self.blocks:
            # === PAIR TO SEQUENCE ===
            bias = block.pair_to_sequence(z)
            stats['pair2seq_bias_list'].append(bias.clone())
            
            y = block.layernorm_1(s)
            y, _ = block.seq_attention(y, mask=mask, bias=bias)
            stats['seq_attention_output_list'].append(y.clone())
            
            s = s + block.drop(y)
            s = block.mlp_seq(s)
            
            # === SEQUENCE TO PAIR ===
            seq2pair_update = block.sequence_to_pair(s)
            stats['seq2pair_update_list'].append(seq2pair_update.clone())
            
            z = z + seq2pair_update
            
            # === TRIANGULAR UPDATES ===
            tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
            
            tri_out = block.tri_mul_out(z, mask=tri_mask)
            tri_in = block.tri_mul_in(z, mask=tri_mask)
            tri_att_start = block.tri_att_start(z, mask=tri_mask, chunk_size=self.chunk_size)
            tri_att_end = block.tri_att_end(z, mask=tri_mask, chunk_size=self.chunk_size)
            
            triangular_update = tri_out + tri_in + tri_att_start + tri_att_end
            stats['triangular_update_list'].append(triangular_update.clone())
            
            z = z + block.row_drop(tri_out)
            z = z + block.col_drop(tri_in)
            z = z + block.row_drop(tri_att_start)
            z = z + block.col_drop(tri_att_end)
            z = block.mlp_pair(z)
            
            stats['s_s_list'].append(s.clone())
            stats['s_z_list'].append(z.clone())
            
        return s, z, stats

    s_s = s_s_0
    s_z = s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

            s_s, s_z, stats = trunk_iter(
                s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask
            )

            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa, mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375, 21.375, self.recycle_bins,
            )

    return {"s_s": s_s, "s_z": s_z, **stats}


def collect_information_flow_forward(self, input_ids, attention_mask=None, position_ids=None,
                                      masking_pattern=None, num_recycles=None, **kwargs):
    """Forward pass that returns layer outputs and communication signals."""
    cfg = self.config.esmfold_config
    aa = input_ids
    B, L = aa.shape[0], aa.shape[1]
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)
    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype).detach()
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    
    s_s_0 = self.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(aa)

    structure = self.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
    
    return SimpleNamespace(**structure)


# ============================================================================
# Hairpin region extraction functions
# ============================================================================

def extract_hairpin_sequence(tensor, hairpin_start, hairpin_end):
    """Extract hairpin region from sequence representation. (1, L, D) -> (1, h, D)"""
    return tensor[:, hairpin_start:hairpin_end, :]


def extract_hairpin_pairwise_intra(tensor, hairpin_start, hairpin_end):
    """Extract INTRA-hairpin region (both i,j in hairpin). (1, L, L, D) -> (1, h, h, D)"""
    return tensor[:, hairpin_start:hairpin_end, hairpin_start:hairpin_end, :]


# ============================================================================
# Statistics computation (focused on pair2seq and seq2pair)
# ============================================================================

def compute_focused_stats(outputs, hairpin_start, hairpin_end):
    """
    Compute statistics focused on pair2seq bias and seq2pair update.
    
    Key metrics:
    - Raw norms (MSA = mean squared activation)
    - Relative contributions (normalized by comparison signals)
    - Hairpin focus (hairpin / global ratio)
    """
    stats_list = []
    
    for block_idx in range(NUM_BLOCKS):
        stats = {'block': block_idx}
        
        # Get tensors
        pair2seq_bias = outputs.pair2seq_bias_list[block_idx]
        seq2pair_update = outputs.seq2pair_update_list[block_idx]
        seq_attention_output = outputs.seq_attention_output_list[block_idx]
        triangular_update = outputs.triangular_update_list[block_idx]
        
        # =====================================================================
        # PAIR TO SEQUENCE PATHWAY
        # =====================================================================
        
        # --- GLOBAL ---
        bias_global_msa = (pair2seq_bias ** 2).mean().item()
        seq_attn_out_global_msa = (seq_attention_output ** 2).mean().item()
        
        stats['pair2seq_bias_global_norm'] = np.sqrt(bias_global_msa)
        stats['seq_attn_out_global_norm'] = np.sqrt(seq_attn_out_global_msa)
        stats['pair2seq_relative_global'] = (
            np.sqrt(bias_global_msa) / (np.sqrt(seq_attn_out_global_msa) + 1e-10)
        )
        
        # --- HAIRPIN ---
        bias_hairpin = extract_hairpin_pairwise_intra(pair2seq_bias, hairpin_start, hairpin_end)
        seq_attn_out_hairpin = extract_hairpin_sequence(seq_attention_output, hairpin_start, hairpin_end)
        
        bias_hairpin_msa = (bias_hairpin ** 2).mean().item()
        seq_attn_out_hairpin_msa = (seq_attn_out_hairpin ** 2).mean().item()
        
        stats['pair2seq_bias_hairpin_norm'] = np.sqrt(bias_hairpin_msa)
        stats['seq_attn_out_hairpin_norm'] = np.sqrt(seq_attn_out_hairpin_msa)
        stats['pair2seq_relative_hairpin'] = (
            np.sqrt(bias_hairpin_msa) / (np.sqrt(seq_attn_out_hairpin_msa) + 1e-10)
        )
        
        # --- HAIRPIN FOCUS ---
        stats['pair2seq_bias_hairpin_focus'] = np.sqrt(bias_hairpin_msa) / (np.sqrt(bias_global_msa) + 1e-10)
        
        # =====================================================================
        # SEQUENCE TO PAIR PATHWAY
        # =====================================================================
        
        # --- GLOBAL ---
        seq2pair_global_msa = (seq2pair_update ** 2).mean().item()
        tri_global_msa = (triangular_update ** 2).mean().item()
        
        stats['seq2pair_update_global_norm'] = np.sqrt(seq2pair_global_msa)
        stats['triangular_update_global_norm'] = np.sqrt(tri_global_msa)
        stats['seq2pair_relative_global'] = (
            np.sqrt(seq2pair_global_msa) / (np.sqrt(tri_global_msa) + 1e-10)
        )
        stats['seq2pair_fraction_global'] = (
            np.sqrt(seq2pair_global_msa) / 
            (np.sqrt(seq2pair_global_msa) + np.sqrt(tri_global_msa) + 1e-10)
        )
        
        # --- HAIRPIN ---
        seq2pair_hairpin = extract_hairpin_pairwise_intra(seq2pair_update, hairpin_start, hairpin_end)
        tri_hairpin = extract_hairpin_pairwise_intra(triangular_update, hairpin_start, hairpin_end)
        
        seq2pair_hairpin_msa = (seq2pair_hairpin ** 2).mean().item()
        tri_hairpin_msa = (tri_hairpin ** 2).mean().item()
        
        stats['seq2pair_update_hairpin_norm'] = np.sqrt(seq2pair_hairpin_msa)
        stats['triangular_update_hairpin_norm'] = np.sqrt(tri_hairpin_msa)
        stats['seq2pair_relative_hairpin'] = (
            np.sqrt(seq2pair_hairpin_msa) / (np.sqrt(tri_hairpin_msa) + 1e-10)
        )
        stats['seq2pair_fraction_hairpin'] = (
            np.sqrt(seq2pair_hairpin_msa) / 
            (np.sqrt(seq2pair_hairpin_msa) + np.sqrt(tri_hairpin_msa) + 1e-10)
        )
        
        # --- HAIRPIN FOCUS ---
        stats['seq2pair_update_hairpin_focus'] = np.sqrt(seq2pair_hairpin_msa) / (np.sqrt(seq2pair_global_msa) + 1e-10)
        
        stats_list.append(stats)
    
    return stats_list


# ============================================================================
# Load dataset - DONOR sequences
# ============================================================================

df = pd.read_csv("data/single_block_patching_successes.csv")
print(f"Loaded {len(df)} rows")

# Get unique DONOR sequences with their hairpin info
sample_df = df.drop_duplicates(subset=['donor_sequence']).reset_index(drop=True)
print(f"Analyzing {len(sample_df)} unique DONOR sequences")

# Check available columns
print(f"Available columns: {df.columns.tolist()}")

# Set up model
model.forward = types.MethodType(collect_information_flow_forward, model)
model.trunk.forward = types.MethodType(collect_information_flow_trunk, model.trunk)

# ============================================================================
# Initialize accumulators
# ============================================================================

all_stats = []
successful = 0
failed = 0

# ============================================================================
# Main processing loop
# ============================================================================

print("\n" + "="*70)
print("PROCESSING DONOR SEQUENCES")
print("="*70)

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing"):
    seq = row['donor_sequence']
    
    # Get hairpin info
    hairpin_start = int(row.get('donor_patch_start', row.get('donor_hairpin_start', 0)))
    hairpin_end = int(row.get('donor_patch_end', row.get('donor_hairpin_end', len(seq))))
    hairpin_len = hairpin_end - hairpin_start
    
    # Validate hairpin info
    if hairpin_start >= hairpin_end or hairpin_end > len(seq) or hairpin_len < 3:
        failed += 1
        continue
    
    try:
        # Forward pass
        with torch.no_grad():
            inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)
        
        # Compute focused stats
        seq_stats = compute_focused_stats(outputs, hairpin_start, hairpin_end)
        
        for block_stats in seq_stats:
            block_stats['seq_idx'] = idx
            block_stats['seq_len'] = len(seq)
            block_stats['hairpin_start'] = hairpin_start
            block_stats['hairpin_end'] = hairpin_end
            block_stats['hairpin_len'] = hairpin_len
            all_stats.append(block_stats)
        
        del outputs
        torch.cuda.empty_cache()
        
        successful += 1
        
    except Exception as e:
        print(f"\n  Failed on seq {idx}: {e}")
        failed += 1
        torch.cuda.empty_cache()
        gc.collect()
        continue
    
    # Checkpoint
    if (idx + 1) % 50 == 0:
        print(f"\n  Checkpoint: {successful} successful, {failed} failed")
        pd.DataFrame(all_stats).to_parquet(f'{OUTPUT_DIR}/stats_checkpoint.parquet', index=False)

print(f"\n\nCompleted: {successful} successful, {failed} failed")

# ============================================================================
# Aggregate results
# ============================================================================

stats_df = pd.DataFrame(all_stats)
print(f"\nCollected {len(stats_df)} stat rows from {stats_df['seq_idx'].nunique()} sequences")


# ============================================================================
# Plotting functions
# ============================================================================

def plot_with_smoothing(stats_df, window, output_dir):
    """Generate all plots with specified smoothing window."""
    
    window_label = "No smoothing" if window <= 1 else f"Window={window}"
    suffix = f"_w{window}"
    
    # =========================================================================
    # Plot 1: Raw norms - pair2seq bias and seq2pair update
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1a: pair2seq bias norms
    ax = axes[0, 0]
    global_avg = stats_df.groupby('block')['pair2seq_bias_global_norm'].mean()
    global_std = stats_df.groupby('block')['pair2seq_bias_global_norm'].std()
    hairpin_avg = stats_df.groupby('block')['pair2seq_bias_hairpin_norm'].mean()
    hairpin_std = stats_df.groupby('block')['pair2seq_bias_hairpin_norm'].std()
    
    global_smooth = smooth_series(global_avg, window)
    hairpin_smooth = smooth_series(hairpin_avg, window)
    
    ax.plot(global_smooth.index, global_smooth.values, 'b--', linewidth=2, label='Global')
    ax.fill_between(global_avg.index, 
                    smooth_series(global_avg - global_std, window), 
                    smooth_series(global_avg + global_std, window), 
                    alpha=0.15, color='blue')
    ax.plot(hairpin_smooth.index, hairpin_smooth.values, 'b-', linewidth=2, label='Hairpin')
    ax.fill_between(hairpin_avg.index, 
                    smooth_series(hairpin_avg - hairpin_std, window), 
                    smooth_series(hairpin_avg + hairpin_std, window), 
                    alpha=0.15, color='cyan')
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('RMS Norm')
    ax.set_title(f'Pair→Seq Bias Norm')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 1b: seq2pair update norms
    ax = axes[0, 1]
    global_avg = stats_df.groupby('block')['seq2pair_update_global_norm'].mean()
    global_std = stats_df.groupby('block')['seq2pair_update_global_norm'].std()
    hairpin_avg = stats_df.groupby('block')['seq2pair_update_hairpin_norm'].mean()
    hairpin_std = stats_df.groupby('block')['seq2pair_update_hairpin_norm'].std()
    
    global_smooth = smooth_series(global_avg, window)
    hairpin_smooth = smooth_series(hairpin_avg, window)
    
    ax.plot(global_smooth.index, global_smooth.values, 'g--', linewidth=2, label='Global')
    ax.fill_between(global_avg.index, 
                    smooth_series(global_avg - global_std, window), 
                    smooth_series(global_avg + global_std, window), 
                    alpha=0.15, color='green')
    ax.plot(hairpin_smooth.index, hairpin_smooth.values, 'g-', linewidth=2, label='Hairpin')
    ax.fill_between(hairpin_avg.index, 
                    smooth_series(hairpin_avg - hairpin_std, window), 
                    smooth_series(hairpin_avg + hairpin_std, window), 
                    alpha=0.15, color='lime')
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('RMS Norm')
    ax.set_title(f'Seq→Pair Update Norm')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 1c: Hairpin focus - pair2seq bias
    ax = axes[1, 0]
    focus_avg = stats_df.groupby('block')['pair2seq_bias_hairpin_focus'].mean()
    focus_std = stats_df.groupby('block')['pair2seq_bias_hairpin_focus'].std()
    focus_smooth = smooth_series(focus_avg, window)
    
    ax.plot(focus_smooth.index, focus_smooth.values, 'b-', linewidth=2)
    ax.fill_between(focus_avg.index,
                    smooth_series(focus_avg - focus_std, window),
                    smooth_series(focus_avg + focus_std, window),
                    alpha=0.2, color='blue')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('Hairpin / Global')
    ax.set_title(f'Pair→Seq Bias: Hairpin Focus')
    ax.grid(alpha=0.3)
    
    # 1d: Hairpin focus - seq2pair update
    ax = axes[1, 1]
    focus_avg = stats_df.groupby('block')['seq2pair_update_hairpin_focus'].mean()
    focus_std = stats_df.groupby('block')['seq2pair_update_hairpin_focus'].std()
    focus_smooth = smooth_series(focus_avg, window)
    
    ax.plot(focus_smooth.index, focus_smooth.values, 'g-', linewidth=2)
    ax.fill_between(focus_avg.index,
                    smooth_series(focus_avg - focus_std, window),
                    smooth_series(focus_avg + focus_std, window),
                    alpha=0.2, color='green')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('Hairpin / Global')
    ax.set_title(f'Seq→Pair Update: Hairpin Focus')
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Raw Norms and Hairpin Focus ({window_label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/norms{suffix}.png', dpi=150)
    plt.close()
    
    # =========================================================================
    # Plot 2: Relative contributions
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2a: pair2seq relative (bias / attn output)
    ax = axes[0, 0]
    global_avg = stats_df.groupby('block')['pair2seq_relative_global'].mean()
    global_std = stats_df.groupby('block')['pair2seq_relative_global'].std()
    hairpin_avg = stats_df.groupby('block')['pair2seq_relative_hairpin'].mean()
    hairpin_std = stats_df.groupby('block')['pair2seq_relative_hairpin'].std()
    
    global_smooth = smooth_series(global_avg, window)
    hairpin_smooth = smooth_series(hairpin_avg, window)
    
    ax.plot(global_smooth.index, global_smooth.values, 'b--', linewidth=2, label='Global')
    ax.fill_between(global_avg.index,
                    smooth_series(global_avg - global_std, window),
                    smooth_series(global_avg + global_std, window),
                    alpha=0.15, color='blue')
    ax.plot(hairpin_smooth.index, hairpin_smooth.values, 'b-', linewidth=2, label='Hairpin')
    ax.fill_between(hairpin_avg.index,
                    smooth_series(hairpin_avg - hairpin_std, window),
                    smooth_series(hairpin_avg + hairpin_std, window),
                    alpha=0.15, color='cyan')
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('||bias|| / ||attn_output||')
    ax.set_title(f'Pair→Seq: Bias Relative to Attention Output')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2b: seq2pair relative (update / triangular)
    ax = axes[0, 1]
    global_avg = stats_df.groupby('block')['seq2pair_relative_global'].mean()
    global_std = stats_df.groupby('block')['seq2pair_relative_global'].std()
    hairpin_avg = stats_df.groupby('block')['seq2pair_relative_hairpin'].mean()
    hairpin_std = stats_df.groupby('block')['seq2pair_relative_hairpin'].std()
    
    global_smooth = smooth_series(global_avg, window)
    hairpin_smooth = smooth_series(hairpin_avg, window)
    
    ax.plot(global_smooth.index, global_smooth.values, 'g--', linewidth=2, label='Global')
    ax.fill_between(global_avg.index,
                    smooth_series(global_avg - global_std, window),
                    smooth_series(global_avg + global_std, window),
                    alpha=0.15, color='green')
    ax.plot(hairpin_smooth.index, hairpin_smooth.values, 'g-', linewidth=2, label='Hairpin')
    ax.fill_between(hairpin_avg.index,
                    smooth_series(hairpin_avg - hairpin_std, window),
                    smooth_series(hairpin_avg + hairpin_std, window),
                    alpha=0.15, color='lime')
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('||seq2pair|| / ||triangular||')
    ax.set_title(f'Seq→Pair: Update Relative to Triangular')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2c: seq2pair fraction
    ax = axes[1, 0]
    global_avg = stats_df.groupby('block')['seq2pair_fraction_global'].mean()
    hairpin_avg = stats_df.groupby('block')['seq2pair_fraction_hairpin'].mean()
    
    global_smooth = smooth_series(global_avg, window)
    hairpin_smooth = smooth_series(hairpin_avg, window)
    
    ax.plot(global_smooth.index, global_smooth.values, 'purple', linestyle='--', linewidth=2, label='Global')
    ax.plot(hairpin_smooth.index, hairpin_smooth.values, 'purple', linestyle='-', linewidth=2, label='Hairpin')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('Fraction')
    ax.set_title(f'Seq→Pair Fraction of Pairwise Update')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2d: Combined view - both pathways (hairpin only, scaled)
    ax = axes[1, 1]
    pair2seq_h = stats_df.groupby('block')['pair2seq_relative_hairpin'].mean()
    seq2pair_h = stats_df.groupby('block')['seq2pair_relative_hairpin'].mean()
    
    pair2seq_smooth = smooth_series(pair2seq_h, window)
    seq2pair_smooth = smooth_series(seq2pair_h, window)
    
    # Scale to 0-1 for comparison
    pair2seq_scaled = (pair2seq_smooth - pair2seq_smooth.min()) / (pair2seq_smooth.max() - pair2seq_smooth.min() + 1e-10)
    seq2pair_scaled = (seq2pair_smooth - seq2pair_smooth.min()) / (seq2pair_smooth.max() - seq2pair_smooth.min() + 1e-10)
    
    ax.plot(pair2seq_scaled.index, pair2seq_scaled.values, 'b-', linewidth=2, label='Pair→Seq (scaled)')
    ax.plot(seq2pair_scaled.index, seq2pair_scaled.values, 'g-', linewidth=2, label='Seq→Pair (scaled)')
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('Scaled Relative Contribution')
    ax.set_title(f'Information Flow (Hairpin, Normalized 0-1)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Relative Contributions ({window_label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/relative_contributions{suffix}.png', dpi=150)
    plt.close()
    
    # =========================================================================
    # Plot 3: Combined summary - all key metrics on one figure
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Norms
    ax = axes[0, 0]
    for metric, color, label in [
        ('pair2seq_bias_global_norm', 'blue', 'Pair→Seq Bias (Global)'),
        ('pair2seq_bias_hairpin_norm', 'cyan', 'Pair→Seq Bias (Hairpin)'),
    ]:
        avg = stats_df.groupby('block')[metric].mean()
        ax.plot(smooth_series(avg, window).index, smooth_series(avg, window).values, 
                color=color, linewidth=2, label=label)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('RMS Norm')
    ax.set_title('Pair→Seq Bias Norms')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    for metric, color, label in [
        ('seq2pair_update_global_norm', 'green', 'Seq→Pair Update (Global)'),
        ('seq2pair_update_hairpin_norm', 'lime', 'Seq→Pair Update (Hairpin)'),
    ]:
        avg = stats_df.groupby('block')[metric].mean()
        ax.plot(smooth_series(avg, window).index, smooth_series(avg, window).values, 
                color=color, linewidth=2, label=label)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('RMS Norm')
    ax.set_title('Seq→Pair Update Norms')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    for metric, color, label in [
        ('triangular_update_global_norm', 'orange', 'Triangular (Global)'),
        ('triangular_update_hairpin_norm', 'red', 'Triangular (Hairpin)'),
    ]:
        avg = stats_df.groupby('block')[metric].mean()
        ax.plot(smooth_series(avg, window).index, smooth_series(avg, window).values, 
                color=color, linewidth=2, label=label)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('RMS Norm')
    ax.set_title('Triangular Update Norms')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 2: Relative contributions
    ax = axes[1, 0]
    for metric, color, ls, label in [
        ('pair2seq_relative_global', 'blue', '--', 'Global'),
        ('pair2seq_relative_hairpin', 'blue', '-', 'Hairpin'),
    ]:
        avg = stats_df.groupby('block')[metric].mean()
        ax.plot(smooth_series(avg, window).index, smooth_series(avg, window).values, 
                color=color, linestyle=ls, linewidth=2, label=label)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('||bias|| / ||attn_output||')
    ax.set_title('Pair→Seq Relative Contribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    for metric, color, ls, label in [
        ('seq2pair_relative_global', 'green', '--', 'Global'),
        ('seq2pair_relative_hairpin', 'green', '-', 'Hairpin'),
    ]:
        avg = stats_df.groupby('block')[metric].mean()
        ax.plot(smooth_series(avg, window).index, smooth_series(avg, window).values, 
                color=color, linestyle=ls, linewidth=2, label=label)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('||seq2pair|| / ||triangular||')
    ax.set_title('Seq→Pair Relative Contribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    for metric, color, label in [
        ('pair2seq_bias_hairpin_focus', 'blue', 'Pair→Seq Bias'),
        ('seq2pair_update_hairpin_focus', 'green', 'Seq→Pair Update'),
    ]:
        avg = stats_df.groupby('block')[metric].mean()
        ax.plot(smooth_series(avg, window).index, smooth_series(avg, window).values, 
                color=color, linewidth=2, label=label)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Block')
    ax.set_ylabel('Hairpin / Global')
    ax.set_title('Hairpin Focus')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Information Flow Summary ({window_label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary{suffix}.png', dpi=150)
    plt.close()


# ============================================================================
# Generate plots for all smoothing windows
# ============================================================================

print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

for window in SMOOTHING_WINDOWS:
    print(f"  Generating plots with window={window}...")
    plot_with_smoothing(stats_df, window, OUTPUT_DIR)


# ============================================================================
# Print summary
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nProcessed: {successful} successful, {failed} failed")
print(f"Total sequences: {stats_df['seq_idx'].nunique()}")
print(f"Average sequence length: {stats_df['seq_len'].mean():.1f}")
print(f"Average hairpin length: {stats_df['hairpin_len'].mean():.1f}")

print("\n--- Relative Contributions by Regime (Hairpin) ---")
print(f"{'Metric':<35} {'Early (0-15)':<15} {'Mid (15-35)':<15} {'Late (35-48)':<15}")
print("-" * 80)

for metric in ['pair2seq_relative_hairpin', 'seq2pair_relative_hairpin', 'seq2pair_fraction_hairpin']:
    early = stats_df[stats_df['block'] < 15][metric].mean()
    mid = stats_df[(stats_df['block'] >= 15) & (stats_df['block'] < 35)][metric].mean()
    late = stats_df[stats_df['block'] >= 35][metric].mean()
    print(f"{metric:<35} {early:<15.3f} {mid:<15.3f} {late:<15.3f}")

print("\n--- Raw Norms by Regime (Hairpin) ---")
for metric in ['pair2seq_bias_hairpin_norm', 'seq2pair_update_hairpin_norm']:
    early = stats_df[stats_df['block'] < 15][metric].mean()
    mid = stats_df[(stats_df['block'] >= 15) & (stats_df['block'] < 35)][metric].mean()
    late = stats_df[stats_df['block'] >= 35][metric].mean()
    print(f"{metric:<35} {early:<15.3f} {mid:<15.3f} {late:<15.3f}")

print("\n--- Hairpin Focus by Regime ---")
for metric in ['pair2seq_bias_hairpin_focus', 'seq2pair_update_hairpin_focus']:
    early = stats_df[stats_df['block'] < 15][metric].mean()
    mid = stats_df[(stats_df['block'] >= 15) & (stats_df['block'] < 35)][metric].mean()
    late = stats_df[stats_df['block'] >= 35][metric].mean()
    print(f"{metric:<35} {early:<15.3f} {mid:<15.3f} {late:<15.3f}")


# ============================================================================
# Save outputs
# ============================================================================

stats_df.to_parquet(f'{OUTPUT_DIR}/stats.parquet', index=False)

metadata = {
    'total_sequences': len(sample_df),
    'successful': successful,
    'failed': failed,
    'smoothing_windows': SMOOTHING_WINDOWS,
}
pd.DataFrame([metadata]).to_csv(f'{OUTPUT_DIR}/metadata.csv', index=False)

print(f"\nSaved outputs to {OUTPUT_DIR}/")
print("Done!")