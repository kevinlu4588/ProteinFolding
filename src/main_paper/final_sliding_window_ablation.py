"""
Sliding Window Patching Ablation
================================

Systematically ablates the contribution of sequence (s) vs pairwise (z)
representations by patching with sliding windows across trunk blocks.

Tests three patch modes:
- sequence: Patch only s representation
- pairwise: Patch only z representation
- combined: Patch both s and z

This reveals that pairwise representations carry the critical structural
information, with sequence representations playing a supporting role.

Usage:
    python final_sliding_window_ablation.py \
        --ablation_csv successful_cases.csv \
        --n_sequence_cases 400 \
        --n_pairwise_cases 200 \
        --output_dir results/
"""

import argparse
import os
import types
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# Suppress BioPython DSSP warnings
warnings.filterwarnings("ignore", message=".*mmCIF.*")
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
    to_pdb,
)
from transformers.utils import ContextManagers, ModelOutput
from src.utils.trunk_utils import detect_hairpins
# ============================================================================
# Alpha Helix Content Calculation
# ============================================================================

def compute_alpha_helix_content(pdb_string: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Compute the percentage of residues in alpha helix from a PDB string.
    Returns (helix_count, total_count, helix_percentage) or (None, None, None) if DSSP fails.
    
    Requires: trunk_utils.py with run_dssp_on_pdb function
    """
    import tempfile
    
    try:
        from src.utils.trunk_utils import run_dssp_on_pdb
    except ImportError:
        print("Warning: utils.trunk_utils not found, cannot compute helix content")
        return None, None, None
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as f:
        f.write(pdb_string)
        pdb_path = f.name
    
    try:
        structure, dssp_df = run_dssp_on_pdb(pdb_path)
        if dssp_df is None:
            return None, None, None
        
        # Count helix residues (H, G, I are helix types in DSSP)
        # SimpleSS == "H" covers all helix types
        total_residues = len(dssp_df)
        helix_residues = len(dssp_df[dssp_df["SimpleSS"] == "H"])
        helix_percentage = (helix_residues / total_residues * 100) if total_residues > 0 else 0
        
        return helix_residues, total_residues, helix_percentage
    
    except Exception as e:
        print(f"DSSP failed: {e}")
        return None, None, None
    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(pdb_path)
        except:
            pass


# ============================================================================
# Custom Output Class with Block Representations
# ============================================================================

@dataclass
class NewEsmForProteinFoldingOutput(ModelOutput):
    """Extended output class that includes per-block representations."""
    frames: Optional[torch.FloatTensor] = None
    sidechain_frames: Optional[torch.FloatTensor] = None
    unnormalized_angles: Optional[torch.FloatTensor] = None
    angles: Optional[torch.FloatTensor] = None
    positions: Optional[torch.FloatTensor] = None
    states: Optional[torch.FloatTensor] = None
    s_s: Optional[torch.FloatTensor] = None
    s_z: Optional[torch.FloatTensor] = None
    distogram_logits: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    aatype: Optional[torch.FloatTensor] = None
    atom14_atom_exists: Optional[torch.FloatTensor] = None
    residx_atom14_to_atom37: Optional[torch.FloatTensor] = None
    residx_atom37_to_atom14: Optional[torch.FloatTensor] = None
    atom37_atom_exists: Optional[torch.FloatTensor] = None
    residue_index: Optional[torch.FloatTensor] = None
    lddt_head: Optional[torch.FloatTensor] = None
    plddt: Optional[torch.FloatTensor] = None
    ptm_logits: Optional[torch.FloatTensor] = None
    ptm: Optional[torch.FloatTensor] = None
    aligned_confidence_probs: Optional[torch.FloatTensor] = None
    predicted_aligned_error: Optional[torch.FloatTensor] = None
    max_predicted_aligned_error: Optional[torch.FloatTensor] = None
    s_s_list: Optional[List[torch.FloatTensor]] = None
    s_z_list: Optional[List[torch.FloatTensor]] = None


# ============================================================================
# Monkey-patched Forward Functions
# ============================================================================

def collect_block_representations(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """
    Modified trunk forward that collects representations from each block.
    """
    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        if no_recycles < 0:
            raise ValueError("Number of recycles must not be negative.")
        no_recycles += 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        s_s_list = []
        s_z_list = []
        for block in self.blocks:
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            s_s_list.append(s)
            s_z_list.append(z)
        return s, z, s_s_list, s_z_list

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

            s_s, s_z, sequence_list, pairwise_list = trunk_iter(
                s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask
            )

            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z
    structure['s_s_list'] = sequence_list
    structure['s_z_list'] = pairwise_list

    return structure


def return_block_representations(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    masking_pattern: Optional[torch.Tensor] = None,
    num_recycles: Optional[int] = None,
    output_hidden_states: Optional[bool] = False,
) -> NewEsmForProteinFoldingOutput:
    """Modified forward that returns block-level representations."""
    cfg = self.config.esmfold_config

    aa = input_ids
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

    if masking_pattern is not None:
        masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
    else:
        masked_aa = aa
        mlm_targets = None

    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype)

    if cfg.esm_ablate_sequence:
        esm_s = esm_s * 0

    esm_s = esm_s.detach()
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = self.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(masked_aa)

    structure: dict = self.trunk(
        s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles
    )
    
    structure = {
        k: v for k, v in structure.items()
        if k in [
            "s_z", "s_s", "frames", "sidechain_frames", "unnormalized_angles",
            "angles", "positions", "states", "s_s_list", "s_z_list"
        ]
    }

    if mlm_targets:
        structure["mlm_targets"] = mlm_targets

    disto_logits = self.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = self.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    
    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = self.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, self.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
    structure["plddt"] = plddt

    ptm_logits = self.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

    return NewEsmForProteinFoldingOutput(**structure)


#Patch block

def ablate_bridges(self, sequence_state, pairwise_state, mask=None, chunk_size=None, ablate_pair_to_seq = False, ablate_seq_to_pair = False, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim mask: B x L boolean
          tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim
        """
        if len(sequence_state.shape) != 3:
            raise ValueError(f"`sequence_state` should be a 3d-tensor, got {len(sequence_state.shape)} dims.")
        if len(pairwise_state.shape) != 4:
            raise ValueError(f"`pairwise_state` should be a 4d-tensor, got {len(pairwise_state.shape)} dims.")
        if mask is not None and len(mask.shape) != 2:
            raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]

        if sequence_state_dim != self.config.sequence_state_dim:
            raise ValueError(
                "`sequence_state` last dimension should be equal to `self.sequence_state_dim`. Got "
                f"{sequence_state_dim} != {self.config.sequence_state_dim}."
            )
        if pairwise_state_dim != self.config.pairwise_state_dim:
            raise ValueError(
                "`pairwise_state` last dimension should be equal to `self.pairwise_state_dim`. Got "
                f"{pairwise_state_dim} != {self.config.pairwise_state_dim}."
            )
        if batch_dim != pairwise_state.shape[0]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent batch size: {batch_dim} != "
                f"{pairwise_state.shape[0]}."
            )
        if seq_dim != pairwise_state.shape[1] or seq_dim != pairwise_state.shape[2]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent sequence length: {seq_dim} != "
                f"{pairwise_state.shape[1]} or {pairwise_state.shape[2]}."
            )

        if ablate_pair_to_seq:
            # Update sequence state
            bias = None
        else:
            bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        if ablate_seq_to_pair:
            pairwise_state = pairwise_state
        else:
            pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(self.tri_mul_out(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.col_drop(self.tri_mul_in(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state
def patch_s_representations_in_trunk(
    self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles,
    donor_s_s_list, donor_s_z_list, target_start, target_end, target_block, patch_mode, donor_hairpin_start, pairwise_mask, ablate_pair_to_seq, ablate_seq_to_pair, ablate_block_indices
):
    """
    Modified trunk forward that patches representations at a specific block.
    
    Args:
        ablate_block_indices: List of block indices where ablation should be applied (can be None or empty)
    """
    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats
    
    # Convert ablate_block_indices to a set for O(1) lookup
    ablate_blocks_set = set(ablate_block_indices) if ablate_block_indices else set()

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        if no_recycles < 0:
            raise ValueError("Number of recycles must not be negative.")
        no_recycles += 1

    def apply_patch(block_idx, s_s, s_z):
        if patch_mode in ('both', 'sequence'):
            donor_block_s_repr = donor_s_s_list[block_idx].to(s_s.device, dtype=s_s.dtype)
            donor_len = donor_block_s_repr.shape[1]
            target_len = target_end - target_start
            assert donor_len == target_len, f"Donor length mismatch: {donor_len} != {target_len}"
            s_s[:, target_start:target_end, :] = donor_block_s_repr
            
        if patch_mode in ('both', 'pairwise'):
            donor_z = donor_s_z_list[block_idx].to(s_z.device, dtype=s_z.dtype)
            mask = pairwise_mask.to(s_z.device)

            #apply mask: for each True position in mask, copy from corresponding donor representation
            for t_i in range(mask.shape[0]):
                for t_j in range(mask.shape[1]):
                    if mask[t_i, t_j]:
                        #map target coords to donor coords via relative position
                        d_i = t_i - target_start + donor_hairpin_start
                        d_j = t_j - target_start + donor_hairpin_start
                        s_z[:, t_i, t_j, :] = donor_z[:, d_i, d_j, :]
    
        return s_s, s_z

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        s_s_list = []
        s_z_list = []
        for block_idx, block in enumerate(self.blocks):
            if block_idx in ablate_blocks_set:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size, ablate_pair_to_seq=ablate_pair_to_seq, ablate_seq_to_pair=ablate_seq_to_pair)
            else:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            if block_idx == target_block:
                s, z = apply_patch(block_idx, s, z)
            s_s_list.append(s)
            s_z_list.append(z)
        return s, z, s_s_list, s_z_list

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

            s_s, s_z, sequence_list, pairwise_list = trunk_iter(
                s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask
            )

            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z
    structure['s_s_list'] = sequence_list
    structure['s_z_list'] = pairwise_list

    return structure

def high_forward_pass(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    masking_pattern: Optional[torch.Tensor] = None,
    num_recycles: Optional[int] = None,
    output_hidden_states: Optional[bool] = False,
    donor_s_s_list=None,
    donor_s_z_list=None,
    target_start=None,
    target_end=None,
    target_block=None,
    patch_mode=None,
    donor_hairpin_start=None,
    pairwise_mask=None,
    ablate_seq_to_pair=None,
    ablate_pair_to_seq=None,
    ablate_block_indices=None,
) -> NewEsmForProteinFoldingOutput:
    """Modified forward that applies patching during inference.
    
    Args:
        ablate_block_indices: List of block indices where ablation should be applied
    """
    cfg = self.config.esmfold_config

    aa = input_ids
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

    if masking_pattern is not None:
        masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
    else:
        masked_aa = aa
        mlm_targets = None

    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype)

    if cfg.esm_ablate_sequence:
        esm_s = esm_s * 0

    esm_s = esm_s.detach()
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = self.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(masked_aa)

    structure: dict = self.trunk(
        s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles,
        donor_s_s_list=donor_s_s_list, donor_s_z_list=donor_s_z_list,
        target_start=target_start, target_end=target_end,
        target_block=target_block, patch_mode=patch_mode,
        donor_hairpin_start=donor_hairpin_start, pairwise_mask=pairwise_mask,
        ablate_pair_to_seq=ablate_pair_to_seq, ablate_seq_to_pair=ablate_seq_to_pair, 
        ablate_block_indices=ablate_block_indices
    )
    
    structure = {
        k: v for k, v in structure.items()
        if k in [
            "s_z", "s_s", "frames", "sidechain_frames", "unnormalized_angles",
            "angles", "positions", "states", "s_s_list", "s_z_list"
        ]
    }

    if mlm_targets:
        structure["mlm_targets"] = mlm_targets

    disto_logits = self.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = self.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    
    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = self.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, self.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
    structure["plddt"] = plddt

    ptm_logits = self.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

    return NewEsmForProteinFoldingOutput(**structure)

# ============================================================================
# Single Block Experiment Runner
# ============================================================================
def create_pairwise_mask(
    donor_hairpin_start: int,
    donor_hairpin_end: int,
    donor_len: int,
    target_start: int,
    target_end: int,
    target_len: int,
    mode: str,
) -> torch.Tensor:
    """Create the pairwise patch mask."""
    patch_mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    
    if mode == "intra":
        patch_mask[target_start:target_end, target_start:target_end] = True
        
    elif mode in ("touch", "hole"):
        # Compute transportable range
        donor_left_extent = donor_hairpin_start
        target_left_extent = target_start
        left_extent = min(donor_left_extent, target_left_extent)
        
        donor_right_extent = donor_len - donor_hairpin_end
        target_right_extent = target_len - target_end
        right_extent = min(donor_right_extent, target_right_extent)
        
        transport_start = target_start - left_extent
        transport_end = target_end + right_extent
        
        # Create cross
        patch_mask[target_start:target_end, transport_start:transport_end] = True
        patch_mask[transport_start:transport_end, target_start:target_end] = True
        
        if mode == "hole":
            # Cut out intra-hairpin region
            patch_mask[target_start:target_end, target_start:target_end] = False
    
    return patch_mask


def visualize_pairwise_mask(
    patch_mask: torch.Tensor,
    target_start: int,
    target_end: int,
    mode: str,
    save_path: str,
    figsize: Tuple[int, int] = (6, 5),
):
    """Save visualization of pairwise mask."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(patch_mask.numpy(), cmap='Greys', vmin=0, vmax=1, origin='upper')
    ax.set_xlabel('Residue Position (Target)')
    ax.set_ylabel('Residue Position (Target)')
    ax.set_title(f'Pairwise Patch Mask ({mode})\nHairpin region: [{target_start}:{target_end}]')
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((target_start - 0.5, target_start - 0.5),
                      target_end - target_start,
                      target_end - target_start,
                      linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved patch visualization to {save_path}")

def run_single_block_experiment_fast(
    model,
    tokenizer,
    device,
    target_seq: str,
    donor_s_blocks: List[torch.Tensor],
    donor_z_blocks: List[torch.Tensor],
    donor_hairpin_start: int,
    target_start: int,
    target_end: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    patch_mask_mode: str,
    num_blocks: int,
    save_pdbs: bool = False,
    pdb_dir: Optional[str] = None,
    case_id: Optional[str] = None,
    compute_helix: bool = False,
    orig_helix_data: Tuple = (None, None, None),
) -> List[Dict[str, Any]]:
    """
    Run patching experiment - assumes donor representations are already extracted.
    """
    results = []
    orig_helix_count, orig_total, orig_helix_pct = orig_helix_data
    
    for block_idx in range(num_blocks):
        model.forward = types.MethodType(high_forward_pass, model)
        model.trunk.forward = types.MethodType(patch_s_representations_in_trunk, model.trunk)
        
        with torch.no_grad():
            target_inputs = tokenizer(
                target_seq, return_tensors='pt', add_special_tokens=False
            ).to(device)
            
            patched_outputs = model(
                **target_inputs,
                num_recycles=0,
                donor_s_s_list=donor_s_blocks,
                donor_s_z_list=donor_z_blocks,
                target_start=target_start,
                target_end=target_end,
                target_block=block_idx,
                patch_mode=patch_mode,
                donor_hairpin_start=donor_hairpin_start,
                pairwise_mask=pairwise_mask,
            )
        
        # Clean up outputs
        patched_outputs_dict = dict(patched_outputs)
        patched_outputs_dict.pop("s_s_list", None)
        patched_outputs_dict.pop("s_z_list", None)
        clean_outputs = EsmForProteinFoldingOutput(**patched_outputs_dict)
        
        # Check for hairpin using detect_hairpins
        hairpin_found, _ = detect_hairpins(clean_outputs, model)
        
        result = {
            "block_idx": block_idx,
            "patch_mode": patch_mode,
            "patch_mask_mode": patch_mask_mode,
            "hairpin_found": hairpin_found,
        }
        
        # Helix analysis
        if save_pdbs or compute_helix:
            pdb_string = model.output_to_pdb(clean_outputs)[0]
            
            if compute_helix:
                patched_helix_count, patched_total, patched_helix_pct = compute_alpha_helix_content(pdb_string)
                
                if orig_helix_pct is not None and patched_helix_pct is not None:
                    helix_absolute_change = patched_helix_pct - orig_helix_pct
                    helix_relative_change = ((patched_helix_pct - orig_helix_pct) / orig_helix_pct * 100) if orig_helix_pct > 0 else None
                else:
                    helix_absolute_change = None
                    helix_relative_change = None
                
                result.update({
                    "original_helix_count": orig_helix_count,
                    "original_total_residues": orig_total,
                    "original_helix_pct": orig_helix_pct,
                    "patched_helix_count": patched_helix_count,
                    "patched_total_residues": patched_total,
                    "patched_helix_pct": patched_helix_pct,
                    "helix_absolute_change": helix_absolute_change,
                    "helix_relative_change": helix_relative_change,
                })
            
            if save_pdbs and pdb_dir:
                pdb_filename = f"{case_id}_block{block_idx}_{patch_mode}_{patch_mask_mode}.pdb"
                pdb_path = os.path.join(pdb_dir, pdb_filename)
                with open(pdb_path, 'w') as f:
                    f.write(pdb_string)
        
        results.append(result)
        torch.cuda.empty_cache()
    
    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================

def parse_patch_region(patch_region_str: str) -> Tuple[int, int]:
    """Parse target_patch_region string like '(11, 27)' to tuple."""
    import ast
    return ast.literal_eval(patch_region_str)


def run_single_ablation_experiment(
    model,
    tokenizer,
    device,
    target_seq: str,
    donor_s_blocks: List[torch.Tensor],
    donor_z_blocks: List[torch.Tensor],
    donor_hairpin_start: int,
    target_start: int,
    target_end: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    patch_mask_mode: str,
    patch_block_idx: int,
    num_blocks: int,
    case_id: Optional[str] = None,
    window_sizes: List[int] = [3, 5, 10, 15],
) -> List[Dict[str, Any]]:
    """
    Run sliding window ablation experiment for a single successful patching case.
    
    For a given patch applied at patch_block_idx, test ablating contiguous windows
    of bridges starting at the patch block and sliding forward.
    
    Args:
        window_sizes: List of window sizes to test (default: [3, 5, 10, 15])
    """
    results = []
    
    ablation_types = [
        ("pair2seq", True, False),
        ("seq2pair", False, True),
        ("both", True, True),
    ]
    
    # Calculate total iterations for this case
    total_iters = 0
    for window_size in window_sizes:
        max_start = num_blocks - window_size
        n_windows = max(0, max_start - patch_block_idx + 1)
        total_iters += n_windows * len(ablation_types)
    
    # Inner progress bar
    pbar = tqdm(total=total_iters, desc="  Windows", position=1, leave=False)
    
    for window_size in window_sizes:
        # Sliding window starts at patch_block_idx and moves forward
        # Window can start at patch_block_idx up to (num_blocks - window_size)
        max_start = num_blocks - window_size
        
        for window_start in range(patch_block_idx, max_start + 1):
            window_end = window_start + window_size  # exclusive
            ablate_block_indices = list(range(window_start, window_end))
            
            for ablation_name, ablate_pair_to_seq, ablate_seq_to_pair in ablation_types:
                model.forward = types.MethodType(high_forward_pass, model)
                model.trunk.forward = types.MethodType(patch_s_representations_in_trunk, model.trunk)
                
                with torch.no_grad():
                    target_inputs = tokenizer(
                        target_seq, return_tensors='pt', add_special_tokens=False
                    ).to(device)
                    
                    patched_outputs = model(
                        **target_inputs,
                        num_recycles=0,
                        donor_s_s_list=donor_s_blocks,
                        donor_s_z_list=donor_z_blocks,
                        target_start=target_start,
                        target_end=target_end,
                        target_block=patch_block_idx,
                        patch_mode=patch_mode,
                        donor_hairpin_start=donor_hairpin_start,
                        pairwise_mask=pairwise_mask,
                        ablate_pair_to_seq=ablate_pair_to_seq,
                        ablate_seq_to_pair=ablate_seq_to_pair,
                        ablate_block_indices=ablate_block_indices,
                    )
                
                # Clean up outputs
                patched_outputs_dict = dict(patched_outputs)
                patched_outputs_dict.pop("s_s_list", None)
                patched_outputs_dict.pop("s_z_list", None)
                clean_outputs = EsmForProteinFoldingOutput(**patched_outputs_dict)
                
                # Check for hairpin using detect_hairpins
                hairpin_found, _ = detect_hairpins(clean_outputs, model)
                
                result = {
                    "patch_block_idx": patch_block_idx,
                    "window_size": window_size,
                    "window_start": window_start,
                    "window_end": window_end,
                    "ablate_block_indices": str(ablate_block_indices),
                    "ablation_type": ablation_name,
                    "ablate_pair_to_seq": ablate_pair_to_seq,
                    "ablate_seq_to_pair": ablate_seq_to_pair,
                    "hairpin_found": hairpin_found,
                }
                
                results.append(result)
                pbar.update(1)
                torch.cuda.empty_cache()
    
    pbar.close()
    return results


def run_ablation_experiment(
    results_csv_path: str,
    n_sequence_cases: int,
    n_pairwise_cases: int,
    output_dir: str,
    device: Optional[str] = None,
    cache_flush_interval: int = 20,
    window_sizes: List[int] = [3, 5, 10, 15],
) -> pd.DataFrame:
    """
    Run sliding window ablation experiments on successful patching cases.
    
    Args:
        results_csv_path: Path to block_patching_successes.csv with successful cases
        n_sequence_cases: Number of sequence patching cases to run
        n_pairwise_cases: Number of pairwise patching cases to run
        output_dir: Directory for outputs
        device: Device to use (auto-detected if None)
        cache_flush_interval: Flush donor cache and save results every N cases (default: 20)
        window_sizes: List of window sizes to test (default: [3, 5, 10, 15])
        
    Returns:
        DataFrame with ablation results
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.eval()
    
    # Monkey-patch the block forward to support ablation
    for block in model.trunk.blocks:
        block.forward = types.MethodType(ablate_bridges, block)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("Model loaded successfully")
    
    # Load results and filter to successful cases
    print(f"Loading results from {results_csv_path}...")
    df = pd.read_csv(results_csv_path)
    
    # Filter to only successful hairpin formations
    successful_df = df[df['hairpin_found'] == True].copy()
    print(f"Found {len(successful_df)} successful cases out of {len(df)} total")
    
    # Separate by patch mode and take specified number from each
    sequence_cases = successful_df[successful_df['patch_mode'] == 'sequence'].head(n_sequence_cases)
    pairwise_cases = successful_df[successful_df['patch_mode'] == 'pairwise'].head(n_pairwise_cases)
    
    print(f"Selected {len(sequence_cases)} sequence patching cases (requested: {n_sequence_cases})")
    print(f"Selected {len(pairwise_cases)} pairwise patching cases (requested: {n_pairwise_cases})")
    
    cases_to_run = pd.concat([sequence_cases, pairwise_cases], ignore_index=True)
    print(f"Total cases to run: {len(cases_to_run)}")
    
    # Parent columns to preserve (updated for new CSV format)
    parent_columns = [
        "case_idx", "target_name", "target_sequence", "target_length",
        "loop_idx", "loop_start", "loop_end", "loop_length", "loop_sequence",
        "target_patch_start", "target_patch_end", "patch_length",
        "donor_pdb", "donor_sequence", "donor_length",
        "donor_hairpin_start", "donor_hairpin_end", "donor_hairpin_length", "donor_hairpin_sequence",
        "patch_mode", "patch_mask_mode", "block_idx",
    ]
    
    all_results = []
    
    # Cache for donor representations (keyed by donor_sequence)
    donor_cache = {}
    cases_processed = 0
    
    # Paths for saving
    results_path = os.path.join(output_dir, "ablation_results.csv")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Main progress bar
    pbar = tqdm(total=len(cases_to_run), desc="Cases", position=0)
    
    skipped_cases = []
    
    for row_idx, row in cases_to_run.iterrows():
        case_id = f"ablate_case_{row_idx}"
        try:
        
            # Extract info from parent row (updated for new CSV format)
            target_seq = row['target_sequence']
            donor_seq = row['donor_sequence']
            patch_mode = row['patch_mode']
            patch_mask_mode = row['patch_mask_mode']
            patch_block_idx = int(row['block_idx'])
        
            # Get patch region from CSV
            target_start = int(row['target_patch_start'])
            target_end = int(row['target_patch_end'])
        
            # Get donor hairpin locations from CSV
            donor_hairpin_start = int(row['donor_hairpin_start'])
            donor_hairpin_end = int(row['donor_hairpin_end'])
        
            # Get parent metadata
            parent_meta = {col: row[col] for col in parent_columns if col in row.index}
        
            # Update progress bar description
            pbar.set_description(f"[{patch_mode[:3]}] {row['target_name'][:10]}<-{row['donor_pdb'][:6]}")
        
            # Get or compute donor representations
            if donor_seq not in donor_cache:
                model.forward = types.MethodType(return_block_representations, model)
                model.trunk.forward = types.MethodType(collect_block_representations, model.trunk)
            
                with torch.no_grad():
                    donor_inputs = tokenizer(donor_seq, return_tensors='pt', add_special_tokens=False).to(device)
                    donor_outputs = model(**donor_inputs, num_recycles=0)
            
                # Extract and cache
                donor_s_blocks = []
                for block_repr in donor_outputs.s_s_list:
                    donor_s_blocks.append(
                        block_repr[:, donor_hairpin_start:donor_hairpin_end, :].detach().cpu()
                    )
            
                donor_z_blocks = []
                for block_repr in donor_outputs.s_z_list:
                    donor_z_blocks.append(block_repr.detach().cpu())
            
                donor_cache[donor_seq] = {
                    'donor_s_blocks': donor_s_blocks,
                    'donor_z_blocks': donor_z_blocks,
                }
            
                del donor_outputs
                torch.cuda.empty_cache()
            else:
                cached = donor_cache[donor_seq]
                donor_s_blocks = cached['donor_s_blocks']
                donor_z_blocks = cached['donor_z_blocks']

            num_blocks = len(donor_s_blocks)
        
            # Create pairwise mask
            pairwise_mask = create_pairwise_mask(
                donor_hairpin_start=donor_hairpin_start,
                donor_hairpin_end=donor_hairpin_end,
                donor_len=len(donor_seq),
                target_start=target_start,
                target_end=target_end,
                target_len=len(target_seq),
                mode=patch_mask_mode,
            )
        
            # Run ablation experiment
            results = run_single_ablation_experiment(
                model=model,
                tokenizer=tokenizer,
                device=device,
                target_seq=target_seq,
                donor_s_blocks=donor_s_blocks,
                donor_z_blocks=donor_z_blocks,
                donor_hairpin_start=donor_hairpin_start,
                target_start=target_start,
                target_end=target_end,
                pairwise_mask=pairwise_mask,
                patch_mode=patch_mode,
                patch_mask_mode=patch_mask_mode,
                patch_block_idx=patch_block_idx,
                num_blocks=num_blocks,
                case_id=case_id,
                window_sizes=window_sizes,
            )
        
            # Add parent metadata to each result
            for r in results:
                r.update(parent_meta)
        
            all_results.extend(results)
            cases_processed += 1
            pbar.update(1)
        
            # Periodically save results, generate plots, and flush cache
            if cases_processed % cache_flush_interval == 0:
                pbar.write(f"\n{'='*60}")
                pbar.write(f"Checkpoint at {cases_processed} cases")
                pbar.write(f"{'='*60}")
            
                # Save current results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(results_path, index=False)
                pbar.write(f"Saved {len(results_df)} results to {results_path}")
            
                # Generate plots
                try:
                    from sliding_window_plotting import generate_ablation_plots, print_ablation_summary
                    print_ablation_summary(results_df)
                    generate_ablation_plots(results_df, plots_dir)
                except Exception as e:
                    pbar.write(f"Warning: Plot generation failed: {e}")
            
                # Flush donor cache
                pbar.write(f"Flushing donor cache ({len(donor_cache)} entries)")
                donor_cache.clear()
                torch.cuda.empty_cache()
            
                pbar.write(f"{'='*60}\n")
        except Exception as e:
            pbar.write(f"WARNING: Skipping case {row_idx} ({case_id}): {e}")
            skipped_cases.append({'row_idx': row_idx, 'error': str(e)})
            pbar.update(1)
            continue
    
    if skipped_cases:
        print(f"\nSkipped {len(skipped_cases)} cases due to errors:")
        for sc in skipped_cases:
            print(f"  Row {sc['row_idx']}: {sc['error']}")
    
    pbar.close()
    
    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    print(f"\nFinal results saved to {results_path}")
    
    # Final plots
    try:
        from sliding_window_plotting import generate_ablation_plots, print_ablation_summary
        print_ablation_summary(results_df)
        generate_ablation_plots(results_df, plots_dir)
    except Exception as e:
        print(f"Warning: Final plot generation failed: {e}")
    
    return results_df


# Add to main entry point
def main():
    parser = argparse.ArgumentParser(
        description="Run ESMFold block patching experiments"
    )
    parser.add_argument(
        "--ablation_csv", type=str, default='data/single_block_patching_successes.csv',
        help="Path to successful_cases.csv"
    )
    parser.add_argument(
        "--patch_modes",
        nargs="+",
        default=["sequence", "pairwise"],
        help="Patch modes"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sliding_window_ablation",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--save_pdbs", action="store_true",
        help="Save PDB structures"
    )
    parser.add_argument(
        "--compute_helix", action="store_true",
        help="Compute alpha helix content (requires DSSP via trunk_utils)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--skip_experiments", action="store_true",
        help="Skip experiments and only generate plots from existing results"
    )
    parser.add_argument(
        "--n_sequence_cases", type=int, default=400,
        help="Number of sequence patching cases to run (default: 400)"
    )
    parser.add_argument(
        "--n_pairwise_cases", type=int, default=200,
        help="Number of pairwise patching cases to run (default: 200)"
    )
    parser.add_argument(
        "--cache_flush_interval", type=int, default=20,
        help="Flush donor cache every N cases to prevent memory buildup (default: 20)"
    )
    parser.add_argument(
        "--window_sizes", type=int, nargs="+", default=[15],
        help="Window sizes for sliding window ablation (default: 3 5 10 15)"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # ABLATION EXPERIMENT
    # =========================================================================
    
    ablation_csv = args.ablation_csv or os.path.join(args.output_dir, "block_patching_successes.csv")
    if not os.path.exists(ablation_csv):
        print(f"Error: No results found at {ablation_csv}")
        return
    
    ablation_output_dir = os.path.join(args.output_dir, "ablation")
    
    results_df = run_ablation_experiment(
        results_csv_path=ablation_csv,
        n_sequence_cases=args.n_sequence_cases,
        n_pairwise_cases=args.n_pairwise_cases,
        output_dir=ablation_output_dir,
        device=args.device,
        cache_flush_interval=args.cache_flush_interval,
        window_sizes=args.window_sizes,
    )
    
    print(f"\nAblation experiment complete!")
    print(f"Results: {os.path.join(ablation_output_dir, 'ablation_results.csv')}")
    print(f"Plots: {os.path.join(ablation_output_dir, 'plots')}")
    return


if __name__ == "__main__":
    main()