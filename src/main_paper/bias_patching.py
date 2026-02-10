"""
Pair-to-Sequence Attention Bias Patching
========================================

Analyzes how the pairwise representation (z) influences sequence attention via
the pair_to_sequence bias pathway in ESMFold's transformer blocks.

This experiment tests whether the attention biases computed from z correlate with
the donor's contact map after patching, providing evidence that z encodes
structural contact information that propagates to sequence attention.

Usage:
    python bias_patching.py --csv data.csv --output results/ --n_cases 5
"""

import argparse
import os
import types
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import EsmFoldingTrunk
from transformers.utils import ContextManagers

from src.utils.representation_utils import CollectedRepresentations, TrunkHooks


# ============================================================================
# PART 1: ATTENTION DATA CONTAINER
# ============================================================================

@dataclass
class AttentionCollector:
    """Container for collected attention data."""
    bias: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_with_bias: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_without_bias: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        self.bias.clear()
        self.attn_with_bias.clear()
        self.attn_without_bias.clear()


# ============================================================================
# PART 2: ATTENTION HOOKS
# ============================================================================

class AttentionHooks:
    """
    Collect attention weights and biases from trunk blocks.
    
    Usage:
        collector = AttentionCollector()
        hooks = AttentionHooks(model.trunk, collector, blocks=[27, 30, 35])
        hooks.register()
        outputs = model(**inputs)
        hooks.remove()
        # collector.bias, collector.attn_with_bias now populated
    """
    
    def __init__(self, trunk: nn.Module, collector: AttentionCollector):
        self.trunk = trunk
        self.collector = collector
        self.handles: List = []
        self.blocks_to_collect: List[int] = []
    
    def _compute_attention(
        self,
        seq_attention_module,
        x: torch.Tensor,
        bias: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights with and without bias."""
        module = seq_attention_module
        
        t = module.proj(x).view(*x.shape[:2], module.num_heads, -1)
        t = t.permute(0, 2, 1, 3)
        q, k, v = t.chunk(3, dim=-1)
        q = module.rescale_factor * q
        
        raw_scores = torch.einsum("...qc,...kc->...qk", q, k)
        bias_permuted = bias.permute(0, 3, 1, 2)
        
        if mask is not None:
            mask_2d = mask[:, None, None, :]
            raw_scores = raw_scores.masked_fill(mask_2d == False, -np.inf)
        
        attn_without_bias = F.softmax(raw_scores, dim=-1)
        scores_with_bias = raw_scores + bias_permuted
        attn_with_bias = F.softmax(scores_with_bias, dim=-1)
        
        # Permute back to [B, L, L, H]
        attn_with_bias = attn_with_bias.permute(0, 2, 3, 1)
        attn_without_bias = attn_without_bias.permute(0, 2, 3, 1)
        
        return attn_with_bias, attn_without_bias
    
    def register(self, blocks: List[int]):
        """Register hooks on specified blocks."""
        self.blocks_to_collect = blocks
        
        for block_idx in blocks:
            if block_idx >= len(self.trunk.blocks):
                continue
            
            block = self.trunk.blocks[block_idx]
            
            # Hook on pair_to_sequence to capture bias
            def make_bias_hook(b_idx):
                def hook(module, inputs, outputs):
                    # outputs is the bias tensor
                    self.collector.bias[b_idx] = outputs.detach().cpu()
                return hook
            
            handle = block.pair_to_sequence.register_forward_hook(make_bias_hook(block_idx))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
# PART 3: PAIRWISE MASK CREATION
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
# PART 4: COLLECTION FUNCTIONS
# ============================================================================

def run_and_collect_attention(
    model,
    tokenizer,
    device: str,
    sequence: str,
    analysis_blocks: List[int],
) -> Tuple[Any, AttentionCollector, torch.Tensor, torch.Tensor]:
    """
    Run model and collect attention at specified blocks.
    
    Returns:
        outputs: Model outputs
        collector: AttentionCollector with bias data
        contact_map: Binary contact map
        distances: Distance matrix
    """
    collector = AttentionCollector()
    
    # We need a custom forward to compute attention from bias
    # Since attention computation requires the sequence representation,
    # we'll use a modified trunk forward
    
    attn_data = {'bias': {}, 'attn_with_bias': {}, 'attn_without_bias': {}}
    
    def make_attention_collecting_forward(blocks_to_collect):
        def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
            device = seq_feats.device
            s_s_0, s_z_0 = seq_feats, pair_feats

            if no_recycles is None:
                no_recycles = self.config.max_recycles
            else:
                no_recycles += 1

            def compute_attention(seq_attention_module, x, bias, attn_mask=None):
                module = seq_attention_module
                t = module.proj(x).view(*x.shape[:2], module.num_heads, -1)
                t = t.permute(0, 2, 1, 3)
                q, k, v = t.chunk(3, dim=-1)
                q = module.rescale_factor * q
                
                raw_scores = torch.einsum("...qc,...kc->...qk", q, k)
                bias_permuted = bias.permute(0, 3, 1, 2)
                
                if attn_mask is not None:
                    mask_2d = attn_mask[:, None, None, :]
                    raw_scores = raw_scores.masked_fill(mask_2d == False, -np.inf)
                
                attn_without = F.softmax(raw_scores, dim=-1)
                attn_with = F.softmax(raw_scores + bias_permuted, dim=-1)
                
                return attn_with.permute(0, 2, 3, 1), attn_without.permute(0, 2, 3, 1)

            def trunk_iter(s, z, residx, mask):
                z = z + self.pairwise_positional_embedding(residx, mask=mask)
                
                for block_idx, block in enumerate(self.blocks):
                    if block_idx in blocks_to_collect:
                        bias = block.pair_to_sequence(z)
                        attn_data['bias'][block_idx] = bias.detach().cpu()
                        
                        seq_normed = block.layernorm_1(s)
                        attn_with, attn_without = compute_attention(
                            block.seq_attention, seq_normed, bias, mask
                        )
                        attn_data['attn_with_bias'][block_idx] = attn_with.detach().cpu()
                        attn_data['attn_without_bias'][block_idx] = attn_without.detach().cpu()
                    
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
                        structure["positions"][-1][:, :, :3],
                        3.375, 21.375, self.recycle_bins,
                    )

            structure["s_s"] = s_s
            structure["s_z"] = s_z
            return structure
        
        return forward
    
    original_forward = model.trunk.forward
    model.trunk.forward = types.MethodType(
        make_attention_collecting_forward(analysis_blocks), 
        model.trunk
    )
    
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)
    finally:
        model.trunk.forward = original_forward
    
    # Compute contact map from structure
    ca_positions = outputs.positions[-1, 0, :, 1, :]  # [L, 3] CA atoms
    diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1))
    contact_map = (distances < 8.0).float()
    
    # Fill collector
    collector.bias = attn_data['bias']
    collector.attn_with_bias = attn_data['attn_with_bias']
    collector.attn_without_bias = attn_data['attn_without_bias']
    
    return outputs, collector, contact_map.cpu(), distances.cpu()


# ============================================================================
# PART 5: PATCHING WITH ATTENTION COLLECTION
# ============================================================================

def make_pairwise_patch_attention_forward(
    donor_z: torch.Tensor,
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_block: int,
    analysis_blocks: List[int],
    attn_data: Dict,
):
    """
    Create trunk forward that patches pairwise and collects attention.
    """
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            no_recycles += 1

        def compute_attention(seq_attention_module, x, bias, attn_mask=None):
            module = seq_attention_module
            t = module.proj(x).view(*x.shape[:2], module.num_heads, -1)
            t = t.permute(0, 2, 1, 3)
            q, k, v = t.chunk(3, dim=-1)
            q = module.rescale_factor * q
            
            raw_scores = torch.einsum("...qc,...kc->...qk", q, k)
            bias_permuted = bias.permute(0, 3, 1, 2)
            
            if attn_mask is not None:
                mask_2d = attn_mask[:, None, None, :]
                raw_scores = raw_scores.masked_fill(mask_2d == False, -np.inf)
            
            attn_without = F.softmax(raw_scores, dim=-1)
            attn_with = F.softmax(raw_scores + bias_permuted, dim=-1)
            
            return attn_with.permute(0, 2, 3, 1), attn_without.permute(0, 2, 3, 1)

        def apply_pairwise_patch(z):
            """Apply donor pairwise to target using mask."""
            donor = donor_z.to(z.device, dtype=z.dtype)
            mask_dev = pairwise_mask.to(z.device)
            
            indices = torch.where(mask_dev)
            for i in range(len(indices[0])):
                ti, tj = indices[0][i].item(), indices[1][i].item()
                di = ti - target_start + donor_start
                dj = tj - target_start + donor_start
                if 0 <= di < donor.shape[1] and 0 <= dj < donor.shape[2]:
                    z[:, ti, tj, :] = donor[:, di, dj, :]
            return z

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            
            for block_idx, block in enumerate(self.blocks):
                # Apply patch at patch_block
                if block_idx == patch_block:
                    z = apply_pairwise_patch(z)
                
                # Collect attention at analysis blocks
                if block_idx in analysis_blocks:
                    bias = block.pair_to_sequence(z)
                    attn_data['bias'][block_idx] = bias.detach().cpu()
                    
                    seq_normed = block.layernorm_1(s)
                    attn_with, attn_without = compute_attention(
                        block.seq_attention, seq_normed, bias, mask
                    )
                    attn_data['attn_with_bias'][block_idx] = attn_with.detach().cpu()
                    attn_data['attn_without_bias'][block_idx] = attn_without.detach().cpu()
                
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
                    structure["positions"][-1][:, :, :3],
                    3.375, 21.375, self.recycle_bins,
                )

        structure["s_s"] = s_s
        structure["s_z"] = s_z
        return structure
    
    return forward


@contextmanager
def patch_pairwise_and_collect_attention(
    model,
    donor_z: torch.Tensor,
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_block: int,
    analysis_blocks: List[int],
    attn_data: Dict,
):
    """
    Context manager for pairwise patching with attention collection.
    """
    original = model.trunk.forward
    
    patched_forward = make_pairwise_patch_attention_forward(
        donor_z, target_start, target_end, donor_start,
        pairwise_mask, patch_block, analysis_blocks, attn_data,
    )
    model.trunk.forward = types.MethodType(patched_forward, model.trunk)
    
    try:
        yield
    finally:
        model.trunk.forward = original


def get_donor_z_at_block(
    model,
    tokenizer,
    device: str,
    donor_sequence: str,
    patch_block: int,
) -> torch.Tensor:
    """
    Run donor and capture pairwise representation at patch_block.
    """
    donor_z = [None]
    
    def make_capture_forward(target_block):
        def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
            device = seq_feats.device
            s_s_0, s_z_0 = seq_feats, pair_feats

            if no_recycles is None:
                no_recycles = self.config.max_recycles
            else:
                no_recycles += 1

            def trunk_iter(s, z, residx, mask):
                z = z + self.pairwise_positional_embedding(residx, mask=mask)
                
                for block_idx, block in enumerate(self.blocks):
                    if block_idx == target_block:
                        donor_z[0] = z.detach().cpu()
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
                        structure["positions"][-1][:, :, :3],
                        3.375, 21.375, self.recycle_bins,
                    )

            structure["s_s"] = s_s
            structure["s_z"] = s_z
            return structure
        
        return forward
    
    original = model.trunk.forward
    model.trunk.forward = types.MethodType(make_capture_forward(patch_block), model.trunk)
    
    try:
        with torch.no_grad():
            inputs = tokenizer(donor_sequence, return_tensors='pt', add_special_tokens=False).to(device)
            _ = model(**inputs, num_recycles=0)
    finally:
        model.trunk.forward = original
    
    return donor_z[0]


def run_patched_and_collect_attention(
    model,
    tokenizer,
    device: str,
    target_sequence: str,
    donor_sequence: str,
    target_region: Tuple[int, int],
    donor_region: Tuple[int, int],
    patch_block: int,
    analysis_blocks: List[int],
    patch_mask_mode: str = 'touch',
) -> Tuple[Any, AttentionCollector, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run target with donor pairwise patched in, collecting attention.
    """
    t_start, t_end = target_region
    d_start, d_end = donor_region
    
    # Get donor z at patch block
    donor_z = get_donor_z_at_block(model, tokenizer, device, donor_sequence, patch_block)
    
    # Create pairwise mask
    pairwise_mask = create_pairwise_mask(
        donor_start=d_start,
        donor_end=d_end,
        donor_len=len(donor_sequence),
        target_start=t_start,
        target_end=t_end,
        target_len=len(target_sequence),
        mode=patch_mask_mode,
    )
    
    attn_data = {'bias': {}, 'attn_with_bias': {}, 'attn_without_bias': {}}
    
    with patch_pairwise_and_collect_attention(
        model, donor_z, t_start, t_end, d_start,
        pairwise_mask, patch_block, analysis_blocks, attn_data,
    ):
        with torch.no_grad():
            inputs = tokenizer(target_sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)
    
    # Compute contact map
    ca_positions = outputs.positions[-1, 0, :, 1, :]
    diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1))
    contact_map = (distances < 8.0).float()
    
    collector = AttentionCollector()
    collector.bias = attn_data['bias']
    collector.attn_with_bias = attn_data['attn_with_bias']
    collector.attn_without_bias = attn_data['attn_without_bias']
    
    return outputs, collector, contact_map.cpu(), distances.cpu(), pairwise_mask


# ============================================================================
# PART 6: ANALYSIS FUNCTIONS
# ============================================================================

def compute_attention_to_contacts(
    attn: torch.Tensor,
    contact_map: torch.Tensor,
    region_start: int,
    region_end: int,
) -> Dict:
    """Compute how much attention goes to contact vs non-contact positions."""
    attn = attn[0]  # Remove batch
    region_len = region_end - region_start
    
    region_attn = attn[region_start:region_end, region_start:region_end, :].mean(dim=-1)
    region_contacts = contact_map[region_start:region_end, region_start:region_end]
    
    mask = ~torch.eye(region_len, dtype=torch.bool)
    attn_flat = region_attn[mask]
    contacts_flat = region_contacts[mask]
    
    contact_mask = contacts_flat > 0.5
    noncontact_mask = ~contact_mask
    
    results = {}
    results['mean_attn_to_contacts'] = attn_flat[contact_mask].mean().item() if contact_mask.sum() > 0 else 0.0
    results['mean_attn_to_noncontacts'] = attn_flat[noncontact_mask].mean().item() if noncontact_mask.sum() > 0 else 0.0
    results['attn_contact_ratio'] = results['mean_attn_to_contacts'] / (results['mean_attn_to_noncontacts'] + 1e-8)
    
    if contact_mask.sum() > 0 and noncontact_mask.sum() > 0:
        try:
            results['attn_contact_auc'] = roc_auc_score(contacts_flat.numpy(), attn_flat.numpy())
        except:
            results['attn_contact_auc'] = None
    
    return results


def compute_attention_alignment_with_foreign_contacts(
    attn: torch.Tensor,
    own_contacts: torch.Tensor,
    foreign_contacts: torch.Tensor,
    region_start: int,
    region_end: int,
    baseline_attn: Optional[torch.Tensor] = None,
) -> Dict:
    """Compute how attention aligns with own vs foreign (donor/target) contacts."""
    attn = attn[0]
    region_len = region_end - region_start
    
    region_attn = attn[region_start:region_end, region_start:region_end, :].mean(dim=-1)
    
    own_region = own_contacts[:region_len, :region_len]
    foreign_region = foreign_contacts[:region_len, :region_len]
    
    mask = ~torch.eye(region_len, dtype=torch.bool)
    attn_flat = region_attn[mask]
    own_flat = own_region[mask]
    foreign_flat = foreign_region[mask]
    
    results = {}
    
    own_contact_mask = own_flat > 0.5
    foreign_contact_mask = foreign_flat > 0.5
    unique_foreign = foreign_contact_mask & ~own_contact_mask
    unique_own = own_contact_mask & ~foreign_contact_mask
    
    results['mean_attn_to_own_contacts'] = attn_flat[own_contact_mask].mean().item() if own_contact_mask.sum() > 0 else 0.0
    results['mean_attn_to_foreign_contacts'] = attn_flat[foreign_contact_mask].mean().item() if foreign_contact_mask.sum() > 0 else 0.0
    results['mean_attn_to_unique_foreign'] = attn_flat[unique_foreign].mean().item() if unique_foreign.sum() > 0 else 0.0
    results['mean_attn_to_unique_own'] = attn_flat[unique_own].mean().item() if unique_own.sum() > 0 else 0.0
    
    # AUCs
    if foreign_contact_mask.sum() > 0 and (~foreign_contact_mask).sum() > 0:
        try:
            results['attn_foreign_auc'] = roc_auc_score(foreign_flat.numpy(), attn_flat.numpy())
        except:
            results['attn_foreign_auc'] = None
    
    if own_contact_mask.sum() > 0 and (~own_contact_mask).sum() > 0:
        try:
            results['attn_own_auc'] = roc_auc_score(own_flat.numpy(), attn_flat.numpy())
        except:
            results['attn_own_auc'] = None
    
    # Percent changes if baseline provided
    if baseline_attn is not None:
        baseline = baseline_attn[0]
        baseline_region = baseline[region_start:region_end, region_start:region_end, :].mean(dim=-1)
        baseline_flat = baseline_region[mask]
        
        if foreign_contact_mask.sum() > 0:
            baseline_to_foreign = baseline_flat[foreign_contact_mask].mean().item()
            results['baseline_attn_to_donor_contacts'] = baseline_to_foreign
            if baseline_to_foreign > 1e-8:
                results['pct_change_attn_to_donor_contacts'] = (
                    (results['mean_attn_to_foreign_contacts'] - baseline_to_foreign) / baseline_to_foreign * 100
                )
        
        if own_contact_mask.sum() > 0:
            baseline_to_own = baseline_flat[own_contact_mask].mean().item()
            results['baseline_attn_to_target_contacts'] = baseline_to_own
            if baseline_to_own > 1e-8:
                results['pct_change_attn_to_target_contacts'] = (
                    (results['mean_attn_to_own_contacts'] - baseline_to_own) / baseline_to_own * 100
                )
        
        if unique_foreign.sum() > 0:
            baseline_unique_foreign = baseline_flat[unique_foreign].mean().item()
            results['baseline_attn_to_unique_donor'] = baseline_unique_foreign
            if baseline_unique_foreign > 1e-8:
                results['pct_change_attn_to_unique_donor'] = (
                    (results['mean_attn_to_unique_foreign'] - baseline_unique_foreign) / baseline_unique_foreign * 100
                )
        
        if unique_own.sum() > 0:
            baseline_unique_own = baseline_flat[unique_own].mean().item()
            results['baseline_attn_to_unique_target'] = baseline_unique_own
            if baseline_unique_own > 1e-8:
                results['pct_change_attn_to_unique_target'] = (
                    (results['mean_attn_to_unique_own'] - baseline_unique_own) / baseline_unique_own * 100
                )
    
    return results


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def visualize_three_way_comparison(
    donor_attn: torch.Tensor,
    target_attn: torch.Tensor,
    patched_attn: torch.Tensor,
    donor_contacts: torch.Tensor,
    target_contacts: torch.Tensor,
    donor_region: Tuple[int, int],
    target_region: Tuple[int, int],
    block_idx: int,
    output_path: str,
    case_name: str = "",
):
    """Visualize attention patterns for donor, target, and patched runs."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    d_start, d_end = donor_region
    t_start, t_end = target_region
    
    donor_region_attn = donor_attn[0, d_start:d_end, d_start:d_end, :].mean(dim=-1).numpy()
    target_region_attn = target_attn[0, t_start:t_end, t_start:t_end, :].mean(dim=-1).numpy()
    patched_region_attn = patched_attn[0, t_start:t_end, t_start:t_end, :].mean(dim=-1).numpy()
    
    donor_contacts_np = donor_contacts[:d_end-d_start, :d_end-d_start].numpy()
    target_contacts_np = target_contacts[:t_end-t_start, :t_end-t_start].numpy()
    
    # Row 1: Attention patterns
    ax = axes[0, 0]
    im = ax.imshow(donor_region_attn, cmap='Blues', aspect='auto')
    ax.contour(donor_contacts_np, levels=[0.5], colors='lime', linewidths=2)
    ax.set_title('Donor Attention\n(green = donor contacts)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 1]
    im = ax.imshow(target_region_attn, cmap='Blues', aspect='auto')
    ax.contour(target_contacts_np, levels=[0.5], colors='red', linewidths=2)
    ax.set_title('Target Attention\n(red = target contacts)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 2]
    im = ax.imshow(patched_region_attn, cmap='Blues', aspect='auto')
    min_size = min(donor_contacts_np.shape[0], patched_region_attn.shape[0])
    donor_overlay = donor_contacts_np[:min_size, :min_size]
    target_overlay = target_contacts_np[:min_size, :min_size]
    ax.contour(donor_overlay, levels=[0.5], colors='lime', linewidths=2)
    ax.contour(target_overlay, levels=[0.5], colors='red', linewidths=2, linestyles='dashed')
    ax.set_title('Patched Attention')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 3]
    diff = patched_region_attn - target_region_attn
    vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title('Patched - Target')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Contact maps
    ax = axes[1, 0]
    ax.imshow(donor_contacts_np, cmap='Greens', aspect='auto')
    ax.set_title('Donor Contacts')
    
    ax = axes[1, 1]
    ax.imshow(target_contacts_np, cmap='Reds', aspect='auto')
    ax.set_title('Target Contacts')
    
    ax = axes[1, 2]
    contact_diff = donor_overlay.astype(float) - target_overlay.astype(float)
    ax.imshow(contact_diff, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_title('Donor - Target Contacts')
    
    ax = axes[1, 3]
    min_size = min(diff.shape[0], contact_diff.shape[0])
    diff_flat = diff[:min_size, :min_size].flatten()
    contact_diff_flat = contact_diff[:min_size, :min_size].flatten()
    n = min_size
    mask = ~np.eye(n, dtype=bool).flatten()
    ax.scatter(contact_diff_flat[mask], diff_flat[mask], alpha=0.5, s=20)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.set_xlabel('Contact Diff')
    ax.set_ylabel('Attention Diff')
    
    plt.suptitle(f'{case_name} - Block {block_idx}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_attention_shift_summary(metrics_df: pd.DataFrame, output_path: str):
    """Visualize attention shifts across blocks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    numeric_cols = [c for c in metrics_df.select_dtypes(include=[np.number]).columns if c != 'block_idx']
    by_block = metrics_df.groupby('block_idx')[numeric_cols].mean().reset_index()
    
    ax = axes[0, 0]
    if 'pct_change_attn_to_donor_contacts' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['pct_change_attn_to_donor_contacts'],
                'o-', label='To donor', color='green', linewidth=2)
    if 'pct_change_attn_to_target_contacts' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['pct_change_attn_to_target_contacts'],
                's-', label='To target', color='red', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Block')
    ax.set_ylabel('% Change')
    ax.set_title('Attention Change After Patching')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    if 'pct_change_attn_to_unique_donor' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['pct_change_attn_to_unique_donor'],
                'o-', label='Donor-only', color='green', linewidth=2)
    if 'pct_change_attn_to_unique_target' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['pct_change_attn_to_unique_target'],
                's-', label='Target-only', color='red', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Block')
    ax.set_ylabel('% Change')
    ax.set_title('Attention to UNIQUE Contacts')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    if 'mean_attn_to_foreign_contacts' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['mean_attn_to_foreign_contacts'],
                'o-', label='Donor (patched)', color='green')
    if 'baseline_attn_to_donor_contacts' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['baseline_attn_to_donor_contacts'],
                'o--', label='Donor (baseline)', color='green', alpha=0.5)
    ax.set_xlabel('Block')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Raw Attention Values')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    if 'attn_foreign_auc' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['attn_foreign_auc'],
                'o-', label='Donor AUC', color='green')
    if 'attn_own_auc' in by_block.columns:
        ax.plot(by_block['block_idx'], by_block['attn_own_auc'],
                's-', label='Target AUC', color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.set_xlabel('Block')
    ax.set_ylabel('AUC')
    ax.set_title('Attention as Contact Predictor')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PART 8: MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/block_patching_successes.csv')
    parser.add_argument('--output', type=str, default='attention_analysis_v2')
    parser.add_argument('--n-cases', type=int, default=5)
    parser.add_argument('--patch-block', type=int, default=27)
    parser.add_argument('--analysis-blocks', type=int, nargs='+', default=[27, 30, 35, 40, 45, 47])
    parser.add_argument('--patch-mask-mode', type=str, default='intra', choices=['touch', 'intra', 'hole'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--flush-every', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Filter for pairwise patches with hairpins found at patch_block
    df_filtered = df[
        (df['patch_mode'] == 'pairwise') & 
        (df['hairpin_found'] == True) & 
        (df['block_idx'] == args.patch_block)
    ]
    
    pairs = df_filtered[['donor_pdb', 'donor_sequence', 'donor_hairpin_start', 'donor_hairpin_end',
                         'target_name', 'target_sequence', 'target_start', 'target_end']].drop_duplicates()
    pairs = pairs.head(args.n_cases)
    print(f"Analyzing {len(pairs)} cases")
    
    all_metrics = []
    results_path = os.path.join(args.output, 'attention_metrics.csv')
    
    for idx, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Cases"):
        donor_seq = row['donor_sequence']
        target_seq = row['target_sequence']
        donor_region = (int(row['donor_hairpin_start']), int(row['donor_hairpin_end']))
        target_region = (int(row['target_start']), int(row['target_end']))
        case_name = f"{row['donor_pdb']}_to_{row['target_name']}"
        
        case_dir = os.path.join(args.output, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Run donor
        _, donor_collector, donor_contacts, _ = run_and_collect_attention(
            model, tokenizer, device, donor_seq, args.analysis_blocks
        )
        donor_region_contacts = donor_contacts[donor_region[0]:donor_region[1], 
                                                donor_region[0]:donor_region[1]]
        
        # Run target baseline
        _, target_collector, target_contacts, _ = run_and_collect_attention(
            model, tokenizer, device, target_seq, args.analysis_blocks
        )
        target_region_contacts = target_contacts[target_region[0]:target_region[1],
                                                  target_region[0]:target_region[1]]
        
        # Run patched
        _, patched_collector, patched_contacts, _, pairwise_mask = run_patched_and_collect_attention(
            model, tokenizer, device, target_seq, donor_seq,
            target_region, donor_region, args.patch_block, args.analysis_blocks,
            args.patch_mask_mode,
        )
        
        # Analyze each block
        for block_idx in args.analysis_blocks:
            if block_idx not in patched_collector.attn_with_bias:
                continue
            
            # Visualize
            visualize_three_way_comparison(
                donor_collector.attn_with_bias[block_idx],
                target_collector.attn_with_bias[block_idx],
                patched_collector.attn_with_bias[block_idx],
                donor_region_contacts,
                target_region_contacts,
                donor_region, target_region,
                block_idx,
                os.path.join(case_dir, f'comparison_block{block_idx}.png'),
                case_name,
            )
            
            # Compute metrics
            baseline_metrics = compute_attention_to_contacts(
                target_collector.attn_with_bias[block_idx],
                target_contacts, target_region[0], target_region[1],
            )
            
            patched_alignment = compute_attention_alignment_with_foreign_contacts(
                patched_collector.attn_with_bias[block_idx],
                target_region_contacts, donor_region_contacts,
                target_region[0], target_region[1],
                baseline_attn=target_collector.attn_with_bias[block_idx],
            )
            
            metrics = {
                'case': case_name,
                'block_idx': block_idx,
                'patch_mask_mode': args.patch_mask_mode,
                **baseline_metrics,
                **patched_alignment,
            }
            all_metrics.append(metrics)
        
        # Flush periodically
        if (len(all_metrics) > 0) and (idx % args.flush_every == 0):
            interim_df = pd.DataFrame(all_metrics)
            interim_df.to_csv(results_path, index=False)
        
        torch.cuda.empty_cache()
    
    # Final save
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(results_path, index=False)
    metrics_df.to_csv(os.path.join(args.output, 'attention_metrics.csv'), index=False)
    
    visualize_attention_shift_summary(metrics_df, os.path.join(args.output, 'summary.png'))
    
    # Print summary
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*60)
    
    for block_idx in args.analysis_blocks:
        block_data = metrics_df[metrics_df['block_idx'] == block_idx]
        if len(block_data) > 0:
            pct_donor = block_data.get('pct_change_attn_to_unique_donor', pd.Series([None])).mean()
            pct_target = block_data.get('pct_change_attn_to_unique_target', pd.Series([None])).mean()
            
            if pd.notna(pct_donor) and pd.notna(pct_target):
                print(f"Block {block_idx}: Donor contacts {pct_donor:+.1f}%, Target contacts {pct_target:+.1f}%")
    
    print(f"\nResults saved to {args.output}/")


if __name__ == '__main__':
    main()