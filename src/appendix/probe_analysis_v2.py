"""
Amino Acid Identity Probing
===========================

Trains linear probes to predict amino acid identity from ESMFold's intermediate
representations (s and z), then analyzes how patching affects these predictions.

After patching, the probe's predictions shift toward the donor's amino acids at
positions where structural information was transplanted, providing evidence that
the representations encode sequence-specific structural features.

Usage:
    python probe_analysis_v2.py \
        --probing_dataset train.csv \
        --patch_dataset patch.csv \
        --output results/
"""

import argparse
import os
import types
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from contextlib import contextmanager

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import EsmFoldingTrunk, EsmForProteinFoldingOutput
from transformers.utils import ContextManagers

from src.utils.trunk_utils import detect_hairpins
from src.utils.representation_utils import CollectedRepresentations, TrunkHooks


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
NUM_AAS = len(AA_LIST)


# ============================================================================
# PART 1: EXTENDED COLLECTION WITH seq2pair AND pair2seq
# ============================================================================

def run_and_collect_all(
    model,
    tokenizer,
    device: str,
    sequence: str,
    num_recycles: int = 0,
) -> Tuple[EsmForProteinFoldingOutput, CollectedRepresentations]:
    """
    Run model and collect ALL intermediate representations:
    - s (sequence repr after each block)
    - z (pairwise repr after each block)
    - seq2pair updates
    - pair2seq biases
    """
    collector = CollectedRepresentations()
    
    trunk_hooks = TrunkHooks(model.trunk, collector)
    trunk_hooks.register(
        collect_s=True,
        collect_z=True,
        collect_seq2pair=True,
        collect_pair2seq=True,
    )
    
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=num_recycles)
    finally:
        trunk_hooks.remove()
    
    return outputs, collector


# ============================================================================
# PART 2: PAIRWISE MASK CREATION
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
# PART 3: PATCHING WITH COMPONENT COLLECTION
# ============================================================================

def make_trunk_patch_and_collect_forward(
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    patch_block: int,
):
    """
    Create a trunk forward that:
    1. Patches at a specific block
    2. Collects all intermediate components (s, z, seq2pair, bias) for probing
    """
    
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            no_recycles += 1

        def apply_patch(block_idx, s, z):
            """Apply donor patch at the target block."""
            if block_idx != patch_block:
                return s, z
            
            if patch_mode in ('both', 'sequence') and block_idx in donor_s_blocks:
                donor_s = donor_s_blocks[block_idx].to(s.device, dtype=s.dtype)
                s = s.clone()
                s[:, target_start:target_end, :] = donor_s
            
            if patch_mode in ('both', 'pairwise') and block_idx in donor_z_blocks:
                donor_z = donor_z_blocks[block_idx].to(z.device, dtype=z.dtype)
                mask_dev = pairwise_mask.to(z.device)
                z = z.clone()
                
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
            
            collected = {
                's_list': [],
                'z_list': [],
                'seq2pair_list': [],
                'pair2seq_bias_list': [],
            }
            
            for block_idx, block in enumerate(self.blocks):
                # Collect pair2seq bias BEFORE block computation
                bias = block.pair_to_sequence(z)
                collected['pair2seq_bias_list'].append(bias.detach().cpu())
                
                # Run block
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
                
                # Apply patch AFTER block computation
                s, z = apply_patch(block_idx, s, z)
                
                # Collect outputs
                collected['s_list'].append(s.detach().cpu())
                collected['z_list'].append(z.detach().cpu())
                
                # Collect seq2pair (would need to extract from block internals)
                # For now, approximate with outer product of s
                seq2pair_approx = block.sequence_to_pair(s)
                collected['seq2pair_list'].append(seq2pair_approx.detach().cpu())
            
            return s, z, collected

        s_s, s_z = s_s_0, s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        for recycle_idx in range(no_recycles):
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

                s_s, s_z, collected = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

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
        structure["collected"] = collected
        return structure
    
    return forward


@contextmanager
def patch_and_collect(
    model,
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    patch_block: int,
):
    """
    Context manager for patching with component collection.
    
    Usage:
        with patch_and_collect(model, ...) as ctx:
            outputs = model(**inputs)
            collected = outputs['collected']  # Contains s_list, z_list, etc.
    """
    original = model.trunk.forward
    
    patched_forward = make_trunk_patch_and_collect_forward(
        donor_s_blocks, donor_z_blocks,
        target_start, target_end, donor_start,
        pairwise_mask, patch_mode, patch_block,
    )
    model.trunk.forward = types.MethodType(patched_forward, model.trunk)
    
    try:
        yield
    finally:
        model.trunk.forward = original


# ============================================================================
# PART 4: LINEAR PROBE
# ============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe for AA prediction."""
    
    def __init__(self, input_dim: int, num_classes: int = NUM_AAS):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================================
# PART 5: PROBE TRAINING
# ============================================================================

def train_probes_online(
    train_seqs: List[str],
    model,
    tokenizer,
    device: str,
    blocks_to_train: List[int],
    n_pairs_per_seq: int = 100,
    n_epochs: int = 3,
    lr: float = 1e-3,
) -> Dict[int, Dict[str, LinearProbe]]:
    """
    Train probes in an online fashion, one sequence at a time.
    
    Returns:
        Dict mapping block_idx -> {'s': probe, 'z': probe, 'seq2pair': probe, 'bias': probe}
    """
    print(f"Training probes on {len(train_seqs)} sequences for {n_epochs} epochs...")
    
    # Get sample to determine dimensions
    sample_seq = next((s for s in train_seqs if all(aa in AA_TO_IDX for aa in s)), None)
    if sample_seq is None:
        raise ValueError("No valid sequences found")
    
    _, sample_collected = run_and_collect_all(model, tokenizer, device, sample_seq)
    
    # Initialize probes
    probes = {}
    optimizers = {}
    
    for block_idx in blocks_to_train:
        probes[block_idx] = {}
        optimizers[block_idx] = {}
        
        s_dim = sample_collected.s_blocks[block_idx].shape[-1]
        z_dim = sample_collected.z_blocks[block_idx].shape[-1]
        seq2pair_dim = sample_collected.seq2pair_updates[block_idx].shape[-1]
        bias_dim = sample_collected.pair2seq_biases[block_idx].shape[-1]
        
        for component, dim in [('s', s_dim), ('z', z_dim), 
                                ('seq2pair', seq2pair_dim), ('bias', bias_dim)]:
            probe = LinearProbe(dim, NUM_AAS).to(device)
            probes[block_idx][component] = probe
            optimizers[block_idx][component] = torch.optim.Adam(
                probe.parameters(), lr=lr, weight_decay=1e-4
            )
    
    del sample_collected
    torch.cuda.empty_cache()
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(n_epochs):
        seq_order = np.random.permutation(len(train_seqs))
        epoch_losses = {b: {c: [] for c in ['s', 'z', 'seq2pair', 'bias']} for b in blocks_to_train}
        
        pbar = tqdm(seq_order, desc=f"Epoch {epoch+1}/{n_epochs}")
        for seq_idx in pbar:
            seq = train_seqs[seq_idx]
            
            if not all(aa in AA_TO_IDX for aa in seq):
                continue
            
            _, collected = run_and_collect_all(model, tokenizer, device, seq)
            L = len(seq)
            
            # Sample pairs for pairwise components
            pairs = [(i, j) for i in range(L) for j in range(L) if i != j]
            n_pairs = min(n_pairs_per_seq, len(pairs))
            sampled_pairs = [pairs[k] for k in np.random.choice(len(pairs), n_pairs, replace=False)]
            
            # Labels
            seq_labels = torch.tensor([AA_TO_IDX[aa] for aa in seq], dtype=torch.long, device=device)
            pair_labels_i = torch.tensor([AA_TO_IDX[seq[i]] for i, j in sampled_pairs], 
                                         dtype=torch.long, device=device)
            
            for block_idx in blocks_to_train:
                s = collected.s_blocks[block_idx][0].to(device)
                z = collected.z_blocks[block_idx][0].to(device)
                seq2pair = collected.seq2pair_updates[block_idx][0].to(device)
                bias = collected.pair2seq_biases[block_idx][0].to(device)
                
                # Train s probe (per-residue)
                probes[block_idx]['s'].train()
                optimizers[block_idx]['s'].zero_grad()
                logits_s = probes[block_idx]['s'](s)
                loss_s = criterion(logits_s, seq_labels)
                loss_s.backward()
                optimizers[block_idx]['s'].step()
                epoch_losses[block_idx]['s'].append(loss_s.item())
                
                # Train pairwise probes
                for component, tensor in [('z', z), ('seq2pair', seq2pair), ('bias', bias)]:
                    pair_features = torch.stack([tensor[i, j] for i, j in sampled_pairs])
                    
                    probes[block_idx][component].train()
                    optimizers[block_idx][component].zero_grad()
                    logits = probes[block_idx][component](pair_features)
                    loss = criterion(logits, pair_labels_i)
                    loss.backward()
                    optimizers[block_idx][component].step()
                    epoch_losses[block_idx][component].append(loss.item())
            
            del collected
            torch.cuda.empty_cache()
            
            avg_loss = np.mean([np.mean(epoch_losses[blocks_to_train[0]][c]) 
                               for c in ['s', 'z', 'seq2pair', 'bias']])
            pbar.set_postfix({'avg_loss': f'{avg_loss:.3f}'})
    
    return probes


def evaluate_probes(
    probes: Dict[int, Dict[str, LinearProbe]],
    test_seqs: List[str],
    model,
    tokenizer,
    device: str,
    blocks_to_train: List[int],
    n_pairs_per_seq: int = 100,
) -> pd.DataFrame:
    """Evaluate probes on test sequences."""
    print(f"Evaluating probes on {len(test_seqs)} test sequences...")
    
    all_preds = {b: {c: [] for c in ['s', 'z', 'seq2pair', 'bias']} for b in blocks_to_train}
    all_labels = {b: {c: [] for c in ['s', 'z', 'seq2pair', 'bias']} for b in blocks_to_train}
    
    for seq in tqdm(test_seqs, desc="Evaluating"):
        if not all(aa in AA_TO_IDX for aa in seq):
            continue
        
        _, collected = run_and_collect_all(model, tokenizer, device, seq)
        L = len(seq)
        
        pairs = [(i, j) for i in range(L) for j in range(L) if i != j]
        n_pairs = min(n_pairs_per_seq, len(pairs))
        sampled_pairs = [pairs[k] for k in np.random.choice(len(pairs), n_pairs, replace=False)]
        
        seq_labels = [AA_TO_IDX[aa] for aa in seq]
        pair_labels_i = [AA_TO_IDX[seq[i]] for i, j in sampled_pairs]
        
        for block_idx in blocks_to_train:
            s = collected.s_blocks[block_idx][0].to(device)
            z = collected.z_blocks[block_idx][0].to(device)
            seq2pair = collected.seq2pair_updates[block_idx][0].to(device)
            bias = collected.pair2seq_biases[block_idx][0].to(device)
            
            # Evaluate s probe
            probes[block_idx]['s'].eval()
            with torch.no_grad():
                preds_s = probes[block_idx]['s'](s).argmax(dim=-1).cpu().tolist()
            all_preds[block_idx]['s'].extend(preds_s)
            all_labels[block_idx]['s'].extend(seq_labels)
            
            # Evaluate pairwise probes
            for component, tensor in [('z', z), ('seq2pair', seq2pair), ('bias', bias)]:
                pair_features = torch.stack([tensor[i, j] for i, j in sampled_pairs])
                
                probes[block_idx][component].eval()
                with torch.no_grad():
                    preds = probes[block_idx][component](pair_features).argmax(dim=-1).cpu().tolist()
                all_preds[block_idx][component].extend(preds)
                all_labels[block_idx][component].extend(pair_labels_i)
        
        del collected
        torch.cuda.empty_cache()
    
    # Compute accuracies
    accuracies = []
    for block_idx in blocks_to_train:
        for component in ['s', 'z', 'seq2pair', 'bias']:
            preds = all_preds[block_idx][component]
            labels = all_labels[block_idx][component]
            acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
            accuracies.append({
                'block_idx': block_idx,
                'component': component,
                'test_accuracy': acc,
                'n_samples': len(labels),
            })
    
    return pd.DataFrame(accuracies)


# ============================================================================
# PART 5.5: PROBE SAVING AND LOADING
# ============================================================================

def save_probes(
    probes: Dict[int, Dict[str, LinearProbe]],
    path: str,
):
    """
    Save trained probes to disk.
    
    Args:
        probes: Dict mapping block_idx -> {'s': probe, 'z': probe, ...}
        path: Path to save the probes (e.g., 'models/residue_probes.pt')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save state dicts along with architecture info
    save_dict = {
        'probes_state': {
            block_idx: {
                component: probe.state_dict()
                for component, probe in block_probes.items()
            }
            for block_idx, block_probes in probes.items()
        },
        'probe_config': {
            block_idx: {
                component: {
                    'input_dim': probe.linear.in_features,
                    'num_classes': probe.linear.out_features,
                }
                for component, probe in block_probes.items()
            }
            for block_idx, block_probes in probes.items()
        },
        'metadata': {
            'num_blocks': len(probes),
            'components': list(next(iter(probes.values())).keys()) if probes else [],
            'aa_list': AA_LIST,
        }
    }
    
    torch.save(save_dict, path)
    print(f"Saved probes to {path}")


def load_probes(
    path: str,
    device: str = 'cpu',
) -> Dict[int, Dict[str, LinearProbe]]:
    """
    Load trained probes from disk.
    
    Args:
        path: Path to the saved probes file
        device: Device to load probes onto
        
    Returns:
        Dict mapping block_idx -> {'s': probe, 'z': probe, ...}
    """
    save_dict = torch.load(path, map_location=device)
    
    probes = {}
    for block_idx, block_config in save_dict['probe_config'].items():
        probes[block_idx] = {}
        for component, config in block_config.items():
            probe = LinearProbe(
                input_dim=config['input_dim'],
                num_classes=config['num_classes'],
            ).to(device)
            probe.load_state_dict(save_dict['probes_state'][block_idx][component])
            probes[block_idx][component] = probe
    
    print(f"Loaded probes from {path}")
    print(f"  Blocks: {list(probes.keys())}")
    print(f"  Components: {save_dict['metadata']['components']}")
    
    return probes


# ============================================================================
# PART 6: PROBE READOUT AFTER PATCHING
# ============================================================================

def get_donor_representations(
    model,
    tokenizer,
    device: str,
    donor_seq: str,
    donor_start: int,
    donor_end: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extract donor representations for patching.
    
    Returns:
        donor_s_blocks: Dict of sequence representations sliced to hairpin region
        donor_z_blocks: Dict of FULL pairwise matrices
    """
    _, donor_collected = run_and_collect_all(model, tokenizer, device, donor_seq)
    
    donor_s_blocks = {
        k: v[:, donor_start:donor_end, :] 
        for k, v in donor_collected.s_blocks.items()
    }
    donor_z_blocks = donor_collected.z_blocks  # Full matrices
    
    return donor_s_blocks, donor_z_blocks


def run_patched_forward_with_collection(
    model,
    tokenizer,
    device: str,
    target_seq: str,
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    patch_block: int,
    patch_mode: str = 'sequence',
    patch_mask_mode: str = 'intra',
) -> Dict[str, List[torch.Tensor]]:
    """
    Run patched forward and collect all intermediate components.
    
    This uses a modified trunk forward that both patches and collects.
    """
    donor_end = donor_start + (target_end - target_start)
    
    pairwise_mask = create_pairwise_mask(
        donor_start=donor_start,
        donor_end=donor_end,
        donor_len=donor_z_blocks[0].shape[1],
        target_start=target_start,
        target_end=target_end,
        target_len=len(target_seq),
        mode=patch_mask_mode,
    )
    
    original_forward = model.trunk.forward
    
    patched_forward = make_trunk_patch_and_collect_forward(
        donor_s_blocks, donor_z_blocks,
        target_start, target_end, donor_start,
        pairwise_mask, patch_mode, patch_block,
    )
    model.trunk.forward = types.MethodType(patched_forward, model.trunk)
    
    collected = None
    try:
        with torch.no_grad():
            inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
            
            # We need to call trunk directly to get the collected outputs
            # First prepare the inputs like ESMFold does
            aa = inputs['input_ids']
            B, L = aa.shape
            
            attention_mask = torch.ones_like(aa, device=device)
            position_ids = torch.arange(L, device=device).expand_as(aa)
            
            # Get ESM embeddings
            esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)
            esm_s = model.compute_language_model_representations(esmaa)
            esm_s = esm_s.to(model.esm_s_combine.dtype).detach()
            esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
            
            s_s_0 = model.esm_s_mlp(esm_s)
            s_z_0 = s_s_0.new_zeros(B, L, L, model.config.esmfold_config.trunk.pairwise_state_dim)
            
            if model.config.esmfold_config.embed_aa:
                s_s_0 = s_s_0 + model.embedding(aa)
            
            # Call trunk with patched forward
            structure = model.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=0)
            
            collected = structure.get('collected', {})
            
    finally:
        model.trunk.forward = original_forward
    
    return collected


def probe_readout_at_block(
    probes: Dict[int, Dict[str, LinearProbe]],
    block_idx: int,
    target_seq: str,
    donor_seq: str,
    target_collected: CollectedRepresentations,
    patched_collected: Dict[str, List[torch.Tensor]],
    target_start: int,
    target_end: int,
    donor_start: int,
    donor_end: int,
    device: str,
) -> pd.DataFrame:
    """
    Use probes to read out AA predictions at a specific block.
    
    Compares predictions from target (unpatched) vs patched representations.
    """
    patch_len = min(target_end - target_start, donor_end - donor_start)
    results = []
    
    for rel_pos in range(patch_len):
        target_pos = target_start + rel_pos
        donor_pos = donor_start + rel_pos
        
        if target_pos >= len(target_seq) or donor_pos >= len(donor_seq):
            continue
        
        target_aa = target_seq[target_pos]
        donor_aa = donor_seq[donor_pos]
        
        if target_aa not in AA_TO_IDX or donor_aa not in AA_TO_IDX:
            continue
        
        target_aa_idx = AA_TO_IDX[target_aa]
        donor_aa_idx = AA_TO_IDX[donor_aa]
        
        result = {
            'block_idx': block_idx,
            'target_pos': target_pos,
            'rel_pos': rel_pos,
            'target_aa': target_aa,
            'donor_aa': donor_aa,
            'aa_changed': target_aa != donor_aa,
        }
        
        # Get representations from target (unpatched)
        s_target = target_collected.s_blocks[block_idx][0]
        z_target = target_collected.z_blocks[block_idx][0]
        seq2pair_target = target_collected.seq2pair_updates.get(block_idx, torch.zeros_like(z_target))[0]
        bias_target = target_collected.pair2seq_biases.get(block_idx, torch.zeros_like(s_target))[0]
        
        # Get representations from patched
        s_patched = patched_collected['s_list'][block_idx][0] if patched_collected.get('s_list') else s_target
        z_patched = patched_collected['z_list'][block_idx][0] if patched_collected.get('z_list') else z_target
        seq2pair_patched = patched_collected['seq2pair_list'][block_idx][0] if patched_collected.get('seq2pair_list') else seq2pair_target
        bias_patched = patched_collected['pair2seq_bias_list'][block_idx][0] if patched_collected.get('pair2seq_bias_list') else bias_target
        
        j_mask = torch.arange(z_target.shape[0]) != target_pos
        
        for component in ['s', 'z', 'seq2pair', 'bias']:
            if block_idx not in probes or component not in probes[block_idx]:
                continue
            
            probe = probes[block_idx][component]
            
            if component == 's':
                feat_base = s_target[target_pos]
                feat_patch = s_patched[target_pos]
            elif component == 'z':
                feat_base = z_target[target_pos, j_mask].mean(dim=0)
                feat_patch = z_patched[target_pos, j_mask].mean(dim=0)
            elif component == 'seq2pair':
                if seq2pair_target.dim() > 1 and seq2pair_target.shape[0] > 1:
                    feat_base = seq2pair_target[target_pos, j_mask].mean(dim=0)
                    feat_patch = seq2pair_patched[target_pos, j_mask].mean(dim=0)
                else:
                    continue  # Skip if dimensions don't match
            elif component == 'bias':
                if bias_target.dim() > 1 and bias_target.shape[0] > 1:
                    feat_base = bias_target[target_pos, j_mask].mean(dim=0)
                    feat_patch = bias_patched[target_pos, j_mask].mean(dim=0)
                else:
                    continue
            
            probe.eval()
            with torch.no_grad():
                logits_base = probe(feat_base.unsqueeze(0).to(device))
                logits_patch = probe(feat_patch.unsqueeze(0).to(device))
                
                probs_base = F.softmax(logits_base, dim=-1)[0].cpu()
                probs_patch = F.softmax(logits_patch, dim=-1)[0].cpu()
                
                pred_base = logits_base.argmax().item()
                pred_patch = logits_patch.argmax().item()
            
            result[f'{component}_pred_base'] = IDX_TO_AA[pred_base]
            result[f'{component}_pred_patch'] = IDX_TO_AA[pred_patch]
            result[f'{component}_prob_target_base'] = probs_base[target_aa_idx].item()
            result[f'{component}_prob_donor_base'] = probs_base[donor_aa_idx].item()
            result[f'{component}_prob_target_patch'] = probs_patch[target_aa_idx].item()
            result[f'{component}_prob_donor_patch'] = probs_patch[donor_aa_idx].item()
            result[f'{component}_predicts_target_base'] = pred_base == target_aa_idx
            result[f'{component}_predicts_donor_patch'] = pred_patch == donor_aa_idx
        
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def plot_probe_accuracies(accuracy_df: pd.DataFrame, output_dir: str):
    """Plot probe accuracies across blocks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'s': 'blue', 'z': 'green', 'seq2pair': 'orange', 'bias': 'red'}
    markers = {'s': 'o', 'z': 's', 'seq2pair': '^', 'bias': 'd'}
    
    for component in ['s', 'z', 'seq2pair', 'bias']:
        subset = accuracy_df[accuracy_df['component'] == component]
        ax.plot(subset['block_idx'], subset['test_accuracy'], 
                f'{markers[component]}-', color=colors[component], 
                label=component, linewidth=2, markersize=6)
    
    ax.axhline(y=1/NUM_AAS, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Block Index', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Probe Test Accuracy by Block and Component', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probe_accuracies.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: probe_accuracies.png")


def plot_probe_readout_summary(results_df: pd.DataFrame, output_dir: str):
    """Plot probe readout summary across all configurations."""
    if len(results_df) == 0:
        print("No results to plot")
        return
    
    # Filter to positions where AA changed
    diff_df = results_df[results_df['aa_changed'] == True]
    
    if len(diff_df) == 0:
        print("No AA changes found in results")
        return
    
    configs = diff_df.groupby(['patch_block', 'patch_mode']).size().reset_index()[['patch_block', 'patch_mode']]
    
    n_configs = len(configs)
    if n_configs == 0:
        return
    
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    colors = {'s': '#2ecc71', 'z': '#3498db', 'seq2pair': '#e67e22', 'bias': '#e74c3c'}
    
    for config_idx, (_, config_row) in enumerate(configs.iterrows()):
        ax = axes[config_idx]
        patch_block = config_row['patch_block']
        patch_mode = config_row['patch_mode']
        
        subset = diff_df[
            (diff_df['patch_block'] == patch_block) &
            (diff_df['patch_mode'] == patch_mode)
        ]
        
        for comp, color in colors.items():
            col = f'{comp}_prob_donor_patch'
            if col in subset.columns:
                grouped = subset.groupby('block_idx')[col].mean()
                ax.plot(grouped.index, grouped.values, 'o-', color=color, 
                        linewidth=2, markersize=5, label=comp)
        
        ax.axhline(y=1/NUM_AAS, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=patch_block, color='red', linewidth=2, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Block', fontsize=10)
        ax.set_ylabel('P(donor AA)', fontsize=10)
        ax.set_title(f'Patch @{patch_block}, mode={patch_mode}', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    
    for idx in range(n_configs, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Probe Readout: P(donor AA) After Patching', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'probe_readout_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: probe_readout_summary.png")


# ============================================================================
# PART 8: MAIN EXPERIMENT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='ESMFold Probing Analysis (Refactored)')
    parser.add_argument('--probing_dataset', type=str, default='data/probing_dataset.csv',
                        help='Path to CSV with train/test sequences')
    parser.add_argument('--patch_dataset', type=str, default='data/block_patching_successes.csv',
                        help='Path to CSV for patching experiments')
    parser.add_argument('--output', type=str, default='probing_results_v2',
                        help='Output directory')
    parser.add_argument('--n_train_seqs', type=int, default=None,
                        help='Number of training sequences')
    parser.add_argument('--n_test_seqs', type=int, default=None,
                        help='Number of test sequences')
    parser.add_argument('--n_patch_cases', type=int, default=5,
                        help='Number of patching cases')
    parser.add_argument('--n_pairs_per_seq', type=int, default=150,
                        help='Pairs to sample per sequence for pairwise probes')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Training epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--patch_blocks', type=int, nargs='+', default=[0, 27],
                        help='Blocks to test patching at')
    parser.add_argument('--flush_every', type=int, default=10,
                        help='Save results every N cases')
    parser.add_argument('--probes_path', type=str, default=None,
                        help='Path to save/load probes (default: {output}/models/residue_probes.pt)')
    parser.add_argument('--load_probes', action='store_true',
                        help='Load existing probes instead of training')
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set default probes path
    probes_path = args.probes_path or os.path.join(args.output, 'models', 'residue_probes.pt')
    
    # Load model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    # =========================================================================
    # EXPERIMENT 1: Probe Training (or Loading)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: PROBE TRAINING")
    print("="*60)
    
    blocks_to_train = list(range(48))
    
    if args.load_probes and os.path.exists(probes_path):
        # Load existing probes
        print(f"\nLoading probes from {probes_path}...")
        probes = load_probes(probes_path, device=device)
    else:
        # Train new probes
        print(f"\nLoading probing dataset from {args.probing_dataset}...")
        probing_df = pd.read_csv(args.probing_dataset)
        
        train_seqs = probing_df[probing_df['split'] == 'train']['sequence'].tolist()
        test_seqs = probing_df[probing_df['split'] == 'test']['sequence'].tolist()
        
        # Filter valid sequences
        train_seqs = [s for s in train_seqs if all(aa in AA_TO_IDX for aa in s)]
        test_seqs = [s for s in test_seqs if all(aa in AA_TO_IDX for aa in s)]
        
        if args.n_train_seqs:
            train_seqs = train_seqs[:args.n_train_seqs]
        if args.n_test_seqs:
            test_seqs = test_seqs[:args.n_test_seqs]
        
        print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")
        
        # Train probes
        probes = train_probes_online(
            train_seqs, model, tokenizer, device, blocks_to_train,
            n_pairs_per_seq=args.n_pairs_per_seq, n_epochs=args.n_epochs,
        )
        
        # Save probes
        save_probes(probes, probes_path)
        
        # Evaluate probes
        accuracy_df = evaluate_probes(
            probes, test_seqs, model, tokenizer, device, blocks_to_train,
            n_pairs_per_seq=args.n_pairs_per_seq,
        )
        accuracy_df.to_csv(os.path.join(args.output, 'probe_accuracies.csv'), index=False)
        plot_probe_accuracies(accuracy_df, args.output)
        
        print("\nProbe accuracies:")
        print(accuracy_df.pivot(index='block_idx', columns='component', values='test_accuracy').to_string())
    
    # =========================================================================
    # EXPERIMENT 2: Patching Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: PATCHING ANALYSIS")
    print("="*60)
    
    print(f"\nLoading patch dataset from {args.patch_dataset}...")
    patch_df = pd.read_csv(args.patch_dataset)
    
    if args.n_patch_cases:
        patch_df = patch_df.head(args.n_patch_cases)
    
    print(f"Running {len(patch_df)} patching cases at blocks {args.patch_blocks}")
    
    all_results = []
    results_path = os.path.join(args.output, 'patching_results.parquet')
    
    for case_idx, row in tqdm(patch_df.iterrows(), total=len(patch_df), desc="Patching"):
        target_seq = row['target_sequence']
        donor_seq = row['donor_sequence']
        target_start = int(row['target_patch_start'])
        target_end = int(row['target_patch_end'])
        donor_start = int(row['donor_hairpin_start'])
        donor_end = int(row['donor_hairpin_end'])
        
        # Get target representations (unpatched)
        _, target_collected = run_and_collect_all(model, tokenizer, device, target_seq)
        
        # Get donor representations
        donor_s_blocks, donor_z_blocks = get_donor_representations(
            model, tokenizer, device, donor_seq, donor_start, donor_end
        )
        
        for patch_block in args.patch_blocks:
            # Run patched forward with collection
            patched_collected = run_patched_forward_with_collection(
                model, tokenizer, device, target_seq,
                donor_s_blocks, donor_z_blocks,
                target_start, target_end, donor_start,
                patch_block, patch_mode='sequence', patch_mask_mode='intra',
            )
            
            # Probe readout at all blocks after the patch block
            for obs_block in range(patch_block + 1, 48):
                readout = probe_readout_at_block(
                    probes, obs_block, target_seq, donor_seq,
                    target_collected, patched_collected,
                    target_start, target_end, donor_start, donor_end,
                    device,
                )
                
                if len(readout) > 0:
                    readout['case_idx'] = case_idx
                    readout['patch_block'] = patch_block
                    readout['patch_mode'] = 'sequence'
                    all_results.append(readout)
        
        # Flush periodically
        if (case_idx + 1) % args.flush_every == 0:
            print(f"\n  Flushing results ({case_idx + 1} cases)...")
            interim_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
            if len(interim_df) > 0:
                interim_df.to_parquet(results_path, index=False)
        
        del target_collected, donor_s_blocks, donor_z_blocks
        torch.cuda.empty_cache()
    
    # Final save
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_parquet(results_path, index=False)
        results_df.to_csv(os.path.join(args.output, 'patching_results.csv'), index=False)
        
        plot_probe_readout_summary(results_df, args.output)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        print(f"Total results: {len(results_df)}")
        print(f"Results saved to {args.output}/")
    
    print("\nDone!")


if __name__ == '__main__':
    main()