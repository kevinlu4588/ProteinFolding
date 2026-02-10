"""
Multi-Module Activation Patching
================================

Compares the structural information content across ESMFold's three main components:
1. ESM Encoder - language model embeddings (36 layers)
2. Folding Trunk - structure-aware transformer (48 blocks)
3. Structure Module - IPA-based coordinate refinement (8 blocks)

For each module, patches ALL block outputs simultaneously from a hairpin-containing
donor into a helical acceptor. Supports patching sequence (s), pairwise (z), or
both representations, with optional masking to target specific residue pairs.

Key finding: The folding trunk is necessary and sufficient for hairpin formation,
while encoder and structure module patching have minimal effect.

Usage:
    python module_patching.py --csv patching_dataset.csv --output_dir results/
"""

import argparse
import os
import types
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    Rigid,
    Rotation,
)
from transformers.utils import ContextManagers

from src.utils.trunk_utils import detect_hairpins


# ============================================================================
# PART 1: DATA CLASSES
# ============================================================================

@dataclass
class CollectedRepresentations:
    """
    Container for collected representations from ESMFold.
    
    All tensors are stored detached and on CPU to avoid memory issues.
    Dict keys are block/layer indices.
    """
    # ESM encoder layers (before folding trunk)
    esm_layers: Dict[int, torch.Tensor] = field(default_factory=dict)  # layer_idx -> [B, L, D]
    
    # Trunk block outputs
    s_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)  # block_idx -> [B, L, D_s]
    z_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)  # block_idx -> [B, L, L, D_z]
    
    # Structure module IPA outputs
    ipa_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)  # sm_block_idx -> [B, L, D]
    
    def clear(self):
        """Clear all collected representations."""
        for attr in self.__dataclass_fields__:
            getattr(self, attr).clear()


# ============================================================================
# PART 2: HOOK MANAGERS (collection only, no forward patching)
# ============================================================================

class ESMEncoderHooks:
    """
    Collect ESM encoder layer outputs via hooks.
    
    Usage:
        collector = CollectedRepresentations()
        hooks = ESMEncoderHooks(model.esm, collector)
        hooks.register()
        outputs = model(**inputs)
        hooks.remove()
        # collector.esm_layers now populated
    """
    
    def __init__(self, esm_module: nn.Module, collector: CollectedRepresentations):
        self.esm = esm_module
        self.collector = collector
        self.handles: List = []
    
    def register(self, layers: Any = 'all'):
        """Register hooks on ESM encoder layers."""
        if layers == 'all':
            layers = range(len(self.esm.encoder.layer))
        
        for idx in layers:
            def make_hook(layer_idx):
                def hook(module, inputs, outputs):
                    # ESM layer outputs hidden states (possibly as tuple)
                    tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                    self.collector.esm_layers[layer_idx] = tensor.detach().cpu()
                return hook
            
            handle = self.esm.encoder.layer[idx].register_forward_hook(make_hook(idx))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class TrunkHooks:
    """
    Collect trunk block outputs via hooks.
    
    Usage:
        collector = CollectedRepresentations()
        hooks = TrunkHooks(model.trunk, collector)
        hooks.register(collect_s=True, collect_z=True)
        outputs = model(**inputs)
        hooks.remove()
        # collector.s_blocks, collector.z_blocks now populated
    """
    
    def __init__(self, trunk: nn.Module, collector: CollectedRepresentations):
        self.trunk = trunk
        self.collector = collector
        self.handles: List = []
    
    def register(
        self,
        blocks: Any = 'all',
        collect_s: bool = True,
        collect_z: bool = True,
    ):
        """Register hooks on trunk blocks."""
        if blocks == 'all':
            blocks = range(len(self.trunk.blocks))
        
        for idx in blocks:
            block = self.trunk.blocks[idx]
            
            def make_hook(block_idx, do_s, do_z):
                def hook(module, inputs, outputs):
                    s, z = outputs
                    if do_s:
                        self.collector.s_blocks[block_idx] = s.detach().cpu()
                    if do_z:
                        self.collector.z_blocks[block_idx] = z.detach().cpu()
                return hook
            
            handle = block.register_forward_hook(make_hook(idx, collect_s, collect_z))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class IPAHooks:
    """
    Collect IPA outputs via hooks.
    
    Note: IPA is called multiple times per forward (once per SM block).
    Call reset() before each forward pass.
    
    Usage:
        collector = CollectedRepresentations()
        hooks = IPAHooks(model.trunk.structure_module, collector)
        hooks.register()
        hooks.reset()  # Important!
        outputs = model(**inputs)
        hooks.remove()
        # collector.ipa_outputs now populated
    """
    
    def __init__(self, structure_module: nn.Module, collector: CollectedRepresentations):
        self.sm = structure_module
        self.collector = collector
        self.handles: List = []
        self._call_idx = 0
    
    def register(self):
        """Register hook on IPA module."""
        def hook(module, inputs, outputs):
            self.collector.ipa_outputs[self._call_idx] = outputs.detach().cpu()
            self._call_idx += 1
        
        handle = self.sm.ipa.register_forward_hook(hook)
        self.handles.append(handle)
    
    def reset(self):
        """Reset call counter. Call before each forward pass."""
        self._call_idx = 0
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
# PART 3: COLLECTION CONVENIENCE FUNCTION
# ============================================================================

def run_and_collect(
    model,
    tokenizer,
    device: str,
    sequence: str,
    collect_esm: bool = True,
    collect_trunk: bool = True,
    collect_s: bool = True,
    collect_z: bool = True,
    collect_ipa: bool = False,
    num_recycles: int = 0,
) -> Tuple[EsmForProteinFoldingOutput, CollectedRepresentations]:
    """
    Run model and collect representations using hooks only.
    
    No monkey-patching - just hooks on existing modules.
    
    Args:
        model: ESMFold model
        tokenizer: ESMFold tokenizer
        device: Device string
        sequence: Protein sequence
        collect_esm: Collect ESM encoder layer outputs
        collect_trunk: Collect trunk block outputs
        collect_s: Collect sequence representations (requires collect_trunk)
        collect_z: Collect pairwise representations (requires collect_trunk)
        collect_ipa: Collect IPA outputs from structure module
        num_recycles: Number of recycling iterations
    
    Returns:
        (model_outputs, collected_representations)
    """
    collector = CollectedRepresentations()
    hook_managers = []
    
    # Set up hooks
    if collect_esm:
        esm_hooks = ESMEncoderHooks(model.esm, collector)
        esm_hooks.register()
        hook_managers.append(esm_hooks)
    
    if collect_trunk:
        trunk_hooks = TrunkHooks(model.trunk, collector)
        trunk_hooks.register(collect_s=collect_s, collect_z=collect_z)
        hook_managers.append(trunk_hooks)
    
    if collect_ipa:
        ipa_hooks = IPAHooks(model.trunk.structure_module, collector)
        ipa_hooks.register()
        ipa_hooks.reset()
        hook_managers.append(ipa_hooks)
    
    # Run model
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=num_recycles)
    finally:
        # Always clean up hooks
        for mgr in hook_managers:
            mgr.remove()
    
    return outputs, collector


# ============================================================================
# PART 4: MASKS
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
    """
    Create pairwise patch mask.
    
    Args:
        donor_start, donor_end: Donor hairpin region (0-indexed, exclusive end)
        donor_len: Total donor sequence length
        target_start, target_end: Target patch region (0-indexed, exclusive end)
        target_len: Total target sequence length
        mode: 'intra' (hairpin region only), 'touch' (cross pattern), 'hole' (cross without intra)
    
    Returns:
        Boolean mask of shape [target_len, target_len]
    """
    mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    
    if mode == "intra":
        # Only patch within the hairpin region
        mask[target_start:target_end, target_start:target_end] = True
        
    elif mode in ("touch", "hole"):
        # Compute how far we can extend based on donor/target boundaries
        left_extent = min(donor_start, target_start)
        right_extent = min(donor_len - donor_end, target_len - target_end)
        
        transport_start = target_start - left_extent
        transport_end = target_end + right_extent
        
        # Create cross pattern
        mask[target_start:target_end, transport_start:transport_end] = True
        mask[transport_start:transport_end, target_start:target_end] = True
        
        if mode == "hole":
            # Remove intra-hairpin region
            mask[target_start:target_end, target_start:target_end] = False
    
    return mask


# ============================================================================
# PART 5: INTERVENTION - ESM PATCHING (hook-based, no forward patch!)
# ============================================================================

@contextmanager
def patch_esm_layers(
    model,
    donor_layers: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
):
    """
    Patch ESM encoder layers with donor representations via hooks.
    
    The hooks MODIFY the output by returning a patched tensor.
    
    Args:
        model: ESMFold model
        donor_layers: Dict mapping layer_idx -> [B, patch_len, D] tensor
        target_start, target_end: Where to patch in target sequence
    
    Usage:
        with patch_esm_layers(model, donor_esm, target_start, target_end):
            outputs = model(**inputs)
    """
    handles = []
    
    for layer_idx, donor_tensor in donor_layers.items():
        if layer_idx >= len(model.esm.encoder.layer):
            continue
        
        def make_patch_hook(donor, t_start, t_end):
            def hook(module, inputs, outputs):
                tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                patched = tensor.clone()
                d = donor.to(patched.device, patched.dtype)
                # +1 for BOS token in ESM
                patched[:, t_start + 1:t_end + 1, :] = d
                if isinstance(outputs, tuple):
                    return (patched,) + outputs[1:]
                return patched
            return hook
        
        layer = model.esm.encoder.layer[layer_idx]
        handle = layer.register_forward_hook(make_patch_hook(donor_tensor, target_start, target_end))
        handles.append(handle)
    
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ============================================================================
# PART 6: INTERVENTION - TRUNK PATCHING (needs forward patch)
# ============================================================================

def make_trunk_all_block_patch_forward(
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
):
    """
    Create a trunk forward that patches ALL blocks with donor representations.
    
    This is an explicit function - you can read exactly what it does.
    If you need different patching behavior, copy and modify.
    
    Args:
        donor_s_blocks: Dict mapping block_idx -> [B, patch_len, D_s] sequence repr
        donor_z_blocks: Dict mapping block_idx -> [B, L, L, D_z] FULL pairwise repr
        target_start, target_end: Where to patch in target
        donor_start: Where donor region starts (for pairwise coordinate mapping)
        pairwise_mask: Boolean mask for which (i,j) pairs to patch
        patch_mode: 'sequence', 'pairwise', or 'both'
    
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
            """Apply donor patch at this block."""
            # Patch sequence
            if patch_mode in ('both', 'sequence') and block_idx in donor_s_blocks:
                donor_s = donor_s_blocks[block_idx].to(s.device, dtype=s.dtype)
                s[:, target_start:target_end, :] = donor_s
            
            # Patch pairwise
            if patch_mode in ('both', 'pairwise') and block_idx in donor_z_blocks:
                donor_z = donor_z_blocks[block_idx].to(z.device, dtype=z.dtype)
                mask_dev = pairwise_mask.to(z.device)
                
                # Apply via mask - map target coords to donor coords
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

        # Standard recycle loop
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
def patch_trunk_all_blocks(
    model,
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
):
    """
    Context manager for trunk all-block patching.
    
    Usage:
        with patch_trunk_all_blocks(model, donor_s, donor_z, ...):
            outputs = model(**inputs)
    """
    original = model.trunk.forward
    
    patched_forward = make_trunk_all_block_patch_forward(
        donor_s_blocks, donor_z_blocks,
        target_start, target_end, donor_start,
        pairwise_mask, patch_mode,
    )
    model.trunk.forward = types.MethodType(patched_forward, model.trunk)
    
    try:
        yield
    finally:
        model.trunk.forward = original


# ============================================================================
# PART 7: INTERVENTION - IPA PATCHING (hook-based!)
# ============================================================================

@contextmanager
def patch_ipa_outputs(
    model,
    donor_ipa: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
):
    """
    Patch IPA outputs with donor representations via hooks.
    
    Note: IPA is called multiple times per forward. We track call count.
    
    Args:
        model: ESMFold model
        donor_ipa: Dict mapping sm_block_idx -> [B, patch_len, D] tensor
        target_start, target_end: Where to patch in target sequence
    """
    handles = []
    call_counter = [0]  # Use list for mutability in closure
    
    def make_patch_hook():
        def hook(module, inputs, outputs):
            idx = call_counter[0]
            call_counter[0] += 1
            
            if idx in donor_ipa:
                patched = outputs.clone()
                donor = donor_ipa[idx].to(patched.device, patched.dtype)
                patched[:, target_start:target_end, :] = donor
                return patched
            return outputs
        return hook
    
    handle = model.trunk.structure_module.ipa.register_forward_hook(make_patch_hook())
    handles.append(handle)
    
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ============================================================================
# PART 8: ANALYSIS UTILITIES
# ============================================================================

def evaluate_hairpin(
    outputs: EsmForProteinFoldingOutput,
    model,
    target_start: int,
    target_end: int,
) -> Dict[str, Any]:
    """
    Evaluate hairpin formation and structure quality.
    
    Returns dict with:
        - hairpin_found: bool
        - mean_plddt: float
        - patch_region_plddt: float
        - ptm: float or None
    """
    # Check for hairpin using trunk_utils
    hairpin_found, _ = detect_hairpins(outputs, model)
    
    # Structure quality metrics
    plddt = outputs.plddt[0].cpu().numpy()
    ptm = outputs.ptm.item() if outputs.ptm is not None else None
    
    return {
        'hairpin_found': hairpin_found,
        'mean_plddt': float(plddt.mean()),
        'patch_region_plddt': float(plddt[target_start:target_end].mean()),
        'ptm': ptm,
    }


def compute_alpha_helix_content(pdb_string: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Compute percentage of residues in alpha helix from a PDB string."""
    import tempfile
    from Bio import PDB
    from Bio.PDB import DSSP
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as f:
        f.write(pdb_string)
        pdb_path = f.name
    
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("model", pdb_path)
        model0 = structure[0]
        
        try:
            dssp = DSSP(model0, pdb_path, dssp="mkdssp")
        except Exception as e:
            print(f"DSSP failed: {e}")
            return None, None, None
        
        total_residues = len(dssp)
        helix_residues = sum(1 for k in dssp.keys() if dssp[k][2] in ["H", "I", "G"])
        helix_percentage = (helix_residues / total_residues * 100) if total_residues > 0 else 0
        
        return helix_residues, total_residues, helix_percentage
        
    except Exception as e:
        print(f"Helix content computation failed: {e}")
        return None, None, None
    finally:
        try:
            os.unlink(pdb_path)
        except:
            pass


# ============================================================================
# PART 9: PLOTTING UTILITIES
# ============================================================================

def generate_summary_plots(results_df: pd.DataFrame, output_dir: str):
    """
    Generate summary plots from results using plotting functions.
    """
    from pathlib import Path
    
    # Import plotting functions from your module
    try:
        from module_plotting import (
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
        print("  Skipping plot generation")


# ============================================================================
# PART 10: MAIN EXPERIMENT
# ============================================================================

def run_experiment_on_dataset(
    csv_path: str,
    output_dir: str,
    patch_modules: List[str] = ["encoder", "trunk", "structure_module"],
    patch_modes: List[str] = ["both"],
    patch_mask_modes: List[str] = ["intra", "touch"],
    save_pdbs: bool = False,
    compute_helix: bool = False,
    n_cases: Optional[int] = None,
    device: Optional[str] = None,
    flush_every: int = 20,
) -> pd.DataFrame:
    """
    Run all-block patching experiments on the patching dataset.
    
    Uses hook-based collection and context managers for clean intervention.
    
    Args:
        csv_path: Path to input CSV
        output_dir: Output directory
        patch_modules: Which modules to patch
        patch_modes: Trunk patch modes (sequence, pairwise, both)
        patch_mask_modes: Pairwise mask modes (intra, touch, hole)
        save_pdbs: Save PDB structures
        compute_helix: Compute helix content
        n_cases: Number of cases to run (None = all)
        device: Device string
        flush_every: Save results and regenerate plots every N cases
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
    print(f"Loading patching dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if n_cases is not None:
        df = df.head(n_cases)
    print(f"Running {len(df)} cases")
    
    # Results storage
    all_results = []
    results_path = os.path.join(output_dir, "all_block_patching_results.parquet")
    
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
        
        print(f"\nCase {case_idx}: loop {row['loop_idx']} <- {row['donor_pdb']}")
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
        # COLLECT DONOR REPRESENTATIONS (using hooks!)
        # =====================================================================
        print("  Collecting donor representations...")
        donor_outputs, donor_collected = run_and_collect(
            model, tokenizer, device, donor_seq,
            collect_esm=True,
            collect_trunk=True,
            collect_s=True,
            collect_z=True,
            collect_ipa=True,
        )
        
        # Extract hairpin region for sequence-like representations
        donor_esm_region = {
            k: v[:, donor_start:donor_end, :] 
            for k, v in donor_collected.esm_layers.items()
        }
        donor_s_region = {
            k: v[:, donor_start:donor_end, :] 
            for k, v in donor_collected.s_blocks.items()
        }
        donor_ipa_region = {
            k: v[:, donor_start:donor_end, :] 
            for k, v in donor_collected.ipa_outputs.items()
        }
        # Keep full z_blocks for pairwise patching (mask handles coordinate mapping)
        donor_z_full = donor_collected.z_blocks
        
        print(f"    ESM layers: {len(donor_esm_region)}")
        print(f"    Trunk blocks (s): {len(donor_s_region)}")
        print(f"    Trunk blocks (z): {len(donor_z_full)}")
        print(f"    IPA outputs: {len(donor_ipa_region)}")
        
        del donor_outputs
        torch.cuda.empty_cache()
        
        # Compute helix content for original target (if requested)
        orig_helix_pct = None
        if compute_helix:
            with torch.no_grad():
                target_inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                orig_outputs = model(**target_inputs, num_recycles=0)
            orig_pdb = model.output_to_pdb(orig_outputs)[0]
            _, _, orig_helix_pct = compute_alpha_helix_content(orig_pdb)
            del orig_outputs
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
        # INPUT INTERVENTION BASELINE
        # =====================================================================
        if "literal_patched_sequence" in row.index and pd.notna(row["literal_patched_sequence"]):
            literal_seq = row["literal_patched_sequence"]
            
            with torch.no_grad():
                literal_inputs = tokenizer(literal_seq, return_tensors='pt', add_special_tokens=False).to(device)
                literal_outputs = model(**literal_inputs, num_recycles=0)
            
            eval_result = evaluate_hairpin(
                literal_outputs, model, target_start, target_end
            )
            
            input_result = {
                "patch_module": "input",
                "patch_mode": "input_intervention",
                "patch_mask_mode": "n/a",
                **eval_result,
            }
            input_result.update(case_meta)
            
            if compute_helix:
                literal_pdb = model.output_to_pdb(literal_outputs)[0]
                _, _, literal_helix_pct = compute_alpha_helix_content(literal_pdb)
                input_result["original_helix_pct"] = orig_helix_pct
                input_result["patched_helix_pct"] = literal_helix_pct
            
            if save_pdbs:
                input_result["pdb_string"] = model.output_to_pdb(literal_outputs)[0]
            
            case_results.append(input_result)
            print(f"  Input intervention: hairpin={eval_result['hairpin_found']}")
            
            del literal_outputs
            torch.cuda.empty_cache()
        
        # =====================================================================
        # ACTIVATION PATCHING EXPERIMENTS
        # =====================================================================
        for patch_module in patch_modules:
            
            # -----------------------------------------------------------------
            # ESM ENCODER PATCHING (hook-based)
            # -----------------------------------------------------------------
            if patch_module == "encoder":
                print("  Running ESM encoder patch...")
                
                with patch_esm_layers(model, donor_esm_region, target_start, target_end):
                    with torch.no_grad():
                        inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                        outputs = model(**inputs, num_recycles=0)
                
                eval_result = evaluate_hairpin(
                    outputs, model, target_start, target_end
                )
                
                result = {
                    "patch_module": "encoder",
                    "patch_mode": "sequence",
                    "patch_mask_mode": "n/a",
                    **eval_result,
                }
                result.update(case_meta)
                
                if compute_helix:
                    pdb_str = model.output_to_pdb(outputs)[0]
                    _, _, patched_helix_pct = compute_alpha_helix_content(pdb_str)
                    result["original_helix_pct"] = orig_helix_pct
                    result["patched_helix_pct"] = patched_helix_pct
                
                if save_pdbs:
                    result["pdb_string"] = model.output_to_pdb(outputs)[0]
                
                case_results.append(result)
                print(f"    ESM patch: hairpin={eval_result['hairpin_found']}")
                
                del outputs
                torch.cuda.empty_cache()
            
            # -----------------------------------------------------------------
            # TRUNK PATCHING (forward patch via context manager)
            # -----------------------------------------------------------------
            elif patch_module == "trunk":
                for patch_mode in patch_modes:
                    for mask_mode in patch_mask_modes:
                        # Skip redundant mask modes for sequence-only patching
                        if patch_mode == "sequence" and mask_mode != patch_mask_modes[0]:
                            continue
                        
                        print(f"  Running trunk patch: {patch_mode}, {mask_mode}...")
                        
                        with patch_trunk_all_blocks(
                            model, donor_s_region, donor_z_full,
                            target_start, target_end, donor_start,
                            pairwise_masks[mask_mode], patch_mode
                        ):
                            with torch.no_grad():
                                inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                                outputs = model(**inputs, num_recycles=0)
                        
                        eval_result = evaluate_hairpin(
                            outputs, model, target_start, target_end
                        )
                        
                        result = {
                            "patch_module": "trunk",
                            "patch_mode": patch_mode,
                            "patch_mask_mode": mask_mode,
                            **eval_result,
                        }
                        result.update(case_meta)
                        
                        if compute_helix:
                            pdb_str = model.output_to_pdb(outputs)[0]
                            _, _, patched_helix_pct = compute_alpha_helix_content(pdb_str)
                            result["original_helix_pct"] = orig_helix_pct
                            result["patched_helix_pct"] = patched_helix_pct
                        
                        if save_pdbs:
                            result["pdb_string"] = model.output_to_pdb(outputs)[0]
                        
                        case_results.append(result)
                        print(f"    Trunk {patch_mode}/{mask_mode}: hairpin={eval_result['hairpin_found']}")
                        
                        del outputs
                        torch.cuda.empty_cache()
            
            # -----------------------------------------------------------------
            # STRUCTURE MODULE IPA PATCHING (hook-based)
            # -----------------------------------------------------------------
            elif patch_module == "structure_module":
                print("  Running IPA patch...")
                
                with patch_ipa_outputs(model, donor_ipa_region, target_start, target_end):
                    with torch.no_grad():
                        inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                        outputs = model(**inputs, num_recycles=0)
                
                eval_result = evaluate_hairpin(
                    outputs, model, target_start, target_end
                )
                
                result = {
                    "patch_module": "structure_module",
                    "patch_mode": "ipa_out",
                    "patch_mask_mode": "n/a",
                    **eval_result,
                }
                result.update(case_meta)
                
                if compute_helix:
                    pdb_str = model.output_to_pdb(outputs)[0]
                    _, _, patched_helix_pct = compute_alpha_helix_content(pdb_str)
                    result["original_helix_pct"] = orig_helix_pct
                    result["patched_helix_pct"] = patched_helix_pct
                
                if save_pdbs:
                    result["pdb_string"] = model.output_to_pdb(outputs)[0]
                
                case_results.append(result)
                print(f"    IPA patch: hairpin={eval_result['hairpin_found']}")
                
                del outputs
                torch.cuda.empty_cache()
        
        all_results.extend(case_results)
        
        # Flush results and regenerate plots every flush_every cases
        cases_completed = case_idx + 1
        if cases_completed % flush_every == 0 or cases_completed == len(df):
            print(f"\n  Flushing results ({cases_completed} cases completed)...")
            interim_df = pd.DataFrame(all_results)
            interim_df.to_parquet(results_path, index=False)
            generate_summary_plots(interim_df, output_dir)
        
        # Clean up
        del donor_collected, donor_esm_region, donor_s_region, donor_z_full, donor_ipa_region
        torch.cuda.empty_cache()
    
    # Final summary
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(results_path, index=False)
    generate_summary_plots(results_df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Total experiments: {len(results_df)}")
    print(f"\nHairpin detection rate by patch_module:")
    print(results_df.groupby("patch_module")["hairpin_found"].mean())
    print(f"\nHairpin detection rate by patch_module × patch_mode:")
    print(results_df.groupby(["patch_module", "patch_mode"])["hairpin_found"].mean())
    
    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all-block ESMFold patching experiments (refactored)"
    )
    parser.add_argument(
        "--csv", type=str, default="data/patching_dataset.csv",
        help="Path to patching_dataset.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results_all_blocks_v2",
        help="Output directory"
    )
    parser.add_argument(
        "--patch_modules", nargs="+", default=["encoder", "trunk", "structure_module"],
        help="Modules to patch"
    )
    parser.add_argument(
        "--patch_modes", nargs="+", default=["both"],
        help="Patch modes for trunk (sequence, pairwise, both)"
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
        "--compute_helix", action="store_true",
        help="Compute alpha helix content"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device"
    )
    parser.add_argument(
        "--flush_every", type=int, default=4,
        help="Save results and regenerate plots every N cases"
    )
    
    args = parser.parse_args()
    
    results_df = run_experiment_on_dataset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        patch_modules=args.patch_modules,
        patch_modes=args.patch_modes,
        patch_mask_modes=args.mask_modes,
        save_pdbs=args.save_pdbs,
        compute_helix=args.compute_helix,
        n_cases=args.n_cases,
        device=args.device,
        flush_every=args.flush_every,
    )

    print(f"\nDone! Results in {args.output_dir}")

if __name__ == "__main__":
    main()
