"""
Representation Collection Utilities
====================================

Hook-based utilities for collecting intermediate representations from ESMFold.
Provides non-invasive collection of activations at any layer without modifying
the model's forward pass.

Classes:
    CollectedRepresentations: Container for collected activations (s, z, etc.)
    ESMEncoderHooks: Collects ESM language model layer outputs
    TrunkHooks: Collects folding trunk block outputs (s, z, seq2pair, pair2seq)
    IPAHooks: Collects structure module IPA outputs
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

@dataclass
class CollectedRepresentations:
    """Container for collected representations from ESMFold."""
    # ESM encoder layers
    esm_layers: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Trunk block outputs
    s_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    z_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Trunk intermediate representations (optional)
    seq2pair_updates: Dict[int, torch.Tensor] = field(default_factory=dict)
    pair2seq_biases: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Structure module IPA outputs
    ipa_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    
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
    """Collect trunk block outputs via hooks."""
    
    def __init__(self, trunk: nn.Module, collector: CollectedRepresentations):
        self.trunk = trunk
        self.collector = collector
        self.handles: List = []
    
    def register(
        self,
        blocks: Any = 'all',
        collect_s: bool = True,
        collect_z: bool = True,
        collect_seq2pair: bool = False,  # NEW - defaults to False
        collect_pair2seq: bool = False,  # NEW - defaults to False
    ):
        """Register hooks on trunk blocks."""
        if blocks == 'all':
            blocks = range(len(self.trunk.blocks))
        
        for idx in blocks:
            block = self.trunk.blocks[idx]
            
            # Block output hook (s, z)
            if collect_s or collect_z:
                def make_block_hook(block_idx, do_s, do_z):
                    def hook(module, inputs, outputs):
                        s, z = outputs
                        if do_s:
                            self.collector.s_blocks[block_idx] = s.detach().cpu()
                        if do_z:
                            self.collector.z_blocks[block_idx] = z.detach().cpu()
                    return hook
                
                handle = block.register_forward_hook(make_block_hook(idx, collect_s, collect_z))
                self.handles.append(handle)
            
            # seq2pair hook
            if collect_seq2pair:
                def make_s2p_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.seq2pair_updates[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.sequence_to_pair.register_forward_hook(make_s2p_hook(idx))
                self.handles.append(handle)
            
            # pair2seq hook
            if collect_pair2seq:
                def make_p2s_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.pair2seq_biases[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.pair_to_sequence.register_forward_hook(make_p2s_hook(idx))
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