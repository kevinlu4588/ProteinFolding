# Source Code

This directory contains all experiment and analysis code. Each top-level module corresponds to a major experiment in the paper, and can be run standalone via command-line arguments or through `reproduce.sh`.

## Background

ESMFold has three main components: the **ESM encoder** (36 transformer layers producing sequence embeddings), the **folding trunk** (48 blocks that iteratively refine sequence `s` and pairwise `z` representations), and the **structure module** (8 blocks of Invariant Point Attention that produce 3D coordinates). Our experiments focus on understanding how the folding trunk encodes secondary structure -- specifically beta-hairpin formation.

## Core Experiment Modules

### Activation Patching

| Module | Description |
|--------|-------------|
| `module_patching.py` | Multi-module activation patching. Replaces all block outputs of a given component (encoder, trunk, or structure module) from a donor sequence into an acceptor, testing which component is necessary and sufficient for hairpin formation. |
| `block_patching.py` | Single-block activation patching. Patches one trunk block at a time to identify which blocks encode hairpin structure. Reveals that early blocks (0-10) have the strongest causal effect. |

### Representation Steering

| Module | Description |
|--------|-------------|
| `charge_steering.py` | Hairpin induction via charge signals. Adds complementary charge directions (from difference-of-means vectors) to sequence representations at specified trunk blocks and window positions. |
| `charge_repulsion.py` | Hairpin disruption via charge signals. Adds same-charge (repulsive) signals to existing hairpins to test whether reducing charge complementarity disrupts structure. |
| `contact_steering.py` | Hairpin induction via learned distance directions. Trains Ridge regression probes to predict CA-CA distances from pairwise representations, then uses probe weights as steering vectors. |

### Training

| Module | Description |
|--------|-------------|
| `charge_dom_training.py` | Trains difference-of-means (DoM) charge direction vectors. Groups residues by amino acid charge (K/R/H positive, D/E negative) and computes the mean direction separating them in representation space. Called automatically by `charge_steering.py`. |

### Plotting

Each core module has a corresponding plotting module that generates paper figures:

- `module_plotting.py` -- Success rates, pLDDT comparisons across modules
- `block_plotting.py` -- Per-block patching statistics
- `charge_steering_plotting.py` -- Hairpin geometry and H-bond counts
- `charge_repulsion_plotting.py` -- Disruption summaries
- `contact_steering_plotting.py` -- Distance steering metrics

## `main_paper/`

Additional analyses from the main body of the paper:

| Module | Description |
|--------|-------------|
| `z_scaling.py` | Scales pairwise (`z`) and sequence (`s`) representations independently and measures the effect on predicted structure. Includes autograd analysis of gradients with respect to scale factors. |
| `z_probing_distance.py` | Trains Ridge probes to predict CA-CA distances from `z` representations. Evaluates probe accuracy and produces the distance-predictive directions used by contact steering. |
| `representation_tracking.py` | Tracks how `s` and `z` representations change across trunk blocks. |
| `final_sliding_window_ablation.py` | Ablates contiguous windows of trunk blocks to characterize the temporal resolution of structural decisions. |
| `bias_analysis2.py` | Analyzes positional biases in hairpin patching results. |
| `bias_patching.py` | Controls for positional biases via targeted patching. |
| `bias_plotting.py` / `bias_map_plotting.py` | Plotting for bias analyses. |

## `appendix/`

Supplementary analyses for the paper appendix:

| Module | Description |
|--------|-------------|
| `block_analysis3.py` | Detailed block-level behavior analysis. |
| `charge_probing_binary2.py` | Binary classification probes for charge. |
| `probe_analysis_v2.py` | Probe performance metrics and evaluation. |
| `charge_projection_histograms.py` | Histograms of charge direction projections. |
| `information_flow_plotting.py` | Visualizations of information flow between blocks. |

## `utils/`

Shared utilities used across experiments:

| Module | Description |
|--------|-------------|
| `trunk_utils.py` | Hairpin detection from DSSP secondary structure assignments, handedness classification (Type I/II), CB coordinate extraction, and 3D visualization helpers. |
| `representation_utils.py` | Hook managers (`TrunkHooks`, `ESMEncoderHooks`) for collecting intermediate representations, dataclasses for storing collected outputs (`CollectedRepresentations`), and patching functions for `s` and `z`. |
