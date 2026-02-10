"""
Information Flow Visualization
==============================

Plots the relative contribution of sequence-to-pair vs pair-to-sequence
information flow across ESMFold trunk blocks, normalized to 0-1 scale.

Creates publication-ready figures with filled area plots showing how
information flow patterns change across the 48 trunk blocks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the stats
OUTPUT_DIR = "information_flow_donor_simplified"
stats_df = pd.read_parquet(f'{OUTPUT_DIR}/stats.parquet')

# Get the data (no smoothing - show raw results)
pair2seq_h = stats_df.groupby('block')['pair2seq_relative_hairpin'].mean()
seq2pair_h = stats_df.groupby('block')['seq2pair_relative_hairpin'].mean()

# Normalize to 0-1
pair2seq_scaled = (pair2seq_h - pair2seq_h.min()) / (pair2seq_h.max() - pair2seq_h.min() + 1e-10)
seq2pair_scaled = (seq2pair_h - seq2pair_h.min()) / (seq2pair_h.max() - seq2pair_h.min() + 1e-10)

# Create the plot
fig, ax = plt.subplots(figsize=(7, 6))

    # colors = {"sequence": "#d95f02", "pairwise": "#1b9e77"}  # Burnt orange, Teal

# Colors matching the reference
teal = '#1b9e77'
orange = '#d95f02'

# Plot with filled areas
ax.fill_between(pair2seq_scaled.index, 0, pair2seq_scaled.values, 
                color=teal, alpha=0.4, label='Pair→Seq')
ax.plot(pair2seq_scaled.index, pair2seq_scaled.values, 
        color=teal, linewidth=2.5)

ax.fill_between(seq2pair_scaled.index, 0, seq2pair_scaled.values, 
                color=orange, alpha=0.4, label='Seq→Pair')
ax.plot(seq2pair_scaled.index, seq2pair_scaled.values, 
        color=orange, linewidth=2.5)

# Add regime boundaries
# ax.axvline(x=15, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
# ax.axvline(x=35, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Style the remaining spines
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Labels with doubled text size
ax.set_xlabel('Block', fontsize=30)
ax.set_ylabel('Normalized Contribution', fontsize=28)

# Tick labels doubled
ax.tick_params(axis='both', which='major', labelsize=24, width=1.5, length=8)

# Legend with doubled text
# ax.legend(fontsize=24, frameon=True, loc='upper center')
ax.legend(
    fontsize=24,
    frameon=True,
    loc='upper center',
    bbox_to_anchor=(0.40, 1.0)  # smaller x = move left
)


# Grid
ax.grid(alpha=0.3, linewidth=1)

# Set limits
ax.set_xlim(0, 47)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/info_flow_hairpin_custom.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/info_flow_hairpin_custom.pdf', bbox_inches='tight')
plt.close()

print(f"Saved to {OUTPUT_DIR}/info_flow_hairpin_custom.png and .pdf")