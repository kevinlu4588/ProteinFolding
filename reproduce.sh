#!/usr/bin/env bash
# reproduce.sh — Reproduce all paper experiments
# Usage: bash reproduce.sh
set -e

###############################################################################
# Configuration — adjust case counts here
###############################################################################
N_MODULE=1          # Module patching cases
N_BLOCK=1           # Block patching cases
N_ABLATION=1        # Sliding window ablation cases (sequence & pairwise)
N_STEERING=1        # Charge steering / repulsion / contact steering cases
N_BIAS=1            # Bias analysis & bias patching cases
N_SCALING=1         # Z scaling / gradient cases

RESULTS="results"    # Top-level output directory

###############################################################################
# Environment
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="src:src/main_paper:${PYTHONPATH:-}"

mkdir -p "$RESULTS"

###############################################################################
# Helper
###############################################################################
step() { echo -e "\n====== Step $1: $2 ======\n"; }

###############################################################################
# Step 1 — Module patching
###############################################################################
step_01_module_patching() {
  step 1 "Module patching"
  python src/module_patching.py \
    --csv data/single_block_patching_successes.csv \
    --output_dir "$RESULTS/module_patching" \
    --n_cases "$N_MODULE" \
    --compute_helix
}

###############################################################################
# Step 2 — Plot module patching results
###############################################################################
step_02_module_plotting() {
  step 2 "Module patching plots"
  python src/module_plotting.py \
    --results "$RESULTS/module_patching/all_block_patching_results.parquet" \
    --output_dir "$RESULTS/module_patching"
}

###############################################################################
# Step 3 — Block patching (includes built-in plotting on flush)
###############################################################################
step_03_block_patching() {
  step 3 "Block patching"
  python src/block_patching.py \
    --csv data/single_block_patching_successes.csv \
    --output_dir "$RESULTS/block_patching" \
    --n_cases "$N_BLOCK"
}

###############################################################################
# Step 4 — Representation tracking (includes built-in plotting)
###############################################################################
step_04_representation_tracking() {
  step 4 "Representation tracking"
  python src/main_paper/representation_tracking.py \
    --dataset data/single_block_patching_successes.csv \
    --output "$RESULTS/representation_tracking"
}

###############################################################################
# Step 5 — Sliding window ablation
###############################################################################
step_05_sliding_window_ablation() {
  step 5 "Sliding window ablation"
  python src/main_paper/final_sliding_window_ablation.py \
    --ablation_csv data/single_block_patching_successes.csv \
    --output_dir "$RESULTS/sliding_window_ablation" \
    --compute_helix \
    --n_sequence_cases "$N_ABLATION" \
    --n_pairwise_cases "$N_ABLATION"
}

###############################################################################
# Step 6 — Charge steering (trains DoM directions, then steers)
###############################################################################
step_06_charge_steering() {
  step 6 "Charge steering"
  python src/charge_steering.py \
    --probing_dataset data/probing_train_test.csv \
    --target_loops_dataset data/target_loops_dataset.csv \
    --output "$RESULTS/charge_steering" \
    --n_cases "$N_STEERING"
}

###############################################################################
# Step 7 — Charge repulsion (reuses directions from step 6)
###############################################################################
step_07_charge_repulsion() {
  step 7 "Charge repulsion"
  python src/charge_repulsion.py \
    --directions "$RESULTS/charge_steering/charge_directions.pt" \
    --hairpin_dataset data/single_block_patching_successes.csv \
    --output "$RESULTS/charge_repulsion" \
    --n_cases "$N_STEERING"
}

###############################################################################
# Step 8 — Contact steering (trains distance probes, then steers)
###############################################################################
step_08_contact_steering() {
  step 8 "Contact steering"
  python src/contact_steering.py \
    --probing_dataset data/probing_train_test.csv \
    --patch_dataset data/single_block_patching_successes.csv \
    --output "$RESULTS/contact_steering" \
    --n_cases "$N_STEERING"
}

###############################################################################
# Step 9a — Bias analysis
###############################################################################
step_09a_bias_analysis() {
  step "9a" "Bias analysis"
  python src/main_paper/bias_analysis2.py \
    --parquet data/all_block_patching_results.parquet \
    --output "$RESULTS/bias_analysis" \
    --n-cases "$N_BIAS"
}

###############################################################################
# Step 9b — Bias plotting
###############################################################################
step_09b_bias_plotting() {
  step "9b" "Bias plotting"
  python src/main_paper/bias_plotting.py \
    --metrics "$RESULTS/bias_analysis/metrics.parquet" \
    --seq_info "$RESULTS/bias_analysis/sequence_info.parquet" \
    --output_dir "$RESULTS/bias_plotting"
}

###############################################################################
# Step 9c — Bias patching
###############################################################################
step_09c_bias_patching() {
  step "9c" "Bias patching"
  python src/main_paper/bias_patching.py \
    --csv data/block_patching_successes.csv \
    --output "$RESULTS/bias_patching" \
    --n-cases "$N_BIAS"
}

###############################################################################
# Step 10 — Z scaling + gradient analysis
###############################################################################
step_10_z_scaling() {
  step 10 "Z scaling + gradient analysis"
  python src/main_paper/z_scaling.py \
    --parquet data/block_patching_successes.csv \
    --output_dir "$RESULTS/z_scaling" \
    --n_cases "$N_SCALING"
}

###############################################################################
# Run all steps — comment out any step to skip it
###############################################################################
step_01_module_patching
step_02_module_plotting
step_03_block_patching
step_04_representation_tracking
step_05_sliding_window_ablation
step_06_charge_steering
step_07_charge_repulsion
step_08_contact_steering
step_09a_bias_analysis
step_09b_bias_plotting
step_09c_bias_patching
step_10_z_scaling

echo -e "\n====== All experiments complete. Results in $RESULTS/ ======\n"
