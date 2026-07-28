#!/bin/bash
# Run this on Margaret AFTER ppo_best_rec retrain job completes.
# Evaluates the best-hparam PPO on recency-second-price-lognormal-eta50, 1000 seeds.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_eval_1000seeds_recency.sh

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO_ID="recency-second-price-lognormal-eta50"
RUN_NAME="ppo_best_recency"

SCENARIO_ID="${SCENARIO_ID}" \
ALGO=ppo \
RUN_NAME="${RUN_NAME}" \
    sbatch --job-name=eval_ppo_best_rec benchmark/run_scripts/run_eval_rl_robustness.sh

echo "Submitted 1000-seed eval for ${SCENARIO_ID} (model: ${RUN_NAME})"
echo "Download results once done:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ benchmark/benchmark_results/"
