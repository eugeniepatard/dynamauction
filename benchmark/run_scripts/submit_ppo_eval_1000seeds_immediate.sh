#!/bin/bash
# Run this on Margaret AFTER ppo_best_imm retrain job completes.
# Evaluates the best-hparam PPO on survival-second-price-exponential-immediate, 1000 seeds.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_eval_1000seeds_immediate.sh

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO_ID="survival-second-price-exponential-immediate"
RUN_NAME="ppo_best_immediate"

SCENARIO_ID="${SCENARIO_ID}" \
ALGO=ppo \
RUN_NAME="${RUN_NAME}" \
EXTRA_EVAL_ARGS="--horizon-eta 50" \
    sbatch --job-name=eval_ppo_best_imm benchmark/run_scripts/run_eval_rl_robustness.sh

echo "Submitted 1000-seed eval for ${SCENARIO_ID} (model: ${RUN_NAME})"
echo "Download results once done:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ benchmark/benchmark_results/"
