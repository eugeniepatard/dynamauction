#!/bin/bash
# Evaluate PPO/SAC/TD3 retrained on the Bernoulli immediate kernel (1000 seeds).
# Submitted as dependency jobs: each algo waits for its own retraining job.
#
# Depends on:
#   419039 — ppo_imm_bern
#   419040 — sac_imm_bern
#   419041 — td3_imm_bern
#
# Models read from:
#   /scratch/epatard/dynamicauc/models/<algo>/survival-second-price-exponential-immediate/squashing_clip/best_model.zip
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_eval_immediate_bern.sh

set -e

SCENARIO="survival-second-price-exponential-immediate"
RUN_NAME="${SCENARIO}/squashing_clip"

echo "=== PPO eval (depends on job 419039) ==="
SCENARIO_ID="${SCENARIO}" \
ALGO=ppo \
RUN_NAME="${RUN_NAME}" \
    sbatch --job-name=eval_ppo_imm_bern \
           --dependency=afterok:419039 \
           --time=4:00:00 \
           benchmark/run_scripts/run_eval_rl_robustness.sh

echo ""
echo "=== SAC eval (depends on job 419040) ==="
SCENARIO_ID="${SCENARIO}" \
ALGO=sac \
RUN_NAME="${RUN_NAME}" \
    sbatch --job-name=eval_sac_imm_bern \
           --dependency=afterok:419040 \
           --time=4:00:00 \
           benchmark/run_scripts/run_eval_rl_robustness.sh

echo ""
echo "=== TD3 eval (depends on job 419041) ==="
SCENARIO_ID="${SCENARIO}" \
ALGO=td3 \
RUN_NAME="${RUN_NAME}" \
    sbatch --job-name=eval_td3_imm_bern \
           --dependency=afterok:419041 \
           --time=4:00:00 \
           benchmark/run_scripts/run_eval_rl_robustness.sh

echo ""
echo "Dependency eval jobs submitted:"
echo "  eval_ppo_imm_bern → afterok:419039"
echo "  eval_sac_imm_bern → afterok:419040"
echo "  eval_td3_imm_bern → afterok:419041"
echo ""
echo "Download results once complete:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ benchmark/benchmark_results/"
