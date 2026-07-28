#!/bin/bash
# Submit 1000-seed evals for the retrained PPO models (10M steps, best hparams).
# Run this after submit_ppo_retrain_hparam.sh jobs complete.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_eval_after_retrain.sh

set -e

echo "=== Recency PPO eval (eta50_hparam, lr=3e-4, ent=0.01, n_steps=1024, gae=0.95) ==="
SCENARIO_ID=recency-second-price-lognormal-eta50 \
ALGO=ppo \
RUN_NAME=recency-second-price-lognormal-eta50/eta50_hparam \
    sbatch --job-name=eval_ppo_recency_hparam \
    benchmark/run_scripts/run_eval_rl_robustness.sh

echo ""
echo "=== Mixed PPO eval (eta50_hparam, lr=1e-4, ent=0.01, n_steps=1024, gae=0.9) ==="
SCENARIO_ID=mixed-second-price-exponential-eta50 \
ALGO=ppo \
RUN_NAME=mixed-second-price-exponential-eta50/eta50_hparam \
    sbatch --job-name=eval_ppo_mixed_hparam \
    benchmark/run_scripts/run_eval_rl_robustness.sh

echo ""
echo "Both eval jobs submitted. Download results with:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ benchmark/benchmark_results/"
