#!/bin/bash
# Retrain PPO for 10M steps using the best hyperparameters found by the grid search.
#
# Recency  — trial_013: lr=3e-4, ent_coef=0.01, n_steps=1024, gae_lambda=0.95
# Mixed    — trial_003: lr=1e-4, ent_coef=0.01, n_steps=1024, gae_lambda=0.9
#
# Models saved to:
#   /scratch/epatard/dynamicauc/models/ppo/<scenario>/eta50_hparam/best_model.zip
#
# After training, run the evals:
#   bash benchmark/run_scripts/submit_ppo_eval_after_retrain.sh
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_retrain_hparam.sh

set -e

echo "=== Recency PPO retrain (trial_013: lr=3e-4, ent=0.01, n_steps=1024, gae=0.95) ==="
ALGO=ppo \
SCENARIO_ID=recency-second-price-lognormal-eta50 \
EXTRA_ARGS="algo.learning_rate=0.0003 algo.n_steps=1024 algo.gae_lambda=0.95 algo.ent_coef=0.01 training.run_name=recency-second-price-lognormal-eta50/eta50_hparam" \
    sbatch --job-name=ppo_retrain_recency --time=16:00:00 \
    benchmark/run_scripts/run_train_rl_robustness.sh

echo ""
echo "=== Mixed PPO retrain (trial_003: lr=1e-4, ent=0.01, n_steps=1024, gae=0.9) ==="
ALGO=ppo \
SCENARIO_ID=mixed-second-price-exponential-eta50 \
EXTRA_ARGS="algo.learning_rate=0.0001 algo.n_steps=1024 algo.gae_lambda=0.9 algo.ent_coef=0.01 training.run_name=mixed-second-price-exponential-eta50/eta50_hparam" \
    sbatch --job-name=ppo_retrain_mixed --time=16:00:00 \
    benchmark/run_scripts/run_train_rl_robustness.sh

echo ""
echo "Both training jobs submitted. Once done, run:"
echo "  bash benchmark/run_scripts/submit_ppo_eval_after_retrain.sh"
