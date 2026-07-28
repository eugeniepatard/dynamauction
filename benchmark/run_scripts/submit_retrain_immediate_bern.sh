#!/bin/bash
# Retrain PPO, SAC, TD3 on the corrected Bernoulli immediate kernel
# (survival-second-price-exponential-immediate with Bernoulli fix applied 2026-07-27).
#
# Uses the same best hyperparameters as the survival-clip models:
#   PPO: lr=3e-4, ent=0.02, n_steps=512, gamma=0.95, n_epochs=5
#   SAC: lr=5e-5, bs=512,  learning_starts=500, gamma=0.95
#   TD3: lr=1e-4, bs=512,  noise=0.1, learning_starts=500, gamma=0.99
#
# Models saved to:
#   /scratch/epatard/dynamicauc/models/<algo>/survival-second-price-exponential-immediate/squashing=clip/
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_retrain_immediate_bern.sh

set -e

SCENARIO="survival-second-price-exponential-immediate"
RUN_NAME="${SCENARIO}/squashing_clip"

echo "=== PPO — Bernoulli immediate retraining ==="
ALGO=ppo \
SCENARIO_ID="${SCENARIO}" \
EXTRA_ARGS="algo.learning_rate=0.0003 algo.ent_coef=0.02 algo.n_steps=512 algo.gamma=0.95 algo.n_epochs=5 training.run_name=${RUN_NAME}" \
    sbatch --job-name=ppo_imm_bern --time=4:00:00 \
    benchmark/run_scripts/run_train_rl_robustness.sh

echo ""
echo "=== SAC — Bernoulli immediate retraining ==="
ALGO=sac \
SCENARIO_ID="${SCENARIO}" \
EXTRA_ARGS="algo.learning_rate=5e-05 algo.batch_size=512 algo.learning_starts=500 algo.gamma=0.95 training.run_name=${RUN_NAME}" \
    sbatch --job-name=sac_imm_bern --time=12:00:00 \
    benchmark/run_scripts/run_train_rl_robustness.sh

echo ""
echo "=== TD3 — Bernoulli immediate retraining ==="
ALGO=td3 \
SCENARIO_ID="${SCENARIO}" \
EXTRA_ARGS="algo.learning_rate=0.0001 algo.batch_size=512 algo.action_noise_sigma=0.1 algo.learning_starts=500 algo.gamma=0.99 training.run_name=${RUN_NAME}" \
    sbatch --job-name=td3_imm_bern --time=12:00:00 \
    benchmark/run_scripts/run_train_rl_robustness.sh

echo ""
echo "All three jobs submitted. Models will be saved to:"
echo "  /scratch/epatard/dynamicauc/models/{ppo,sac,td3}/${RUN_NAME}/"
