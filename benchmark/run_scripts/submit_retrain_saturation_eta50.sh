#!/bin/bash
# Retrain PPO and TD3 on saturation-second-price-uniform-eta50 with better exploration.
#
# Root cause: both policies converged to always-bid-0 (degenerate policy).
# PPO already had ent_coef=0.01 — not enough for sparse saturation rewards.
# TD3 had action_noise_sigma=0.1 — too small to escape the zero-bid trap.
#
# Fixes (saved to *-v2 subdir so old models are preserved):
#   PPO: ent_coef 0.01 → 0.05  (5× entropy bonus to force bid diversity)
#   TD3: action_noise_sigma 0.1 → 0.5  (5× larger exploration noise)
#
# Run this on Margaret: bash benchmark/submit_retrain_saturation_eta50.sh

SCENARIO_ID=saturation-second-price-uniform-eta50
RUN_NAME="${SCENARIO_ID}-v2"

echo "Submitting PPO retrain (ent_coef=0.05)..."
ALGO=ppo \
SCENARIO_ID="${SCENARIO_ID}" \
EXTRA_ARGS="algo.ent_coef=0.05 training.run_name=${RUN_NAME}" \
    sbatch --job-name=rl_ppo_sat50_v2 benchmark/run_train_rl_robustness.sh

sleep 5

echo "Submitting TD3 retrain (action_noise_sigma=0.5)..."
ALGO=td3 \
SCENARIO_ID="${SCENARIO_ID}" \
EXTRA_ARGS="algo.action_noise_sigma=0.5 training.run_name=${RUN_NAME}" \
    sbatch --job-name=rl_td3_sat50_v2 benchmark/run_train_rl_robustness.sh

echo ""
echo "Submitted 2 jobs.  After both complete (~10h), run:"
echo "  bash benchmark/submit_eval_saturation_eta50_v2.sh"
