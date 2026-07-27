#!/bin/bash
# Evaluate the v2 PPO and TD3 models for saturation-second-price-uniform-eta50.
# Run this on Margaret AFTER submit_retrain_saturation_eta50.sh jobs complete.

SCENARIO_ID=saturation-second-price-uniform-eta50
RUN_NAME="${SCENARIO_ID}-v2"

for ALGO in ppo td3; do
    echo "Submitting eval for ${ALGO} (v2)..."
    SCENARIO_ID="${SCENARIO_ID}" \
    ALGO="${ALGO}" \
    RUN_NAME="${RUN_NAME}" \
        sbatch --job-name="eval_${ALGO}_sat50_v2" benchmark/run_eval_rl_robustness.sh
    sleep 5
done

echo "Submitted 2 eval jobs."
