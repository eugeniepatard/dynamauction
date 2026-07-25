#!/bin/bash
# Submit RL evaluation jobs for all 7 trained scenarios × 3 algorithms = 21 jobs.
# Run this on Margaret: bash benchmark/submit_rl_evals.sh

SCENARIOS=(
    mixed-second-price-exponential
    mixed-second-price-exponential-eta50
    recency-second-price-lognormal
    recency-second-price-lognormal-eta50
    saturation-second-price-uniform
    saturation-second-price-uniform-eta50
    survival-second-price-exponential-immediate
)

ALGOS=(ppo sac td3)

for SCENARIO in "${SCENARIOS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        SHORT=$(echo "${SCENARIO}" | sed 's/second-price-//' | sed 's/exponential/exp/' | sed 's/lognormal/logn/' | sed 's/uniform/unif/')
        JOB_NAME="${ALGO}_${SHORT}"
        echo "Submitting: ${JOB_NAME}"
        SCENARIO_ID="${SCENARIO}" ALGO="${ALGO}" \
            sbatch --job-name="${JOB_NAME}" benchmark/run_eval_rl_robustness.sh
        sleep 5  # stagger submissions so evaluate.py timestamps don't collide
    done
done

echo "Submitted $((${#SCENARIOS[@]} * ${#ALGOS[@]})) jobs."
