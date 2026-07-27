#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Usage: SCENARIO_ID=saturation-second-price-uniform ALGO=ppo sbatch --job-name=ppo_saturation run_eval_rl_robustness.sh
# Run after RL training completes (run_train_rl_robustness.sh).
# ALGO must be one of: ppo, sac, td3

set -e

if [ -z "${SCENARIO_ID}" ]; then
    echo "ERROR: SCENARIO_ID env var must be set"
    exit 1
fi
if [ -z "${ALGO}" ]; then
    echo "ERROR: ALGO env var must be set (ppo, sac, or td3)"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

# Models saved by run_train_rl_robustness.sh go to scratch; fall back to home for legacy runs
# RUN_NAME overrides the model subdirectory (default: SCENARIO_ID). Use when retraining with a
# different run_name (e.g. saturation-second-price-uniform-eta50-v2).
RUN_SUBDIR="${RUN_NAME:-${SCENARIO_ID}}"
SCRATCH_MODEL=/scratch/epatard/dynamicauc/models/${ALGO}/${RUN_SUBDIR}/best_model
HOME_MODEL=benchmark/models/${ALGO}/${RUN_SUBDIR}/best_model
if [ -f "${SCRATCH_MODEL}.zip" ]; then
    MODEL="${SCRATCH_MODEL}"
elif [ -f "${HOME_MODEL}.zip" ]; then
    MODEL="${HOME_MODEL}"
else
    echo "ERROR: model not found at ${SCRATCH_MODEL}.zip or ${HOME_MODEL}.zip"
    exit 1
fi
echo "Using model: ${MODEL}"

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO_ID}" \
    --algos "${ALGO}" \
    --${ALGO}-model "${MODEL}" \
    --baselines none \
    --sim-policies none \
    --n-seeds 1000
