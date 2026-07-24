#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Usage: ALGO=ppo SCENARIO_ID=survival-second-price-exponential-immediate \
#          sbatch --job-name=rl_ppo_immediate benchmark/run_train_rl_robustness.sh
# ALGO must be one of: ppo, sac, td3
# Trains for 10M timesteps (robustness budget). Use training.total_timesteps=30000000
# to match the main survival experiment budget.

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

echo "Training ${ALGO} on ${SCENARIO_ID} for 10M timesteps"

python -m benchmark.train_rl \
    algo="${ALGO}" \
    scenario_id="${SCENARIO_ID}" \
    training.total_timesteps=10000000
