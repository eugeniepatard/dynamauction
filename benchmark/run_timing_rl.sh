#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Timing benchmark: RL inference for one algorithm.
# ALGO must be one of: ppo, sac, td3
#
# Usage:
#   ALGO=ppo sbatch --job-name=timing_ppo benchmark/run_timing_rl.sh
#   ALGO=sac sbatch --job-name=timing_sac benchmark/run_timing_rl.sh
#   ALGO=td3 sbatch --job-name=timing_td3 benchmark/run_timing_rl.sh

set -e

if [ -z "${ALGO}" ]; then
    echo "ERROR: ALGO must be set (ppo, sac, td3)"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

N_SEEDS=100
OUT_DIR="benchmark/benchmark_results/timing"
mkdir -p "${OUT_DIR}"

# Find the first available model for this algo and derive the scenario from its path.
# We only need inference latency — the model must run on the scenario it was trained on.
MODEL_ZIP=$(ls /scratch/epatard/dynamicauc/models/${ALGO}/*/best_model.zip 2>/dev/null | head -1)
if [ -z "${MODEL_ZIP}" ]; then
    echo "ERROR: no trained ${ALGO} model found in /scratch/epatard/dynamicauc/models/${ALGO}/"
    exit 1
fi
MODEL="${MODEL_ZIP%.zip}"
# Extract scenario name from path: .../models/<algo>/<scenario>/best_model.zip
SCENARIO=$(basename "$(dirname "${MODEL_ZIP}")")
echo "Using model: ${MODEL}  (scenario: ${SCENARIO})"

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO}" \
    --algos "${ALGO}" \
    --${ALGO}-model "${MODEL}" \
    --baselines none \
    --sim-policies none \
    --n-seeds "${N_SEEDS}"

LATEST=$(ls -t benchmark/benchmark_results/${SCENARIO}_*.csv 2>/dev/null | head -1)
mv "${LATEST}" "${OUT_DIR}/timing_${ALGO}.csv"
echo "Saved -> ${OUT_DIR}/timing_${ALGO}.csv"
