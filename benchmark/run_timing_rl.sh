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

SCENARIO="survival-second-price-exponential"
N_SEEDS=100
OUT_DIR="benchmark/benchmark_results/timing"
mkdir -p "${OUT_DIR}"

SCRATCH_MODEL=/scratch/epatard/dynamicauc/models/${ALGO}/${SCENARIO}/best_model
HOME_MODEL=benchmark/models/${ALGO}/${SCENARIO}/best_model
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
    --scenario-id "${SCENARIO}" \
    --horizon-eta 50 \
    --algos "${ALGO}" \
    --${ALGO}-model "${MODEL}" \
    --baselines none \
    --sim-policies none \
    --n-seeds "${N_SEEDS}"

LATEST=$(ls -t benchmark/benchmark_results/${SCENARIO}_*.csv 2>/dev/null | head -1)
mv "${LATEST}" "${OUT_DIR}/timing_${ALGO}.csv"
echo "Saved -> ${OUT_DIR}/timing_${ALGO}.csv"
