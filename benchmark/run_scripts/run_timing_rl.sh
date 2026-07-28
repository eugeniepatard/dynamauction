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

N_SEEDS=1000
OUT_DIR="benchmark/benchmark_results/timing"
mkdir -p "${OUT_DIR}"

SCENARIO="survival-second-price-exponential"

# Best-model paths for the survival scenario (η=50).
# Using explicit paths to avoid glob depth mismatch (models/<algo>/<scenario>/<run_name>/).
case "${ALGO}" in
    ppo) MODEL="benchmark/models/ppo/squashing=clip/eta50/lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5/best_model" ;;
    sac) MODEL="benchmark/models/sac/squashing=clip/eta50/lr=5e-05_bs=512_warmup=500_gamma=0.95/best_model" ;;
    td3) MODEL="benchmark/models/td3/squashing=clip/eta50/lr=0.0001_bs=512_noise=0.1_warmup=500_gamma=0.99/best_model" ;;
    *)   echo "ERROR: unknown ALGO=${ALGO}"; exit 1 ;;
esac

if [ ! -f "${MODEL}.zip" ]; then
    echo "ERROR: model not found at ${MODEL}.zip"
    exit 1
fi
echo "Using model: ${MODEL}"
echo "Using model: ${MODEL}  (scenario: ${SCENARIO})"

# Capture the exact output CSV path from evaluate.py stdout to avoid race conditions
CSV_PATH=$(python -m benchmark.evaluate \
    --scenario-id "${SCENARIO}" \
    --horizon-eta 50 \
    --algos "${ALGO}" \
    --${ALGO}-model "${MODEL}" \
    --baselines none \
    --sim-policies none \
    --n-seeds "${N_SEEDS}" 2>/dev/null \
  | grep "Results saved ->" | tail -1 | awk '{print $NF}')

if [ -z "${CSV_PATH}" ] || [ ! -f "${CSV_PATH}" ]; then
    echo "ERROR: could not determine output CSV path"
    exit 1
fi
mv "${CSV_PATH}" "${OUT_DIR}/timing_${ALGO}.csv"
echo "Saved -> ${OUT_DIR}/timing_${ALGO}.csv"
