#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Timing benchmark: FiPla + GreedyMyopic (myopic baselines).
# Usage: sbatch --job-name=timing_baselines benchmark/run_timing_baselines.sh

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO="survival-second-price-exponential"
N_SEEDS=100
OUT_DIR="benchmark/benchmark_results/timing"
mkdir -p "${OUT_DIR}"

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO}" \
    --horizon-eta 50 \
    --algos none \
    --baselines fipla greedy \
    --sim-policies none \
    --n-seeds "${N_SEEDS}"

LATEST=$(ls -t benchmark/benchmark_results/${SCENARIO}_*.csv 2>/dev/null | head -1)
mv "${LATEST}" "${OUT_DIR}/timing_baselines.csv"
echo "Saved -> ${OUT_DIR}/timing_baselines.csv"
