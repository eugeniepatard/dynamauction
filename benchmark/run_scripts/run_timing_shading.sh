#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Timing benchmark: Shading policy (best alpha=0.7 for survival scenario).
# Usage: sbatch --job-name=timing_shading benchmark/run_timing_shading.sh

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
    --baselines none \
    --sim-policies shading \
    --alphas 0.7 \
    --n-rollouts 10 \
    --n-seeds "${N_SEEDS}"

LATEST=$(ls -t benchmark/benchmark_results/${SCENARIO}_*.csv 2>/dev/null | head -1)
mv "${LATEST}" "${OUT_DIR}/timing_shading.csv"
echo "Saved -> ${OUT_DIR}/timing_shading.csv"
