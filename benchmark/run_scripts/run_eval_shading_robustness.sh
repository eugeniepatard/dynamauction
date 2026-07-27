#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Usage: SCENARIO_ID=saturation-second-price-uniform BEST_ALPHA=0.8 sbatch --job-name=shading_saturation run_eval_shading_robustness.sh
# Run after the alpha sweep (run_shading_sweep_robustness.sh) to determine BEST_ALPHA.

set -e

if [ -z "${SCENARIO_ID}" ]; then
    echo "ERROR: SCENARIO_ID env var must be set"
    exit 1
fi
if [ -z "${BEST_ALPHA}" ]; then
    echo "ERROR: BEST_ALPHA env var must be set (determine from alpha sweep results)"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "Running Shading with alpha=${BEST_ALPHA} for scenario ${SCENARIO_ID}"

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO_ID}" \
    --algos none \
    --baselines none \
    --sim-policies shading \
    --alphas "${BEST_ALPHA}" \
    --n-seeds 1000
