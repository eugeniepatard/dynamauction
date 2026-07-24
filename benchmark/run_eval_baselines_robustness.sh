#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Usage: SCENARIO_ID=saturation-second-price-uniform sbatch --job-name=baselines_saturation run_eval_baselines_robustness.sh

set -e

if [ -z "${SCENARIO_ID}" ]; then
    echo "ERROR: SCENARIO_ID env var must be set"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO_ID}" \
    --algos none \
    --baselines fipla \
    --sim-policies greedy \
    --n-seeds 500
