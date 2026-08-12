#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc

cd ~/projects/dynamicauc

python -m benchmark.evaluate     --scenario-id "${SCENARIO_ID}"     --algos none     --baselines none     --sim-policies shading     --n-rollouts 10     --alphas 0.25 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5 1.75 2.0     --n-seeds 200
