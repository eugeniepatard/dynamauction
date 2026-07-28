#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc

cd ~/projects/dynamicauc

python -m benchmark.train_mlp     --scenario-id "${SCENARIO_ID}"     --n-episodes "${N_EPISODES:-500}"     --n-epochs 500
