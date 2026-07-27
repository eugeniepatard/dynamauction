#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-134%70
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%A_%a.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%A_%a.err

# PPO hyperparameter grid search — saturation-second-price-uniform-eta50
#
# 135 trials covering:
#   ent_coef      [0.01, 0.05, 0.1, 0.2, 0.5]
#   learning_rate [1e-4, 3e-4, 1e-3]
#   n_steps       [512, 1024, 2048]
#   gae_lambda    [0.90, 0.95, 0.99]
#
# Each trial trains PPO for 1M steps, then checks if the policy escaped
# the zero-bid degenerate optimum (mean_bid, frac_nonzero, mean_reward).
#
# Submit (from ~/projects/dynamicauc on Margaret):
#   sbatch benchmark/run_scripts/run_ppo_hparam_search.sh
#
# After all jobs finish, download results:
#   rsync -e ssh -av margaret02:/scratch/epatard/dynamicauc/ppo_hparam_search/ \
#         benchmark/results/ppo_hparam_search/

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== PPO hparam search: trial ${SLURM_ARRAY_TASK_ID} / 134 ==="
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Array:  job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"

export SCRATCH_DIR=/scratch/epatard/dynamicauc

python benchmark/ppo_hparam_search.py
