#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-134%70
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%A_%a.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%A_%a.err

# PPO hyperparameter grid search — survival-second-price-exponential-immediate (eta=50)
#
# 135 trials covering:
#   ent_coef      [0.01, 0.05, 0.1, 0.2, 0.5]
#   learning_rate [1e-4, 3e-4, 1e-3]
#   n_steps       [512, 1024, 2048]
#   gae_lambda    [0.90, 0.95, 0.99]
#
# Submit:
#   sbatch benchmark/run_scripts/run_ppo_hparam_search_immediate.sh
#
# After all jobs finish, download results:
#   rsync -e ssh -av margaret02:/scratch/epatard/dynamicauc/ppo_hparam_search_immediate/ \
#         benchmark/results/ppo_hparam_search_immediate/

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== PPO hparam search (immediate): trial ${SLURM_ARRAY_TASK_ID} / 134 ==="
echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}"

export SCRATCH_DIR=/scratch/epatard/dynamicauc

python benchmark/ppo_hparam_search_immediate.py
