#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc

cd ~/projects/dynamicauc

MLP=/scratch/epatard/mlp_valuation_head/squashing=sigmoid/survival-second-price-exponential_h64-64_acttanh_ep10000_e500_lr0.001_s42.pt
PPO=/scratch/epatard/models/ppo/best_ppo_squashing=sigmoid_eta50/best_model
SAC=/scratch/epatard/models/sac/best_sac_squashing=sigmoid_eta50/best_model
TD3=/scratch/epatard/models/td3/best_td3_squashing=sigmoid_eta50/best_model

python -m benchmark.evaluate \
    --squashing sigmoid \
    --horizon-eta 50 \
    --n-seeds 1000 \
    --constant-bid 0.4737 \
    --alphas 0.89 \
    --mpc-models "$MLP" \
    --ppo-model "$PPO" \
    --sac-model "$SAC" \
    --td3-model "$TD3" \
    $EVAL_ARGS
