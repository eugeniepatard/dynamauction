#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc

cd ~/projects/dynamicauc

# MLP: use the ep500_e500 version
MLP=$(ls benchmark/models/mlp_valuation_head/${SCENARIO_ID}_h64-64_acttanh_ep500_e500_*.pt 2>/dev/null | head -1)
if [ -z "$MLP" ]; then
  echo "ERROR: no ep500_e500 MLP found for ${SCENARIO_ID}" >&2; exit 1
fi
echo "Using MLP: $MLP"

# RL model paths (saved by train_rl.py under benchmark/models/<algo>/<scenario_id>/)
PPO_MODEL=benchmark/models/ppo/${SCENARIO_ID}/best_model
SAC_MODEL=benchmark/models/sac/${SCENARIO_ID}/best_model
TD3_MODEL=benchmark/models/td3/${SCENARIO_ID}/best_model

python -m benchmark.evaluate     --scenario-id "${SCENARIO_ID}"     --algos ppo sac td3     --ppo-model "${PPO_MODEL}"     --sac-model "${SAC_MODEL}"     --td3-model "${TD3_MODEL}"     --baselines fipla     --sim-policies greedy shading mpc mpc-timebin     --n-rollouts 10     --alphas 1.0     --mpc-horizons 10 25     --mpc-n-bins 2     --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5     --mpc-models "${MLP}"     --n-seeds 500
