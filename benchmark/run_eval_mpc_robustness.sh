#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Usage: SCENARIO_ID=saturation-second-price-uniform sbatch --job-name=mpc_saturation run_eval_mpc_robustness.sh
# Evaluates MPC(H=10), MPC(H=25), MPCTimeBinned(H=10,K=2), MPCTimeBinned(H=25,K=2)
# Requires the MLP valuation head to exist in benchmark/models/mlp_valuation_head/

set -e

if [ -z "${SCENARIO_ID}" ]; then
    echo "ERROR: SCENARIO_ID env var must be set"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

MLP=$(ls benchmark/models/mlp_valuation_head/${SCENARIO_ID}_h64-64_acttanh_ep500_e500_*.pt 2>/dev/null | head -1)
if [ -z "$MLP" ]; then
    echo "ERROR: no ep500_e500 MLP found for ${SCENARIO_ID} in benchmark/models/mlp_valuation_head/"
    echo "Available files:"
    ls benchmark/models/mlp_valuation_head/${SCENARIO_ID}_*.pt 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "Using MLP: $MLP"

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO_ID}" \
    --algos none \
    --baselines none \
    --sim-policies mpc mpc-timebin \
    --n-rollouts 10 \
    --mpc-horizons 10 25 \
    --mpc-n-bins 2 \
    --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5 \
    --mpc-models "${MLP}" \
    --n-seeds 500
