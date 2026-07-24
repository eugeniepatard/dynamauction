#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Train MLP with N_EPISODES training episodes, then evaluate MPC variants using that MLP.
# Usage: N_EPISODES=50 sbatch --job-name=mlp_sens_ep50 benchmark/run_mlp_sensitivity.sh

set -e

N_EPISODES="${N_EPISODES:-500}"
SCENARIO_ID="survival-second-price-exponential"
ETA=50

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

# Step 1: Train MLP
echo "=== Training MLP: n_episodes=${N_EPISODES}, scenario=${SCENARIO_ID} ==="
python -m benchmark.train_mlp \
    --scenario-id "${SCENARIO_ID}" \
    --n-episodes "${N_EPISODES}" \
    --n-epochs 500 \
    --hidden-dims 64 64 \
    --activation tanh \
    --seed 42

MLP_STEM="${SCENARIO_ID}_h64-64_acttanh_ep${N_EPISODES}_e500_lr0.001_s42"
MLP="benchmark/models/mlp_valuation_head/${MLP_STEM}.pt"
if [ ! -f "${MLP}" ]; then
    echo "ERROR: expected MLP not found at ${MLP}"
    exit 1
fi
echo "MLP: ${MLP}"

# Step 2: Evaluate MPC variants with this MLP
echo "=== Evaluating MPC with ep${N_EPISODES} MLP ==="
python -m benchmark.evaluate \
    --horizon-eta "${ETA}" \
    --algos none \
    --baselines none \
    --sim-policies mpc mpc-timebin \
    --n-rollouts 10 \
    --mpc-horizons 10 25 \
    --mpc-n-bins 2 \
    --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5 \
    --mpc-models "${MLP}" \
    --n-seeds 500

# Step 3: Move output CSV into sensitivity subdirectory with n_episodes in the name
RESULTS_DIR="benchmark/benchmark_results"
SENS_DIR="${RESULTS_DIR}/mlp_sensitivity"
mkdir -p "${SENS_DIR}"

LATEST_CSV=$(ls -t "${RESULTS_DIR}/${SCENARIO_ID}_eta${ETA}_"*.csv 2>/dev/null | head -1)
if [ -z "${LATEST_CSV}" ]; then
    echo "ERROR: no output CSV found in ${RESULTS_DIR}"
    exit 1
fi

DEST="${SENS_DIR}/${SCENARIO_ID}_eta${ETA}_ep${N_EPISODES}.csv"
mv "${LATEST_CSV}" "${DEST}"
echo "Results -> ${DEST}"
