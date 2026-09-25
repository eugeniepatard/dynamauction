#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Timing benchmark: one MPC variant at a time.
# VARIANT must be one of: mpc_h10, mpc_h25, mpcbin_h10, mpcbin_h25
#
# Usage:
#   VARIANT=mpc_h10    sbatch --job-name=timing_mpc_h10    --time=2:00:00 benchmark/run_timing_mpc.sh
#   VARIANT=mpc_h25    sbatch --job-name=timing_mpc_h25    --time=2:00:00 benchmark/run_timing_mpc.sh
#   VARIANT=mpcbin_h10 sbatch --job-name=timing_mpcbin_h10 --time=4:00:00 benchmark/run_timing_mpc.sh
#   VARIANT=mpcbin_h25 sbatch --job-name=timing_mpcbin_h25 --time=4:00:00 benchmark/run_timing_mpc.sh

set -e

if [ -z "${VARIANT}" ]; then
    echo "ERROR: VARIANT must be set (mpc_h10, mpc_h25, mpcbin_h10, mpcbin_h25)"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO="survival-second-price-exponential"
N_SEEDS=100
OUT_DIR="benchmark/benchmark_results/timing"
mkdir -p "${OUT_DIR}"

MLP=$(ls benchmark/models/mlp_valuation_head/${SCENARIO}_h64-64_acttanh_ep500_e500_*.pt 2>/dev/null | head -1)
if [ -z "${MLP}" ]; then
    echo "ERROR: no ep500 MLP found for ${SCENARIO}"
    exit 1
fi
echo "Using MLP: ${MLP}"

case "${VARIANT}" in
    mpc_h10)
        SIM_POLICY="mpc"
        HORIZON=10
        BINS=1
        ;;
    mpc_h25)
        SIM_POLICY="mpc"
        HORIZON=25
        BINS=1
        ;;
    mpcbin_h10)
        SIM_POLICY="mpc-timebin"
        HORIZON=10
        BINS=2
        ;;
    mpcbin_h25)
        SIM_POLICY="mpc-timebin"
        HORIZON=25
        BINS=2
        ;;
    *)
        echo "ERROR: unknown VARIANT=${VARIANT}"
        exit 1
        ;;
esac

python -m benchmark.evaluate \
    --scenario-id "${SCENARIO}" \
    --horizon-eta 50 \
    --algos none \
    --baselines none \
    --sim-policies "${SIM_POLICY}" \
    --mpc-horizons "${HORIZON}" \
    --mpc-n-bins "${BINS}" \
    --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5 \
    --mpc-models "${MLP}" \
    --n-rollouts 10 \
    --n-seeds "${N_SEEDS}"

LATEST=$(ls -t benchmark/benchmark_results/${SCENARIO}_*.csv 2>/dev/null | head -1)
mv "${LATEST}" "${OUT_DIR}/timing_${VARIANT}.csv"
echo "Saved -> ${OUT_DIR}/timing_${VARIANT}.csv"
