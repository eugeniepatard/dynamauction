#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Sweep n_rollouts and measure avg bid latency for one policy variant.
# VARIANT must be one of: shading, mpc_h10, mpc_h25, mpcbin_h10, mpcbin_h25
#
# Usage (time limits chosen conservatively per variant):
#   VARIANT=shading    sbatch --job-name=nr_shading    --time=4:00:00  benchmark/run_timing_nrollouts.sh
#   VARIANT=mpc_h10    sbatch --job-name=nr_mpc_h10    --time=4:00:00  benchmark/run_timing_nrollouts.sh
#   VARIANT=mpc_h25    sbatch --job-name=nr_mpc_h25    --time=6:00:00  benchmark/run_timing_nrollouts.sh
#   VARIANT=mpcbin_h10 sbatch --job-name=nr_mpcbin_h10 --time=8:00:00  benchmark/run_timing_nrollouts.sh
#   VARIANT=mpcbin_h25 sbatch --job-name=nr_mpcbin_h25 --time=12:00:00 benchmark/run_timing_nrollouts.sh

set -e

if [ -z "${VARIANT}" ]; then
    echo "ERROR: VARIANT must be set (shading, mpc_h10, mpc_h25, mpcbin_h10, mpcbin_h25)"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO="survival-second-price-exponential"
N_SEEDS=30
ROLLOUTS_LIST="1 5 10 25 50"
OUT_DIR="benchmark/benchmark_results/timing/nrollouts"
mkdir -p "${OUT_DIR}"

# Resolve policy-specific flags
case "${VARIANT}" in
    shading)
        SIM_POLICY="shading"
        EXTRA_FLAGS="--alphas 0.7"
        MLP_FLAGS=""
        ;;
    mpc_h10)
        SIM_POLICY="mpc"
        EXTRA_FLAGS="--mpc-horizons 10 --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5"
        ;;
    mpc_h25)
        SIM_POLICY="mpc"
        EXTRA_FLAGS="--mpc-horizons 25 --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5"
        ;;
    mpcbin_h10)
        SIM_POLICY="mpc-timebin"
        EXTRA_FLAGS="--mpc-horizons 10 --mpc-n-bins 2 --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5"
        ;;
    mpcbin_h25)
        SIM_POLICY="mpc-timebin"
        EXTRA_FLAGS="--mpc-horizons 25 --mpc-n-bins 2 --mpc-alpha-candidates 0.6 0.8 1.0 1.2 1.5"
        ;;
    *)
        echo "ERROR: unknown VARIANT=${VARIANT}"
        exit 1
        ;;
esac

# Resolve MLP model (needed for MPC variants, ignored for shading)
MLP_FLAGS=""
if [ "${VARIANT}" != "shading" ]; then
    MLP=$(ls benchmark/models/mlp_valuation_head/${SCENARIO}_h64-64_acttanh_ep500_e500_*.pt 2>/dev/null | head -1)
    if [ -z "${MLP}" ]; then
        echo "ERROR: no ep500 MLP found for ${SCENARIO}"
        exit 1
    fi
    MLP_FLAGS="--mpc-models ${MLP}"
    echo "Using MLP: ${MLP}"
fi

for NR in ${ROLLOUTS_LIST}; do
    echo "=== VARIANT=${VARIANT}  n_rollouts=${NR}  seeds=${N_SEEDS} ==="

    CSV_PATH=$(python -m benchmark.evaluate \
        --scenario-id "${SCENARIO}" \
        --horizon-eta 50 \
        --algos none \
        --baselines none \
        --sim-policies "${SIM_POLICY}" \
        --n-rollouts "${NR}" \
        ${EXTRA_FLAGS} \
        ${MLP_FLAGS} \
        --n-seeds "${N_SEEDS}" 2>/dev/null \
      | grep "Results saved ->" | tail -1 | awk '{print $NF}')

    if [ -z "${CSV_PATH}" ] || [ ! -f "${CSV_PATH}" ]; then
        echo "ERROR: could not find output CSV for n_rollouts=${NR}"
        exit 1
    fi

    # Tag with n_rollouts column
    python3 -c "
import pandas as pd
df = pd.read_csv('${CSV_PATH}')
df.insert(2, 'n_rollouts', ${NR})
df.to_csv('${CSV_PATH}', index=False)
"
    DEST="${OUT_DIR}/${VARIANT}_nr${NR}.csv"
    mv "${CSV_PATH}" "${DEST}"
    echo "Saved -> ${DEST}"
done

# Merge all per-rollouts CSVs into one sensitivity file
MERGED="${OUT_DIR}/${VARIANT}_nrollouts.csv"
python3 -c "
import pandas as pd, glob
files = sorted(glob.glob('${OUT_DIR}/${VARIANT}_nr*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('${MERGED}', index=False)
# Summary: avg bid latency per n_rollouts
if 'bid_time_s' in df.columns:
    print(df.groupby('n_rollouts')['bid_time_s'].mean().mul(1000).rename('avg_bid_ms').to_string())
print(f'Merged {len(files)} files -> ${MERGED}')
"
echo "Done: ${VARIANT}"
