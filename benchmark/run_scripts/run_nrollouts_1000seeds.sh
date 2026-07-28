#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# n_rollouts sensitivity at 1000 seeds — one job per (VARIANT, NR) pair.
#
# Submit one job per rollout count (parallel):
#   VARIANT=mpc_h10 NR=1  sbatch --job-name=nr1000_mpc_h10_nr1  --time=0:30:00 benchmark/run_scripts/run_nrollouts_1000seeds.sh
#   VARIANT=mpc_h10 NR=5  sbatch --job-name=nr1000_mpc_h10_nr5  --time=1:30:00 benchmark/run_scripts/run_nrollouts_1000seeds.sh
#   VARIANT=mpc_h10 NR=10 sbatch --job-name=nr1000_mpc_h10_nr10 --time=3:00:00 benchmark/run_scripts/run_nrollouts_1000seeds.sh
#   VARIANT=mpc_h10 NR=25 sbatch --job-name=nr1000_mpc_h10_nr25 --time=6:00:00 benchmark/run_scripts/run_nrollouts_1000seeds.sh
#   VARIANT=mpc_h10 NR=50 sbatch --job-name=nr1000_mpc_h10_nr50 --time=8:00:00 benchmark/run_scripts/run_nrollouts_1000seeds.sh
#
# Results saved to:
#   /scratch/epatard/dynamicauc/nrollouts_1000seeds/${VARIANT}_nr${NR}.csv
#
# Download:
#   rsync -e ssh -av margaret02:/scratch/epatard/dynamicauc/nrollouts_1000seeds/ \
#         benchmark/benchmark_results/timing/nrollouts_1000seeds/

set -e

if [ -z "${VARIANT}" ] || [ -z "${NR}" ]; then
    echo "ERROR: both VARIANT and NR must be set"
    echo "  VARIANT: mpc_h10 | mpc_h25 | mpcbin_h10 | mpcbin_h25 | shading"
    echo "  NR:      1 | 5 | 10 | 25 | 50"
    exit 1
fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO="survival-second-price-exponential"
N_SEEDS=1000
OUT_DIR="/scratch/epatard/dynamicauc/nrollouts_1000seeds"
mkdir -p "${OUT_DIR}"

case "${VARIANT}" in
    shading)
        SIM_POLICY="shading"
        EXTRA_FLAGS="--alphas 0.7"
        MLP_FLAGS=""
        ;;
    mpc_h10)
        SIM_POLICY="mpc"
        EXTRA_FLAGS="--mpc-horizons 10 --mpc-alpha-candidates 0.5 0.6 0.7 0.8 0.9 1.0"
        ;;
    mpc_h25)
        SIM_POLICY="mpc"
        EXTRA_FLAGS="--mpc-horizons 25 --mpc-alpha-candidates 0.5 0.6 0.7 0.8 0.9 1.0"
        ;;
    mpcbin_h10)
        SIM_POLICY="mpc-timebin"
        EXTRA_FLAGS="--mpc-horizons 10 --mpc-n-bins 2 --mpc-alpha-candidates 0.5 0.6 0.7 0.8 0.9 1.0"
        ;;
    mpcbin_h25)
        SIM_POLICY="mpc-timebin"
        EXTRA_FLAGS="--mpc-horizons 25 --mpc-n-bins 2 --mpc-alpha-candidates 0.5 0.6 0.7 0.8 0.9 1.0"
        ;;
    *)
        echo "ERROR: unknown VARIANT=${VARIANT}"
        exit 1
        ;;
esac

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
    echo "ERROR: could not find output CSV"
    exit 1
fi

python3 -c "
import pandas as pd
df = pd.read_csv('${CSV_PATH}')
df.insert(2, 'n_rollouts', ${NR})
df.to_csv('${CSV_PATH}', index=False)
print(f'  avg_bid_ms = {df[\"bid_time_s\"].mean()*1000:.1f}')
print(f'  mean_reward = {df.sort_values([\"seed\",\"step\"]).groupby(\"seed\")[\"cum_reward\"].last().mean():.3f}')
"

DEST="${OUT_DIR}/${VARIANT}_nr${NR}.csv"
mv "${CSV_PATH}" "${DEST}"
echo "Saved -> ${DEST}"
