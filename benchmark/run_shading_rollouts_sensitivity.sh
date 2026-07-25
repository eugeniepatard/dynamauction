#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err

# Sweep n_rollouts for the Shading policy and measure how reward changes.
# Usage:
#   SCENARIO_ID=saturation-second-price-uniform BEST_ALPHA=0.7 \
#     sbatch --job-name=shading_nr_saturation benchmark/run_shading_rollouts_sensitivity.sh

set -e

if [ -z "${SCENARIO_ID}" ]; then echo "ERROR: SCENARIO_ID must be set"; exit 1; fi
if [ -z "${BEST_ALPHA}" ]; then echo "ERROR: BEST_ALPHA must be set"; exit 1; fi

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SENS_DIR="benchmark/benchmark_results/shading_nrollouts_sensitivity"
mkdir -p "${SENS_DIR}"

N_SEEDS=300
ROLLOUTS_LIST="10 25 50 100 200"

for NR in ${ROLLOUTS_LIST}; do
    echo "=== ${SCENARIO_ID}  alpha=${BEST_ALPHA}  n_rollouts=${NR}  seeds=${N_SEEDS} ==="

    python -m benchmark.evaluate \
        --scenario-id "${SCENARIO_ID}" \
        --algos none \
        --baselines none \
        --sim-policies shading \
        --alphas "${BEST_ALPHA}" \
        --n-rollouts "${NR}" \
        --n-seeds "${N_SEEDS}"

    LATEST=$(ls -t benchmark/benchmark_results/${SCENARIO_ID}_*.csv 2>/dev/null | head -1)
    if [ -z "${LATEST}" ]; then
        echo "ERROR: no CSV found after evaluation"; exit 1
    fi

    # Tag with n_rollouts so runs are distinguishable when merged
    python3 -c "
import pandas as pd
df = pd.read_csv('${LATEST}')
df.insert(1, 'n_rollouts', ${NR})
df.to_csv('${LATEST}', index=False)
"
    DEST="${SENS_DIR}/${SCENARIO_ID}_nr${NR}.csv"
    mv "${LATEST}" "${DEST}"
    echo "Saved -> ${DEST}"
done

# Merge all per-rollout CSVs into one sensitivity file for this scenario
MERGED="${SENS_DIR}/${SCENARIO_ID}_nrollouts_sensitivity.csv"
python3 -c "
import pandas as pd, glob
files = sorted(glob.glob('${SENS_DIR}/${SCENARIO_ID}_nr*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('${MERGED}', index=False)
print(f'Merged {len(files)} files -> ${MERGED}')
print(df.groupby('n_rollouts')['cum_reward'].describe() if 'cum_reward' in df.columns else df.head())
"
echo "Done: ${SCENARIO_ID}"
