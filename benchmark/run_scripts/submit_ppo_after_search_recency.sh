#!/bin/bash
# Run this on Margaret AFTER jobs 414526 (ppo_hp_rec) complete.
# Picks the best trial and submits a 1000-seed eval directly on its saved model.
# No retrain needed — the best trial model (1M steps) is used as-is.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_after_search_recency.sh

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== Selecting best hyperparameters (recency) ==="
eval "$(python benchmark/ppo_select_best.py --scenario recency | grep -v '^#')"
echo "Best trial ${BEST_TRIAL}: reward=${BEST_REWARD}"
echo "  ent_coef=${BEST_ENT_COEF}  lr=${BEST_LR}  n_steps=${BEST_N_STEPS}  gae=${BEST_GAE_LAMBDA}"

RUN_NAME="hparam_search_recency/trial_${BEST_TRIAL}"

SCENARIO_ID="recency-second-price-lognormal-eta50" \
ALGO=ppo \
RUN_NAME="${RUN_NAME}" \
    sbatch --job-name=eval_ppo_best_rec benchmark/run_scripts/run_eval_rl_robustness.sh

echo "Submitted 1000-seed eval for recency-eta50 (model: ${RUN_NAME})"
