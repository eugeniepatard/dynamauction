#!/bin/bash
# Run this on Margaret AFTER jobs 414525 (ppo_hp_imm) complete.
# Picks the best trial and submits a 1000-seed eval directly on its saved model.
# No retrain needed — the best trial model (1M steps) is used as-is.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_after_search_immediate.sh

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== Selecting best hyperparameters (immediate) ==="
eval "$(python benchmark/ppo_select_best.py --scenario immediate | grep -v '^#')"
echo "Best trial ${BEST_TRIAL}: reward=${BEST_REWARD}"
echo "  ent_coef=${BEST_ENT_COEF}  lr=${BEST_LR}  n_steps=${BEST_N_STEPS}  gae=${BEST_GAE_LAMBDA}"

# The hparam search already saved best_model for each trial at:
# /scratch/epatard/dynamicauc/models/ppo/hparam_search_immediate/trial_XXX/best_model
RUN_NAME="hparam_search_immediate/trial_${BEST_TRIAL}"

SCENARIO_ID="survival-second-price-exponential-immediate" \
ALGO=ppo \
RUN_NAME="${RUN_NAME}" \
EXTRA_EVAL_ARGS="--horizon-eta 50" \
    sbatch --job-name=eval_ppo_best_imm benchmark/run_scripts/run_eval_rl_robustness.sh

echo "Submitted 1000-seed eval for survival-immediate (model: ${RUN_NAME})"
