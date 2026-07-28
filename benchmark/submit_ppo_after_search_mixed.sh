#!/bin/bash
# Run this on Margaret AFTER jobs 414527 (ppo_hp_mix) complete.
# Picks the best hyperparameters and submits a full 10M-step retrain.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_after_search_mixed.sh
#
# After the retrain job finishes (~3h), run:
#   bash benchmark/run_scripts/submit_ppo_eval_1000seeds_mixed.sh

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== Selecting best hyperparameters (mixed) ==="
eval "$(python benchmark/ppo_select_best.py --scenario mixed | grep -v '^#')"
echo "Best trial ${BEST_TRIAL}: reward=${BEST_REWARD}"
echo "  ent_coef=${BEST_ENT_COEF}  lr=${BEST_LR}  n_steps=${BEST_N_STEPS}  gae=${BEST_GAE_LAMBDA}"

RUN_NAME="ppo_best_mixed"

ALGO=ppo \
SCENARIO_ID="mixed-second-price-exponential-eta50" \
EXTRA_ARGS="algo.ent_coef=${BEST_ENT_COEF} algo.learning_rate=${BEST_LR} algo.n_steps=${BEST_N_STEPS} algo.gae_lambda=${BEST_GAE_LAMBDA} training.mask_expected_reward=true training.run_name=${RUN_NAME}" \
    sbatch --job-name=ppo_best_mix benchmark/run_scripts/run_train_rl_robustness.sh

echo ""
echo "Submitted retrain job.  After it completes, run:"
echo "  bash benchmark/run_scripts/submit_ppo_eval_1000seeds_mixed.sh"
