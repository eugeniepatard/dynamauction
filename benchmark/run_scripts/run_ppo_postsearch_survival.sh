#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%j.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%j.err
#
# Post-search orchestrator for survival-second-price-exponential (delayed, clip).
# 1. Picks the best trial via ppo_select_best.py --scenario survival
# 2. Submits a 10M-step retrain with those hyperparameters
# 3. Submits a 1000-seed eval with dependency on the retrain
#
# Submitted automatically by submit_ppo_search_retrain_eval_survival.sh.

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

SCENARIO="survival-second-price-exponential"
RUN_NAME="${SCENARIO}/best_hparam_10M"

echo "=== Selecting best hyperparameters (survival delayed) ==="
eval "$(python benchmark/ppo_select_best.py --scenario survival | grep -v '^#')"
echo "Best trial ${BEST_TRIAL}: reward=${BEST_REWARD}"
echo "  lr=${BEST_LR}  ent_coef=${BEST_ENT_COEF}  n_steps=${BEST_N_STEPS}  gae_lambda=${BEST_GAE_LAMBDA}"

echo ""
echo "=== Submitting 10M-step retrain ==="
RETRAIN_JID=$(
  ALGO=ppo \
  SCENARIO_ID="${SCENARIO}" \
  EXTRA_ARGS="algo.learning_rate=${BEST_LR} algo.ent_coef=${BEST_ENT_COEF} algo.n_steps=${BEST_N_STEPS} algo.gae_lambda=${BEST_GAE_LAMBDA} training.run_name=${RUN_NAME}" \
      sbatch --job-name=ppo_retrain_surv_best \
             --time=16:00:00 \
             --parsable \
             benchmark/run_scripts/run_train_rl_robustness.sh
)
echo "Retrain job ID: ${RETRAIN_JID}"

echo ""
echo "=== Submitting 1000-seed eval (depends on retrain ${RETRAIN_JID}) ==="
EVAL_JID=$(
  SCENARIO_ID="${SCENARIO}" \
  ALGO=ppo \
  RUN_NAME="${RUN_NAME}" \
      sbatch --job-name=eval_ppo_best_surv \
             --dependency=afterok:${RETRAIN_JID} \
             --time=4:00:00 \
             --parsable \
             benchmark/run_scripts/run_eval_rl_robustness.sh
)
echo "Eval job ID: ${EVAL_JID}"

echo ""
echo "Pipeline submitted:"
echo "  retrain  ${RETRAIN_JID}  (ppo_retrain_surv_best, 10M steps)"
echo "  eval     ${EVAL_JID}    (eval_ppo_best_surv, 1000 seeds, afterok:${RETRAIN_JID})"
echo "  model    /scratch/epatard/dynamicauc/models/ppo/${RUN_NAME}/best_model.zip"
