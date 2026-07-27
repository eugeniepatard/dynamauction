#!/bin/bash
# Oracle immediate experiment: train PPO/SAC/TD3 on the immediate-reward survival
# scenario with mask_expected_reward=False, giving RL full oracle access to the
# expected-reward signal that FiPla uses.
#
# Purpose: isolate whether RL underperformance on the immediate scenario is due to
# information masking (can't see expected_reward) or fundamental exploration failure.
#
# Run on Margaret (NOT on login node):
#   bash benchmark/run_scripts/submit_oracle_immediate.sh

set -e

SCENARIO_ID=survival-second-price-exponential-immediate
RUN_NAME="${SCENARIO_ID}-oracle"

echo "=== Oracle immediate experiment ==="
echo "Scenario : ${SCENARIO_ID}"
echo "Run name : ${RUN_NAME}"
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
# mask_expected_reward=False gives RL the same expected_reward obs that FiPla uses.

echo "Submitting training jobs..."

PPO_TRAIN=$(
  ALGO=ppo \
  SCENARIO_ID="${SCENARIO_ID}" \
  EXTRA_ARGS="training.mask_expected_reward=False training.run_name=${RUN_NAME}" \
  sbatch --parsable --job-name=rl_ppo_imm_oracle \
    benchmark/run_scripts/run_train_rl_robustness.sh
)
echo "  PPO train → job ${PPO_TRAIN}"

SAC_TRAIN=$(
  ALGO=sac \
  SCENARIO_ID="${SCENARIO_ID}" \
  EXTRA_ARGS="training.mask_expected_reward=False training.run_name=${RUN_NAME}" \
  sbatch --parsable --job-name=rl_sac_imm_oracle \
    benchmark/run_scripts/run_train_rl_robustness.sh
)
echo "  SAC train → job ${SAC_TRAIN}"

TD3_TRAIN=$(
  ALGO=td3 \
  SCENARIO_ID="${SCENARIO_ID}" \
  EXTRA_ARGS="training.mask_expected_reward=False training.run_name=${RUN_NAME}" \
  sbatch --parsable --job-name=rl_td3_imm_oracle \
    benchmark/run_scripts/run_train_rl_robustness.sh
)
echo "  TD3 train → job ${TD3_TRAIN}"

# ── Evaluation (depends on training) ─────────────────────────────────────────
# --no-mask-expected-reward must match training (model expects full obs space).

echo ""
echo "Submitting eval jobs (after training completes)..."

ALGO=ppo \
SCENARIO_ID="${SCENARIO_ID}" \
RUN_NAME="${RUN_NAME}" \
EXTRA_EVAL_ARGS="--no-mask-expected-reward" \
sbatch --dependency=afterok:${PPO_TRAIN} \
       --job-name=eval_ppo_imm_oracle \
       benchmark/run_scripts/run_eval_rl_robustness.sh
echo "  PPO eval  → after job ${PPO_TRAIN}"

ALGO=sac \
SCENARIO_ID="${SCENARIO_ID}" \
RUN_NAME="${RUN_NAME}" \
EXTRA_EVAL_ARGS="--no-mask-expected-reward" \
sbatch --dependency=afterok:${SAC_TRAIN} \
       --job-name=eval_sac_imm_oracle \
       benchmark/run_scripts/run_eval_rl_robustness.sh
echo "  SAC eval  → after job ${SAC_TRAIN}"

ALGO=td3 \
SCENARIO_ID="${SCENARIO_ID}" \
RUN_NAME="${RUN_NAME}" \
EXTRA_EVAL_ARGS="--no-mask-expected-reward" \
sbatch --dependency=afterok:${TD3_TRAIN} \
       --job-name=eval_td3_imm_oracle \
       benchmark/run_scripts/run_eval_rl_robustness.sh
echo "  TD3 eval  → after job ${TD3_TRAIN}"

echo ""
echo "All 6 jobs submitted."
echo "Results will appear in benchmark_results/ when eval jobs complete."
echo "Download with:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ \\"
echo "        benchmark/benchmark_results/"
