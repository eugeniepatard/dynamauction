#!/bin/bash
# Evaluate survival-trained PPO/SAC/TD3 on the immediate kernel (1000 seeds).
#
# Models used: best hyperparameters from survival-second-price-exponential (delayed),
# to test whether they transfer to the immediate-reward variant after the Bernoulli fix.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_eval_survival_on_immediate.sh

SCENARIO="survival-second-price-exponential-immediate"

echo "Submitting PPO eval..."
SCENARIO_ID="${SCENARIO}" \
ALGO=ppo \
RUN_NAME="eta50/lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5" \
    sbatch --job-name=eval_ppo_surv_imm benchmark/run_scripts/run_eval_rl_robustness.sh

echo "Submitting SAC eval..."
SCENARIO_ID="${SCENARIO}" \
ALGO=sac \
RUN_NAME="eta50/lr=5e-05_bs=512_warmup=500_gamma=0.95" \
    sbatch --job-name=eval_sac_surv_imm benchmark/run_scripts/run_eval_rl_robustness.sh

echo "Submitting TD3 eval..."
SCENARIO_ID="${SCENARIO}" \
ALGO=td3 \
RUN_NAME="eta50/lr=0.0001_bs=512_noise=0.1_warmup=500_gamma=0.99" \
    sbatch --job-name=eval_td3_surv_imm benchmark/run_scripts/run_eval_rl_robustness.sh

echo ""
echo "Submitted 3 eval jobs. Download once complete:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ benchmark/benchmark_results/"
