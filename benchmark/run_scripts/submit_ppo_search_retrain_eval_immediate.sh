#!/bin/bash
# Full pipeline for PPO on survival-second-price-exponential-immediate:
#   1. 135-trial hyperparameter search (1M steps each, array job)
#   2. Post-search job picks best trial, submits 10M-step retrain
#   3. Retrain finishes → dependency 1000-seed eval auto-submitted
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_ppo_search_retrain_eval_immediate.sh

set -e

echo "=== Step 1: hyperparameter search (135 trials, 1M steps each) ==="
SEARCH_JID=$(sbatch --job-name=ppo_hp_imm \
                    --parsable \
                    benchmark/run_scripts/run_ppo_hparam_search_immediate.sh)
echo "Search array job ID: ${SEARCH_JID}"

echo ""
echo "=== Step 2: post-search orchestrator (depends on array ${SEARCH_JID}) ==="
POSTSEARCH_JID=$(sbatch --job-name=ppo_postsearch_imm \
                        --dependency=afterany:${SEARCH_JID} \
                        --parsable \
                        benchmark/run_scripts/run_ppo_postsearch_immediate.sh)
echo "Post-search job ID: ${POSTSEARCH_JID}"

echo ""
echo "Pipeline submitted. Full chain:"
echo "  [${SEARCH_JID}]  ppo_hp_imm         — 135-trial hparam search"
echo "  [${POSTSEARCH_JID}]  ppo_postsearch_imm — picks best, submits retrain + eval"
echo "  [auto]              ppo_retrain_imm_best — 10M-step retrain with best hparams"
echo "  [auto]              eval_ppo_best_imm    — 1000-seed eval on immediate kernel"
echo ""
echo "Download results once eval completes:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/ benchmark/benchmark_results/"
