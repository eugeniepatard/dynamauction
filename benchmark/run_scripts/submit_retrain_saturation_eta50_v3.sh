#!/bin/bash
# Retrain SAC and TD3 on saturation-second-price-uniform-eta50 (v3 fix).
#
# Root cause of the v2 collapse (rl_ppo_sat50_v2 / rl_td3_sat50_v2, see
# submit_retrain_saturation_eta50.sh): both policies STILL converged to
# always-bid-0, even with higher entropy/noise. Two compounding problems:
#
#   1. VecNormalize(norm_reward=True) divides reward by a running std that
#      shrinks toward 0 as the policy collapses, amplifying noise instead
#      of providing a useful gradient signal once rewards get small.
#   2. max_bid=5.0 wastes most of the action range: competing bids are
#      Uniform(0,1) under second-price rules, so any bid >= 1 always wins
#      with the same payment distribution. ~80% of the tanh-squashed
#      policy's output range was a functionally flat "wasteland".
#
# Validated on a short 300k-step run (see conversation/notebook) with
#   training.norm_reward=false training.max_bid=1.5
# -> SAC and TD3 both escape the zero-bid trap (eval reward climbs to a
#    steady positive 0.4-2.6 range). PPO does NOT escape it (confirmed at
#    ent_coef up to 0.2): PPO's entropy bonus is computed on the pre-tanh
#    Gaussian, which stays "high" even as the MEAN saturates the tanh
#    boundary (bid ~ 0) — a structural issue, not a tuning issue. SAC's
#    squash-corrected entropy (computed post-tanh) is why it thrives here;
#    TD3's action noise is added post-squash (normalized [-1,1] space) so
#    it isn't blunted by mean saturation either. PPO is intentionally
#    skipped in this retrain.
#
# Run this on Margaret: bash benchmark/submit_retrain_saturation_eta50_v3.sh

SCENARIO_ID=saturation-second-price-uniform-eta50
RUN_NAME="${SCENARIO_ID}-v3"
FIX_ARGS="training.norm_reward=false training.max_bid=1.5"

echo "Submitting TD3 retrain (action_noise_sigma=0.5, ${FIX_ARGS})..."
ALGO=td3 \
SCENARIO_ID="${SCENARIO_ID}" \
EXTRA_ARGS="algo.action_noise_sigma=0.5 training.run_name=${RUN_NAME} ${FIX_ARGS}" \
    sbatch --job-name=rl_td3_sat50_v3 benchmark/run_train_rl_robustness.sh

sleep 5

echo "Submitting SAC retrain (default config, ${FIX_ARGS})..."
ALGO=sac \
SCENARIO_ID="${SCENARIO_ID}" \
EXTRA_ARGS="training.run_name=${RUN_NAME} ${FIX_ARGS}" \
    sbatch --job-name=rl_sac_sat50_v3 benchmark/run_train_rl_robustness.sh

echo ""
echo "Submitted 2 jobs (PPO skipped — confirmed still degenerate at ent_coef=0.2)."
echo "After both complete (TD3 ~8-9h based on v2 timing), run:"
echo "  bash benchmark/submit_eval_saturation_eta50_v2.sh   # (update SCENARIO run_name to -v3 first)"
