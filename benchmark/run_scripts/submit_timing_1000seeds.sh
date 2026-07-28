#!/bin/bash
# Relaunch all timing benchmark jobs with 1000 seeds on the survival scenario.
# Time limits account for parallelism across 4 CPUs.
#
# Usage (on Margaret, from ~/projects/dynamicauc):
#   bash benchmark/run_scripts/submit_timing_1000seeds.sh

set -e

echo "=== RL inference (PPO / SAC / TD3) ==="
ALGO=ppo sbatch --job-name=timing_ppo --time=0:30:00 benchmark/run_scripts/run_timing_rl.sh
ALGO=sac sbatch --job-name=timing_sac --time=0:30:00 benchmark/run_scripts/run_timing_rl.sh
ALGO=td3 sbatch --job-name=timing_td3 --time=0:30:00 benchmark/run_scripts/run_timing_rl.sh

echo ""
echo "=== Myopic baselines (FiPla + GreedyMyopic) ==="
sbatch --job-name=timing_baselines benchmark/run_scripts/run_timing_baselines.sh

echo ""
echo "=== Shading (α=0.7) ==="
sbatch --job-name=timing_shading benchmark/run_scripts/run_timing_shading.sh

echo ""
echo "=== MPC variants ==="
# Time limits: 1000 seeds / 4 CPUs
#   mpc_h10:    94ms/bid  x 50 bids x 250 seeds = ~20 min  → 1h
#   mpc_h25:   173ms/bid  x 50 bids x 250 seeds = ~36 min  → 2h
#   mpcbin_h10: 470ms/bid x 50 bids x 250 seeds = ~98 min  → 4h
#   mpcbin_h25: 835ms/bid x 50 bids x 250 seeds = ~174 min → 6h
VARIANT=mpc_h10    sbatch --job-name=timing_mpc_h10    --time=1:00:00 benchmark/run_scripts/run_timing_mpc.sh
VARIANT=mpc_h25    sbatch --job-name=timing_mpc_h25    --time=2:00:00 benchmark/run_scripts/run_timing_mpc.sh
VARIANT=mpcbin_h10 sbatch --job-name=timing_mpcbin_h10 --time=4:00:00 benchmark/run_scripts/run_timing_mpc.sh
VARIANT=mpcbin_h25 sbatch --job-name=timing_mpcbin_h25 --time=6:00:00 benchmark/run_scripts/run_timing_mpc.sh

echo ""
echo "All timing jobs submitted (1000 seeds each)."
echo "Once done, download with:"
echo "  rsync -e ssh -av margaret02:~/projects/dynamicauc/benchmark/benchmark_results/timing/ benchmark/benchmark_results/timing/"
