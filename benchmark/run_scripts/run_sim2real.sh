#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%A_%a.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%A_%a.err

# Sim-to-real sensitivity for MPC — 15 mismatch configurations:
#
# MPC's sim is FIXED at LogNormal(mu=0, sigma=1.0) / Exp(eta=50).
# The REAL evaluation environment VARIES:
#
#   trials 0-7  : competition mismatch  (sim LN(mu=0), real LN(mu=X))
#                 mu ∈ {-2.0, -1.5, -1.0, -0.5, 0.0*, 0.5, 1.0, 1.5}
#   trials 8-14 : horizon mismatch      (sim Exp(50), real Exp(eta))
#                 eta  ∈ {10, 20, 35, 50*, 75, 100, 200}
#
#   * = no mismatch (sim and real are the same)
#
# MPC uses a genuinely separate sim env — no deepcopy of the real env.
# Only current state (t, ctx, step_idx, pending_conversions) is transplanted
# into the sim env before each bid; rollout dynamics (competition, horizon,
# arrival) come entirely from the sim scenario.
#
# Each job: 100 seeds × MPCTimeBinned(H=10,K=2)[mismatch] + FiPla + Shading.
#
# Submit:
#   sbatch benchmark/run_scripts/run_sim2real.sh
#
# After completion, download and plot:
#   rsync -e ssh -av margaret02:/scratch/epatard/dynamicauc/sim2real/ \
#         benchmark/benchmark_results/sim2real/

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== sim2real trial ${SLURM_ARRAY_TASK_ID}/14 ==="
echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}"

export SCRATCH_DIR=/scratch/epatard

python benchmark/sim2real_sensitivity.py
