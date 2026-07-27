#!/bin/bash
#SBATCH --partition=normal-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/epatard/submitit_logs/%x_%A_%a.out
#SBATCH --error=/scratch/epatard/submitit_logs/%x_%A_%a.err

# Sim-to-real matched-vs-mismatched comparison — 15 mismatch configurations.
#
# Each trial runs SIX policies:
#   MPC(H=10)[mismatch]               — sim uses fixed nominal (mu=0, sigma=1.0, eta=50)
#   MPCTimeBinned(H=10,K=2)[mismatch] — same
#   MPC(H=10)[matched]                — sim uses the SAME scenario as the real env
#   MPCTimeBinned(H=10,K=2)[matched]  — same
#   FiPla                             — myopic-optimal, no sim model
#   Shading(alpha=0.7)                — heuristic shading, no planning horizon
#
# The matched vs mismatched comparison isolates the cost of wrong rollout dynamics.
# V0 is fixed (trained on canonical sigma=1.0) in both cases.
#
# Results saved to /scratch/epatard/dynamicauc/sim2real_matched/
#
# Submit:
#   sbatch benchmark/run_scripts/run_sim2real_matched.sh
#
# Download results:
#   rsync -e ssh -av margaret02:/scratch/epatard/dynamicauc/sim2real_matched/ \
#         benchmark/benchmark_results/sim2real_matched/

set -e

source /scratch/epatard/miniconda3/etc/profile.d/conda.sh
conda activate dynamicauc
cd ~/projects/dynamicauc

echo "=== sim2real matched trial ${SLURM_ARRAY_TASK_ID}/14 ==="
echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}"

export SCRATCH_DIR=/scratch/epatard
export SIM2REAL_MATCHED=1

python benchmark/sim2real_sensitivity.py
