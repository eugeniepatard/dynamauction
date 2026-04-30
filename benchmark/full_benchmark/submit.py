"""
Submit full benchmark jobs to the Inria Margaret (SLURM) cluster via submitit.

Jobs are split by policy group so each runs independently:
  greedy_fipla  — 1 job per horizon_eta
  shading       — 1 job per horizon_eta
  mpc           — 1 job per horizon_eta × mpc_horizon          (6 horizons → 6 jobs/eta)
  mpc_timebin   — 1 job per horizon_eta × mpc_horizon × n_bins (6 × 4 = 24 jobs/eta)
  rl            — 1 job per horizon_eta (via run_rl.py, requires model paths)

Default: submit all job types for all horizon_etas.

Usage (from repo root, on the cluster)
---------------------------------------
    python benchmark/full_benchmark/submit.py --dry-run
    python benchmark/full_benchmark/submit.py
    python benchmark/full_benchmark/submit.py --job-types greedy_fipla shading mpc mpc_timebin
    python benchmark/full_benchmark/submit.py --job-types rl \\
        --ppo-model /scratch/epatard/models/ppo/best_model
    python benchmark/full_benchmark/submit.py --horizon-etas 50 100
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import submitit

from benchmark.full_benchmark.run import (
    run_job,
    MPC_HORIZONS,
    MPC_N_BINS,
    JOB_TYPES,
)

HORIZON_ETAS = [50, 100]

_SCRATCH_DIR         = Path("/scratch/epatard")
_SUBMITIT_LOG_FOLDER = str(_SCRATCH_DIR / "submitit_logs" / "full_benchmark" / "%j")

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":           2880,
    "slurm_partition":       "normal-best,gpu-best",
    "gpus_per_node":         0,
    "cpus_per_task":         32,
    "tasks_per_node":        1,
    "nodes":                 1,
    "mem_gb":                16,
    "slurm_signal_delay_s":  120,
}


def _build_planning_jobs(
    horizon_etas: list[int],
    job_types: list[str],
    n_seeds: int,
    n_workers: int | None,
) -> list[tuple]:
    """Build the list of (positional) argument tuples for run_job."""
    jobs = []
    for eta in horizon_etas:
        for jt in job_types:
            if jt in ("greedy_fipla", "shading"):
                # mpc_horizon and n_bins are ignored for these types
                jobs.append((eta, jt, 0.0, 2, n_seeds, None, None, n_workers))
            elif jt == "mpc":
                for H in MPC_HORIZONS:
                    jobs.append((eta, jt, H, 2, n_seeds, None, None, n_workers))
            elif jt == "mpc_timebin":
                for H in MPC_HORIZONS:
                    for K in MPC_N_BINS:
                        jobs.append((eta, jt, H, K, n_seeds, None, None, n_workers))
    return jobs


def main() -> None:
    all_types = list(JOB_TYPES) + ["rl"]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--job-types", nargs="+", choices=all_types, default=all_types,
        help=f"Policy groups to submit (default: all). Choices: {all_types}",
    )
    parser.add_argument(
        "--horizon-etas", type=int, nargs="+", default=HORIZON_ETAS,
        help=f"horizon_eta values to submit (default: {HORIZON_ETAS}).",
    )
    parser.add_argument("--n-seeds",    type=int,  default=30)
    parser.add_argument("--n-workers",  type=int,  default=None,
                        help="Worker processes per job (default: SLURM_CPUS_PER_TASK).")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--log-folder", default=_SUBMITIT_LOG_FOLDER)
    parser.add_argument("--ppo-model",  type=Path, default=None,
                        help="PPO best_model path (without .zip) — forwarded to RL jobs.")
    parser.add_argument("--sac-model",  type=Path, default=None,
                        help="SAC best_model path (without .zip) — forwarded to RL jobs.")
    parser.add_argument("--td3-model",  type=Path, default=None,
                        help="TD3 best_model path (without .zip) — forwarded to RL jobs.")
    args = parser.parse_args()

    planning_types = [jt for jt in args.job_types if jt != "rl"]
    do_rl          = "rl" in args.job_types

    planning_jobs = _build_planning_jobs(
        args.horizon_etas, planning_types, args.n_seeds, args.n_workers
    )

    # RL jobs: (horizon_eta, n_seeds, out_dir, n_workers, ppo, sac, td3)
    rl_jobs: list[tuple] = []
    if do_rl:
        for eta in args.horizon_etas:
            rl_jobs.append((
                eta, args.n_seeds, None, args.n_workers,
                args.ppo_model, args.sac_model, args.td3_model,
            ))

    total = len(planning_jobs) + len(rl_jobs)

    print(f"Job breakdown (dry-run={args.dry_run}):")
    print(f"  horizon_etas  : {args.horizon_etas}")
    print(f"  planning types: {planning_types}")
    print(f"  planning jobs : {len(planning_jobs)}")
    print(f"  RL jobs       : {len(rl_jobs)}")
    print(f"  Total         : {total}")

    if args.dry_run:
        for job_args in planning_jobs:
            eta, jt, H, K, *_ = job_args
            print(f"  [DRY RUN] run_job(eta={eta}, type={jt}, H={H:g}, K={K})")
        for job_args in rl_jobs:
            eta, *_ = job_args
            print(f"  [DRY RUN] run_rl_job(eta={eta})")
        print(f"\nTotal: {total} jobs (dry run, nothing submitted)")
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    job_ids: list[str] = []

    if planning_jobs:
        executor.update_parameters(**SLURM_PARAMS, slurm_job_name="dynac_bench")
        p_jobs = executor.map_array(run_job, *zip(*planning_jobs))
        job_ids.extend(j.job_id for j in p_jobs)

    if rl_jobs:
        from benchmark.full_benchmark.run_rl import run_rl_job
        executor.update_parameters(**SLURM_PARAMS, slurm_job_name="dynac_rl_bench")
        r_jobs = executor.map_array(run_rl_job, *zip(*rl_jobs))
        job_ids.extend(j.job_id for j in r_jobs)

    print(f"\nSubmitted {len(job_ids)} jobs — IDs: {job_ids[0]} … {job_ids[-1]}")
    print(f"Check status  : squeue -u epatard")
    print(f"Logs          : {args.log_folder}")


if __name__ == "__main__":
    main()
