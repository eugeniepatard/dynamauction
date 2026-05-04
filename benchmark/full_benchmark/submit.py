"""
Submit full benchmark jobs to the Inria Margaret (SLURM) cluster via submitit.

Jobs are split by policy group so each runs independently:
  greedy_fipla  — 1 job per horizon_eta
  shading       — 1 job per horizon_eta
  mpc           — 1 job per horizon_eta × mpc_horizon
  mpc_oracle    — 1 job per horizon_eta × mpc_horizon
  mpc_timebin   — 1 job per horizon_eta × mpc_horizon × n_bins
  rl            — 1 job per horizon_eta × algo

Default: submit all job types for all horizon_etas.

Usage (from repo root, on the cluster)
---------------------------------------
    python benchmark/full_benchmark/submit.py --dry-run
    python benchmark/full_benchmark/submit.py
    python benchmark/full_benchmark/submit.py --job-types greedy_fipla shading mpc mpc_timebin
    python benchmark/full_benchmark/submit.py --job-types rl
    python benchmark/full_benchmark/submit.py --job-types rl --algos ppo td3
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
    _DEFAULT_PPO_MODEL,
    _DEFAULT_SAC_MODEL,
    _DEFAULT_TD3_MODEL,
)

HORIZON_ETAS = [50, 100]
RL_ALGOS     = ["ppo", "sac", "td3"]

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

# run_job positional order:
#   horizon_eta, job_type, mpc_horizon, n_bins, mlp_model,
#   ppo_model, sac_model, td3_model, algos, n_seeds, out_dir, n_workers
_ALGO_MODEL_IDX = {"ppo": 5, "sac": 6, "td3": 7}  # position of each model arg in the tuple


def _planning_tuple(eta, jt, H, K, n_seeds, n_workers):
    return (eta, jt, H, K, None, None, None, None, None, n_seeds, None, n_workers)


def _rl_tuple(eta, algo, model_path, n_seeds, n_workers):
    ppo = model_path if algo == "ppo" else None
    sac = model_path if algo == "sac" else None
    td3 = model_path if algo == "td3" else None
    return (eta, "rl", 0.0, 2, None, ppo, sac, td3, [algo], n_seeds, None, n_workers)


def _build_jobs(
    horizon_etas: list[int],
    planning_types: list[str],
    rl_algos: list[str],
    algo_models: dict[str, Path],
    n_seeds: int,
    n_workers: int | None,
) -> list[tuple]:
    jobs = []

    for eta in horizon_etas:
        for jt in planning_types:
            if jt in ("greedy_fipla", "shading"):
                jobs.append(_planning_tuple(eta, jt, 0.0, 2, n_seeds, n_workers))
            elif jt in ("mpc", "mpc_oracle"):
                for H in MPC_HORIZONS:
                    jobs.append(_planning_tuple(eta, jt, H, 2, n_seeds, n_workers))
            elif jt == "mpc_timebin":
                for H in MPC_HORIZONS:
                    for K in MPC_N_BINS:
                        jobs.append(_planning_tuple(eta, jt, H, K, n_seeds, n_workers))

        for algo in rl_algos:
            jobs.append(_rl_tuple(eta, algo, algo_models[algo], n_seeds, n_workers))

    return jobs


def main() -> None:
    planning_only_types = [jt for jt in JOB_TYPES if jt != "rl"]
    all_types           = planning_only_types + ["rl"]

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
        help=f"horizon_eta values (default: {HORIZON_ETAS}).",
    )
    parser.add_argument(
        "--algos", nargs="+", choices=RL_ALGOS, default=RL_ALGOS,
        help=f"RL algorithms to submit (default: all). Only used when rl is in --job-types.",
    )
    parser.add_argument("--n-seeds",   type=int,  default=30)
    parser.add_argument("--n-workers", type=int,  default=None,
                        help="Worker processes per job (default: SLURM_CPUS_PER_TASK).")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--log-folder", default=_SUBMITIT_LOG_FOLDER)
    parser.add_argument("--ppo-model", type=Path, default=None,
                        help=f"PPO model path (without .zip). Default: {_DEFAULT_PPO_MODEL}")
    parser.add_argument("--sac-model", type=Path, default=None,
                        help=f"SAC model path (without .zip). Default: {_DEFAULT_SAC_MODEL}")
    parser.add_argument("--td3-model", type=Path, default=None,
                        help=f"TD3 model path (without .zip). Default: {_DEFAULT_TD3_MODEL}")
    args = parser.parse_args()

    planning_types = [jt for jt in args.job_types if jt != "rl"]
    rl_algos       = args.algos if "rl" in args.job_types else []

    algo_models = {
        "ppo": args.ppo_model or _DEFAULT_PPO_MODEL,
        "sac": args.sac_model or _DEFAULT_SAC_MODEL,
        "td3": args.td3_model or _DEFAULT_TD3_MODEL,
    }

    all_jobs = _build_jobs(
        args.horizon_etas, planning_types, rl_algos, algo_models,
        args.n_seeds, args.n_workers,
    )

    n_planning = sum(1 for j in all_jobs if j[1] != "rl")
    n_rl       = sum(1 for j in all_jobs if j[1] == "rl")

    print(f"Job breakdown (dry-run={args.dry_run}):")
    print(f"  horizon_etas  : {args.horizon_etas}")
    print(f"  planning types: {planning_types}")
    print(f"  RL algos      : {rl_algos}")
    print(f"  planning jobs : {n_planning}")
    print(f"  RL jobs       : {n_rl}  (1 per eta × algo)")
    print(f"  Total         : {len(all_jobs)}")

    if args.dry_run:
        for j in all_jobs:
            eta, jt, H, K = j[0], j[1], j[2], j[3]
            algos = j[8]
            if jt == "rl":
                print(f"  [DRY RUN] run_job(eta={eta}, type=rl, algo={algos[0]})")
            else:
                print(f"  [DRY RUN] run_job(eta={eta}, type={jt}, H={H:g}, K={K})")
        print(f"\nTotal: {len(all_jobs)} jobs (dry run, nothing submitted)")
        return

    if not all_jobs:
        print("Nothing to submit.")
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS, slurm_job_name="dynac_bench")
    jobs    = executor.map_array(run_job, *zip(*all_jobs))
    job_ids = [j.job_id for j in jobs]

    print(f"\nSubmitted {len(job_ids)} jobs — IDs: {job_ids[0]} … {job_ids[-1]}")
    print(f"Check status  : squeue -u epatard")
    print(f"Logs          : {args.log_folder}")


if __name__ == "__main__":
    main()
