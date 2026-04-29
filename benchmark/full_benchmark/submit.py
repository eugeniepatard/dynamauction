"""
Submit full benchmark jobs to the Inria Margaret (SLURM) cluster via submitit.

One job is created per horizon_eta value.
RL model paths are configured in benchmark/conf/eval.yaml (rl_models section).
Results land in /scratch/epatard/full_benchmark/results/.

Usage (from repo root, on the cluster)
---------------------------------------
    python benchmark/full_benchmark/submit.py --dry-run
    python benchmark/full_benchmark/submit.py
    python benchmark/full_benchmark/submit.py --horizon-etas 50 100
    python benchmark/full_benchmark/submit.py --config /path/to/eval.yaml
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import submitit

from benchmark.full_benchmark.run import run_job, _SCRATCH_RESULTS

HORIZON_ETAS = [50, 100, 500]

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--horizon-etas", type=int, nargs="+", default=HORIZON_ETAS,
        help=f"horizon_eta values to submit (default: {HORIZON_ETAS}).",
    )
    parser.add_argument("--n-seeds",    type=int,  default=30)
    parser.add_argument("--n-workers",  type=int,  default=None,
                        help="Worker processes per job (default: SLURM_CPUS_PER_TASK). Use 1 for sequential.")
    parser.add_argument("--config",     type=Path, default=None,
                        help="Path to eval YAML config (default: benchmark/conf/eval.yaml).")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--log-folder", default=_SUBMITIT_LOG_FOLDER)
    args = parser.parse_args()

    n_etas = len(args.horizon_etas)

    print(f"Jobs to submit : horizon_eta ∈ {args.horizon_etas}, {args.n_seeds} seeds each")
    print(f"Results dir    : {_SCRATCH_RESULTS}")

    if args.dry_run:
        for eta in args.horizon_etas:
            print(
                f"  [DRY RUN] run_job(horizon_eta={eta}, n_seeds={args.n_seeds}, "
                f"n_workers={args.n_workers}, cfg_path={args.config})"
            )
        print(f"\nTotal: {n_etas} jobs (dry run, nothing submitted)")
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**{**SLURM_PARAMS, "slurm_job_name": "dynac_bench"})

    jobs = executor.map_array(
        run_job,
        args.horizon_etas,
        [args.n_seeds]   * n_etas,   # n_seeds
        [None]           * n_etas,   # out_dir
        [None]           * n_etas,   # mlp_model
        [args.n_workers] * n_etas,   # n_workers
        [args.config]    * n_etas,   # cfg_path
    )
    ids = [j.job_id for j in jobs]
    print(f"\nSubmitted {len(jobs)} jobs — IDs: {ids[0]} … {ids[-1]}")
    print(f"Check status  : squeue -u epatard")
    print(f"Logs          : {args.log_folder}")


if __name__ == "__main__":
    main()
