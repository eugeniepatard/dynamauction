"""
Submit full benchmark jobs to the Inria Margaret (SLURM) cluster via submitit.

One job is created per horizon_eta value (50 and 100).
Results land in /scratch/epatard/full_benchmark/results/.

Usage (from repo root, on the cluster)
---------------------------------------
    # dry run — print jobs without submitting
    python benchmark/full_benchmark/submit.py --dry-run

    # submit both horizon_eta jobs
    python benchmark/full_benchmark/submit.py

    # submit only one value
    python benchmark/full_benchmark/submit.py --horizon-etas 50

Before submitting, copy the MLP model to the cluster:
    scp -J ssh.inria.fr benchmark/models/mlp_valuation_head/survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt epatard@margaret02:/home/fairplay/epatard/projects/dynamicauc/benchmark/models/mlp_valuation_head/
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
    "timeout_min":           2880,                       # 48 h
    "slurm_partition":       "normal-best,gpu-best",
    "gpus_per_node":         0,
    "cpus_per_task":         8,
    "tasks_per_node":        1,
    "nodes":                 1,
    "mem_gb":                16,
    "slurm_signal_delay_s":  120,
    "slurm_job_name":        "dynac_full_bench",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--horizon-etas", type=int, nargs="+", default=HORIZON_ETAS,
        choices=HORIZON_ETAS,
        help="Which horizon_eta values to submit (default: 50 100).",
    )
    parser.add_argument("--n-seeds",    type=int,  default=30)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print jobs without submitting anything.")
    parser.add_argument("--log-folder", default=_SUBMITIT_LOG_FOLDER,
                        help="submitit log folder template (%%j = SLURM job id).")
    args = parser.parse_args()

    print(f"Jobs to submit : horizon_eta ∈ {args.horizon_etas}, {args.n_seeds} seeds each")
    print(f"Results dir    : {_SCRATCH_RESULTS}")
    print(f"submitit logs  : {args.log_folder}")

    if args.dry_run:
        for eta in args.horizon_etas:
            print(f"  [DRY RUN] run_job(horizon_eta={eta}, n_seeds={args.n_seeds})")
        print(f"\nTotal: {len(args.horizon_etas)} jobs (dry run, nothing submitted)")
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS)

    jobs = executor.map_array(
        run_job,
        args.horizon_etas,
        [args.n_seeds] * len(args.horizon_etas),
    )
    ids = [j.job_id for j in jobs]
    print(f"\nSubmitted {len(jobs)} jobs — IDs: {ids[0]} … {ids[-1]}")
    print(f"Check status  : squeue -u epatard")
    print(f"Logs          : {args.log_folder}")


if __name__ == "__main__":
    main()
