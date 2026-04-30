"""
Submit a shading α grid-search job to the Inria Margaret (SLURM) cluster via submitit.

Wraps benchmark/tune_shading_alpha.py — all its parameters are forwarded.
One job is submitted per horizon_eta value.
Results land in /scratch/epatard/full_benchmark/results/shading_alpha_results/.

Usage (from repo root, on the cluster)
---------------------------------------
    python benchmark/full_benchmark/submit_shading_alpha.py --dry-run
    python benchmark/full_benchmark/submit_shading_alpha.py --horizon-etas 50 100
    python benchmark/full_benchmark/submit_shading_alpha.py \\
        --horizon-etas 50 100 500 --n-alphas 20 --alpha-min 0.0 --alpha-max 1.5 --n-seeds 30
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import submitit

from benchmark.tune_shading_alpha import tune_shading_alpha

HORIZON_ETAS = [50, 100, 500]

_SCRATCH_DIR         = Path("/scratch/epatard")
_SCRATCH_RESULTS     = _SCRATCH_DIR / "full_benchmark" / "results" / "shading_alpha_results"
_LOCAL_RESULTS       = Path(__file__).resolve().parent / "results" / "shading_alpha_results"
_SUBMITIT_LOG_FOLDER = str(_SCRATCH_DIR / "submitit_logs" / "shading_alpha" / "%j")

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":           2880,
    "slurm_partition":       "normal-best,gpu-best",
    "gpus_per_node":         0,
    "cpus_per_task":         8,
    "tasks_per_node":        1,
    "nodes":                 1,
    "mem_gb":                8,
    "slurm_signal_delay_s":  60,
}


def _default_out_dir() -> Path:
    return _SCRATCH_RESULTS if _SCRATCH_DIR.exists() else _LOCAL_RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--horizon-etas", type=int, nargs="+", default=HORIZON_ETAS,
        help=f"horizon_eta values (default: {HORIZON_ETAS}). One job per value.",
    )
    parser.add_argument("--n-alphas",    type=int,   default=10,
                        help="Number of α values evenly spaced in [alpha-min, alpha-max].")
    parser.add_argument("--alpha-min",   type=float, default=0.1)
    parser.add_argument("--alpha-max",   type=float, default=1.0)
    parser.add_argument("--n-seeds",     type=int,   default=30)
    parser.add_argument("--n-rollouts",  type=int,   default=10,
                        help="MC rollouts per decision step inside the estimator.")
    parser.add_argument("--policy-seed", type=int,   default=0)
    parser.add_argument("--out-dir",     type=Path,  default=None,
                        help="Output directory for CSVs (default: scratch on cluster, local otherwise).")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--log-folder",  default=_SUBMITIT_LOG_FOLDER)
    args = parser.parse_args()

    import numpy as np
    alphas  = list(np.linspace(args.alpha_min, args.alpha_max, args.n_alphas))
    out_dir = args.out_dir or _default_out_dir()

    print(f"Job breakdown (dry-run={args.dry_run}):")
    print(f"  horizon_etas : {args.horizon_etas}")
    print(f"  alphas       : [{alphas[0]:.4g}, …, {alphas[-1]:.4g}]  ({len(alphas)} values)")
    print(f"  n_seeds      : {args.n_seeds}")
    print(f"  n_rollouts   : {args.n_rollouts}")
    print(f"  out_dir      : {out_dir}")
    print(f"  Total jobs   : {len(args.horizon_etas)}")

    if args.dry_run:
        for eta in args.horizon_etas:
            print(f"  [DRY RUN] tune_shading_alpha(horizon_eta={eta}, n_alphas={len(alphas)}, "
                  f"alpha=[{alphas[0]:.4g}..{alphas[-1]:.4g}], n_seeds={args.n_seeds})")
        return

    # Build argument lists for map_array:
    # tune_shading_alpha(horizon_eta, alphas, n_seeds, n_rollouts, policy_seed, out_dir)
    job_args = [
        (eta, alphas, args.n_seeds, args.n_rollouts, args.policy_seed, out_dir)
        for eta in args.horizon_etas
    ]

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS, slurm_job_name="dynac_shading_alpha")

    jobs = executor.map_array(tune_shading_alpha, *zip(*job_args))
    ids  = [j.job_id for j in jobs]
    print(f"\nSubmitted {len(jobs)} jobs — IDs: {ids[0]} … {ids[-1]}")
    print(f"Check status  : squeue -u epatard")
    print(f"Logs          : {args.log_folder}")


if __name__ == "__main__":
    main()
