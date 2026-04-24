"""
Hyperparameter sweep for PPO / SAC / TD3 on the Inria Margaret (SLURM) cluster.

Each hyperparameter combination becomes one independent SLURM job submitted via
submitit.  Results land in:
    /scratch/epatard/models/<algo>/<run_name>/best_model.zip
    /scratch/epatard/logs/<algo>/<run_name>/

Usage (from the repo root, on the cluster):
    # dry run — print all jobs without submitting
    python benchmark/params_search/sweep.py --dry-run

    # submit PPO jobs only
    python benchmark/params_search/sweep.py --algo ppo

    # submit all algos
    python benchmark/params_search/sweep.py --algo ppo sac td3
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import submitit

from benchmark.train_rl import TrainingConfig, train

PPO_GRID: dict[str, list] = {
    "learning_rate": [5e-5,1e-4, 2e-4, 3e-4],
    "ppo_ent_coef":  [0.0, 0.01, 0.02],
    "ppo_n_steps":   [256, 512],
    "gamma":         [0.95, 0.99],
    "ppo_n_epochs":       [2, 5, 10],
}

SAC_GRID: dict[str, list] = {
    "learning_rate":              [5e-5, 1e-4, 3e-4, 1e-3],
    "off_policy_batch_size":      [256, 512],
    "off_policy_learning_starts": [500, 1_000, 1500],
    "gamma":                      [0.95, 0.99],
}

TD3_GRID: dict[str, list] = {
    "learning_rate":          [5e-5, 1e-4, 3e-4, 1e-3],
    "off_policy_batch_size":  [256, 512],
    "td3_action_noise_sigma": [0.1, 0.2],
    "off_policy_learning_starts": [500, 1_000, 1_500],
    "gamma":                  [0.95, 0.99],
}

GRIDS = {"ppo": PPO_GRID, "sac": SAC_GRID, "td3": TD3_GRID}

# ── Margaret SLURM parameters ────────────────────────────────────────────────

SCRATCH_DIR = Path("/scratch/epatard")

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":          2880,                    # 48 h
    "slurm_partition":      "normal-best,gpu-best",
    "gpus_per_node":        0,
    "cpus_per_task":        8,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 120,                     # send SIGUSR1 120 s before timeout
    "slurm_max_num_timeout":1,
    "slurm_array_parallelism": 3,                    # max 3 jobs running concurrently
    "slurm_job_name":       "dynac_sweep",
}

_SUBMITIT_LOG_FOLDER = str(SCRATCH_DIR / "submitit_logs" / "%j")

_KEY_ALIASES = {
    "learning_rate":          "lr",
    "ppo_ent_coef":           "ent",
    "ppo_n_steps":            "steps",
    "ppo_n_epochs":           "epochs",
    "ppo_gae_lambda":         "gae",
    "off_policy_batch_size":  "bs",
    "off_policy_learning_starts": "warmup",
    "td3_action_noise_sigma": "noise",
}

def _make_run_name(overrides: dict[str, Any]) -> str:
    """Compact identifier for a hyperparameter combination used as a subdirectory name."""
    parts = []
    for k, v in overrides.items():
        short = _KEY_ALIASES.get(k, k)
        parts.append(f"{short}={v}")
    return "_".join(parts)


def _expand_grid(algo: str, total_timesteps: int) -> list[tuple[TrainingConfig, str]]:
    """Return (config, run_name) for every point in the algo's grid."""
    base = TrainingConfig(
        total_timesteps=total_timesteps,
        scratch_dir=SCRATCH_DIR,
        verbose=False,  # suppress per-step output in cluster jobs
    )
    grid = GRIDS[algo]
    out = []
    for values in itertools.product(*grid.values()):
        overrides = dict(zip(grid.keys(), values))
        run_name  = _make_run_name(overrides)
        cfg       = replace(base, run_name=run_name, **overrides)
        out.append((cfg, run_name))
    return out


def _train_job(algo: str, cfg: TrainingConfig) -> None:
    """Callable submitted to SLURM — one job per hyperparameter combination."""
    train(algo, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo", nargs="+", choices=["ppo", "sac", "td3"],
        default=["ppo", "sac", "td3"],
        help="Algorithms to sweep (default: all three)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Total training timesteps per job (default: 5M)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all jobs without submitting anything",
    )
    parser.add_argument(
        "--log-folder", default=_SUBMITIT_LOG_FOLDER,
        help="submitit log folder template (%%j = SLURM job id)",
    )
    args = parser.parse_args()

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS)

    total = 0
    for algo in args.algo:
        combos = _expand_grid(algo, args.timesteps)
        n = len(combos)
        total += n

        if args.dry_run:
            print(f"\n{algo.upper()}: {n} jobs")
            for _, run_name in combos:
                print(f"  {run_name}")
            continue

        jobs = executor.map_array(
            _train_job,
            [algo] * n,
            [cfg for cfg, _ in combos],
        )
        ids = [j.job_id for j in jobs]
        print(f"[{algo.upper()}] {n} jobs submitted — IDs {ids[0]} … {ids[-1]}")

    if args.dry_run:
        print(f"\nTotal: {total} jobs (dry run, nothing submitted)")
    else:
        print(f"\nTotal: {total} jobs submitted.  Logs → {args.log_folder}")


if __name__ == "__main__":
    main()
