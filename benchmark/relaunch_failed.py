"""
Relaunch the 15 jobs that failed or were killed in previous SLURM runs.

Three groups:
  1. RL sweep   — 7 TD3 configs from benchmark/params_search/sweep.py
  2. Benchmark  — 7 planning jobs from benchmark/full_benchmark/submit.py
  3. Shading α  — 1 shading alpha tune job (eta=50) from submit_shading_alpha.py

Usage (from repo root, on the cluster):
    python benchmark/relaunch_failed.py --dry-run
    python benchmark/relaunch_failed.py
    python benchmark/relaunch_failed.py --group rl
    python benchmark/relaunch_failed.py --group benchmark
    python benchmark/relaunch_failed.py --group shading
"""
from __future__ import annotations

import sys
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import submitit

from benchmark.train_rl import TrainingConfig, train
from benchmark.full_benchmark.run import run_job
from benchmark.tune_shading_alpha import tune_shading_alpha
from benchmark.params_search.sweep import _train_job

SCRATCH_DIR = Path("/scratch/epatard")
_SHADING_OUT = SCRATCH_DIR / "full_benchmark" / "results" / "shading_alpha_results"

# ── SLURM parameters (match original submissions) ────────────────────────────

_RL_SLURM: dict[str, Any] = {
    "timeout_min":             2880,
    "slurm_partition":         "normal-best,gpu-best",
    "gpus_per_node":           0,
    "cpus_per_task":           8,
    "tasks_per_node":          1,
    "nodes":                   1,
    "mem_gb":                  8,
    "slurm_signal_delay_s":    120,
    "slurm_array_parallelism": 3,
    "slurm_job_name":          "dynac_sweep",
}

_BENCH_SLURM: dict[str, Any] = {
    "timeout_min":          2880,
    "slurm_partition":      "normal-best,gpu-best",
    "gpus_per_node":        0,
    "cpus_per_task":        32,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               16,
    "slurm_signal_delay_s": 120,
    "slurm_job_name":       "dynac_bench",
}

_SHADING_SLURM: dict[str, Any] = {
    "timeout_min":          2880,
    "slurm_partition":      "normal-best,gpu-best",
    "gpus_per_node":        0,
    "cpus_per_task":        8,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 60,
    "slurm_job_name":       "dynac_shading_alpha",
}

# ── Failed job definitions ────────────────────────────────────────────────────

# Group 1: TD3 RL sweep — 7 failed configs (178797_61/63/68/69/70/71/76)
_BASE_CFG = TrainingConfig(
    total_timesteps=5_000_000,
    scratch_dir=SCRATCH_DIR,
    verbose=False,
)

_RL_FAILED = [
    # (run_name, overrides)                                  original task
    ("lr=0.0003_bs=512_noise=0.1_warmup=500_gamma=0.99",   # 178797_61
     dict(learning_rate=3e-4, off_policy_batch_size=512, td3_action_noise_sigma=0.1,
          off_policy_learning_starts=500,  gamma=0.99)),
    ("lr=0.0003_bs=512_noise=0.1_warmup=1000_gamma=0.99",  # 178797_63
     dict(learning_rate=3e-4, off_policy_batch_size=512, td3_action_noise_sigma=0.1,
          off_policy_learning_starts=1000, gamma=0.99)),
    ("lr=0.0003_bs=512_noise=0.2_warmup=1000_gamma=0.95",  # 178797_68
     dict(learning_rate=3e-4, off_policy_batch_size=512, td3_action_noise_sigma=0.2,
          off_policy_learning_starts=1000, gamma=0.95)),
    ("lr=0.0003_bs=512_noise=0.2_warmup=1000_gamma=0.99",  # 178797_69
     dict(learning_rate=3e-4, off_policy_batch_size=512, td3_action_noise_sigma=0.2,
          off_policy_learning_starts=1000, gamma=0.99)),
    ("lr=0.0003_bs=512_noise=0.2_warmup=1500_gamma=0.95",  # 178797_70
     dict(learning_rate=3e-4, off_policy_batch_size=512, td3_action_noise_sigma=0.2,
          off_policy_learning_starts=1500, gamma=0.95)),
    ("lr=0.0003_bs=512_noise=0.2_warmup=1500_gamma=0.99",  # 178797_71
     dict(learning_rate=3e-4, off_policy_batch_size=512, td3_action_noise_sigma=0.2,
          off_policy_learning_starts=1500, gamma=0.99)),
    ("lr=0.001_bs=256_noise=0.1_warmup=1500_gamma=0.95",   # 178797_76
     dict(learning_rate=1e-3, off_policy_batch_size=256, td3_action_noise_sigma=0.1,
          off_policy_learning_starts=1500, gamma=0.95)),
]

# Group 2: Full benchmark — 7 failed run_job calls
# run_job(horizon_eta, job_type, mpc_horizon, n_bins, n_seeds, out_dir, mlp_model, n_workers)
_BENCH_FAILED = [
    # (eta, type,          H,     K, n_seeds)         original task
    (50,  "mpc",           10.0,  2, 1000),           # 200483_1
    (50,  "mpc_timebin",   10.0,  3, 1000),           # 201791_3
    (50,  "mpc_timebin",   25.0,  3, 1000),           # 201791_5
    (100, "mpc",           10.0,  2, 1000),           # 201791_6
    (100, "mpc_timebin",   10.0,  3, 1000),           # 201791_9
    (100, "mpc_timebin",   25.0,  2, 1000),           # 201791_10
    (100, "mpc_timebin",   25.0,  3, 1000),           # 201791_11
]

# Group 3: Shading alpha — 1 failed job (199134_0, eta=50)
# Same alpha grid as the completed eta=100 job (199134_1)
_SHADING_ALPHAS = list(np.linspace(0.6, 0.8, 21))
_SHADING_FAILED = [
    # (eta, alphas,        n_seeds, n_rollouts, policy_seed, out_dir)
    (50, _SHADING_ALPHAS, 1000,    10,          0,           _SHADING_OUT),  # 199134_0
]


def _submit_rl(executor: submitit.AutoExecutor, dry_run: bool) -> list[str]:
    configs = []
    for run_name, overrides in _RL_FAILED:
        cfg = replace(_BASE_CFG, run_name=run_name, **overrides)
        configs.append(("td3", cfg))

    if dry_run:
        for algo, cfg in configs:
            print(f"  [DRY RUN] _train_job(algo={algo}, run_name={cfg.run_name})")
        return []

    executor.update_parameters(**_RL_SLURM)
    jobs = executor.map_array(_train_job, [a for a, _ in configs], [c for _, c in configs])
    return [j.job_id for j in jobs]


def _submit_bench(executor: submitit.AutoExecutor, dry_run: bool) -> list[str]:
    # run_job positional order:
    #   horizon_eta, job_type, mpc_horizon, n_bins, mlp_model,
    #   ppo_model, sac_model, td3_model, algos, n_seeds, out_dir, n_workers
    job_args = [
        (eta, jt, H, K, None, None, None, None, None, n_seeds, None, None)
        for eta, jt, H, K, n_seeds in _BENCH_FAILED
    ]

    if dry_run:
        for eta, jt, H, K, n_seeds, *_ in job_args:
            print(f"  [DRY RUN] run_job(eta={eta}, type={jt}, H={H:g}, K={K}, n_seeds={n_seeds})")
        return []

    executor.update_parameters(**_BENCH_SLURM)
    jobs = executor.map_array(run_job, *zip(*job_args))
    return [j.job_id for j in jobs]


def _submit_shading(executor: submitit.AutoExecutor, dry_run: bool) -> list[str]:
    if dry_run:
        for eta, alphas, n_seeds, *_ in _SHADING_FAILED:
            print(f"  [DRY RUN] tune_shading_alpha(eta={eta}, n_alphas={len(alphas)}, "
                  f"alpha=[{alphas[0]:.2f}..{alphas[-1]:.2f}], n_seeds={n_seeds})")
        return []

    executor.update_parameters(**_SHADING_SLURM)
    jobs = executor.map_array(tune_shading_alpha, *zip(*_SHADING_FAILED))
    return [j.job_id for j in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--group", choices=["rl", "benchmark", "shading"], nargs="+",
        default=["rl", "benchmark", "shading"],
        help="Which groups of failed jobs to relaunch (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = set(args.group)
    dry   = args.dry_run

    rl_executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "%j"))
    bench_executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "full_benchmark" / "%j"))
    shading_executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "shading_alpha" / "%j"))

    all_ids: list[str] = []

    if "rl" in groups:
        print(f"\n[RL sweep — {len(_RL_FAILED)} TD3 jobs]")
        ids = _submit_rl(rl_executor, dry)
        all_ids.extend(ids)
        if ids:
            print(f"  Submitted {len(ids)} jobs — IDs: {ids[0]} … {ids[-1]}")

    if "benchmark" in groups:
        print(f"\n[Full benchmark — {len(_BENCH_FAILED)} planning jobs]")
        ids = _submit_bench(bench_executor, dry)
        all_ids.extend(ids)
        if ids:
            print(f"  Submitted {len(ids)} jobs — IDs: {ids[0]} … {ids[-1]}")

    if "shading" in groups:
        print(f"\n[Shading α — {len(_SHADING_FAILED)} job]")
        ids = _submit_shading(shading_executor, dry)
        all_ids.extend(ids)
        if ids:
            print(f"  Submitted {len(ids)} jobs — IDs: {ids}")

    if dry:
        print(f"\nTotal: {len(_RL_FAILED) + len(_BENCH_FAILED) + len(_SHADING_FAILED)} jobs "
              f"(dry run, nothing submitted)")
    else:
        print(f"\nTotal submitted: {len(all_ids)} jobs")
        print(f"Check status   : squeue -u epatard")


if __name__ == "__main__":
    main()
