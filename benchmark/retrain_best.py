"""
Retrain the best PPO / SAC / TD3 configuration on the cluster (SLURM via
submitit), optionally switching the SurvivalKernel probability mapping between
'clip' (paper default) and 'sigmoid'.

Best hyperparameters are hardcoded from the sweep results (see
Table~\ref{tab:rl_hyperparams} in the paper).

Usage (from repo root, on the Margaret cluster):
    # dry run — print jobs without submitting
    python -m benchmark.retrain_best --dry-run

    # submit with sigmoid squashing (default for this script)
    python -m benchmark.retrain_best

    # submit with clip squashing (matches paper results)
    python -m benchmark.retrain_best --squashing clip

    # one algo, custom horizon
    python -m benchmark.retrain_best --algo ppo --horizon-eta 100

    # custom timesteps
    python -m benchmark.retrain_best --timesteps 10_000_000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import submitit

from benchmark.train_rl import TrainingConfig, train

# ── Cluster paths ─────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("/scratch/epatard")

# ── Best hyperparameters (from sweep, see Table tab:rl_hyperparams) ───────────

# Clip best configs (grid sweep, η=50)
_BEST_CLIP_CONFIGS: dict[str, dict[str, Any]] = {
    "ppo": dict(
        learning_rate=3e-4,
        ppo_ent_coef=0.02,
        ppo_n_steps=512,
        gamma=0.95,
        ppo_n_epochs=5,
    ),
    "sac": dict(
        learning_rate=5e-5,
        off_policy_batch_size=512,
        off_policy_learning_starts=500,
        gamma=0.95,
    ),
    "td3": dict(
        learning_rate=1e-4,
        off_policy_batch_size=512,
        td3_action_noise_sigma=0.1,
        off_policy_learning_starts=500,
        gamma=0.99,
    ),
}

# Sigmoid best configs (Bayesian sweep, η=50)
_BEST_SIGMOID_CONFIGS: dict[str, dict[str, Any]] = {
    "ppo": dict(
        learning_rate=3.083e-4,
        ppo_ent_coef=0.02038,
        ppo_n_steps=1024,
        gamma=0.99,
        ppo_n_epochs=2,
    ),
    "sac": dict(
        learning_rate=8.844e-4,
        off_policy_batch_size=256,
        off_policy_learning_starts=1500,
        gamma=0.99,
    ),
    "td3": dict(
        learning_rate=1.223e-4,
        off_policy_batch_size=256,
        td3_action_noise_sigma=0.07322,
        off_policy_learning_starts=500,
        gamma=0.99,
    ),
}

_BEST_CONFIGS_BY_SQUASHING: dict[str, dict[str, dict[str, Any]]] = {
    "clip":    _BEST_CLIP_CONFIGS,
    "sigmoid": _BEST_SIGMOID_CONFIGS,
}

# ── Margaret SLURM parameters ─────────────────────────────────────────────────

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":          2880,                   # 48 h
    "slurm_partition":      "normal-best,gpu-best",
    "gpus_per_node":        0,
    "cpus_per_task":        8,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 120,
    "slurm_job_name":       "dynac_retrain_best",
}


def _build_config(algo: str, squashing: str, timesteps: int, horizon_eta: int) -> TrainingConfig:
    run_name = f"best_{algo}_squashing={squashing}_eta{horizon_eta}"
    best_configs = _BEST_CONFIGS_BY_SQUASHING[squashing]
    return TrainingConfig(
        total_timesteps=timesteps,
        scratch_dir=SCRATCH_DIR,
        verbose=False,
        run_name=run_name,
        **best_configs[algo],
    )


def _train_job(algo: str, cfg: TrainingConfig, squashing: str, horizon_eta: int) -> None:
    """Callable submitted to SLURM."""
    scenario_overrides = {
        "horizon_eta":      horizon_eta,
        "kernel_squashing": squashing,
    }
    train(algo, cfg, scenario_overrides=scenario_overrides)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo", nargs="+", choices=["ppo", "sac", "td3"],
        default=["ppo", "sac", "td3"],
        help="Algorithms to retrain (default: all three).",
    )
    parser.add_argument(
        "--squashing", choices=["sigmoid", "clip"], default="sigmoid",
        help="Probability squashing in SurvivalKernel (default: sigmoid).",
    )
    parser.add_argument(
        "--horizon-eta", type=int, default=50,
        help="Exponential horizon rate η (default: 50).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Training timesteps per job (default: 5M).",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run directly in-process instead of submitting to SLURM "
             "(use this on a machine without SLURM).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job configs without submitting.",
    )
    parser.add_argument(
        "--log-folder", default=str(SCRATCH_DIR / "submitit_logs" / "%j"),
        help="submitit log folder template (%%j = SLURM job id).",
    )
    args = parser.parse_args()

    configs = {
        algo: _build_config(algo, args.squashing, args.timesteps, args.horizon_eta)
        for algo in args.algo
    }

    if args.dry_run:
        print(f"\nsquashing={args.squashing}  horizon_eta={args.horizon_eta}  timesteps={args.timesteps:,}\n")
        for algo, cfg in configs.items():
            print(f"  {algo.upper():4s}  run_name={cfg.run_name}")
            for k, v in _BEST_CONFIGS_BY_SQUASHING[args.squashing][algo].items():
                print(f"         {k}={v}")
        print("\n(dry run — nothing submitted)")
        return

    if args.local:
        for algo, cfg in configs.items():
            print(f"[{algo.upper()}] Running locally ({cfg.run_name})")
            _train_job(algo, cfg, args.squashing, args.horizon_eta)
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS)

    for algo, cfg in configs.items():
        job = executor.submit(_train_job, algo, cfg, args.squashing, args.horizon_eta)
        print(f"[{algo.upper()}] Submitted job {job.job_id}  ({cfg.run_name})")


if __name__ == "__main__":
    main()