"""
Hyperparameter sweep for PPO / SAC / TD3 on the Inria Margaret (SLURM) cluster.

Two search modes
----------------
grid (default)
    Every combination in the pre-defined grids becomes one independent SLURM job.
    Results land in:
        /scratch/epatard/models/<algo>/<run_name>/best_model.zip
        /scratch/epatard/logs/<algo>/<run_name>/

bayesian
    Optuna TPE sampler, one SLURM job per trial.  Each job asks for its own
    hyper-parameters from a shared Optuna study (JournalFileBackend on NFS),
    trains, then reports the result back.
    Study files land in:
        /scratch/epatard/optuna/<algo>_eta<eta>_probfn<prob_fn>.log

Scenario knobs (both modes)
---------------------------
--horizon-eta FLOAT    Exponential-horizon parameter η  (default 50)
--prob-fn clip|sigmoid  Survival-kernel probability mapping  (default clip)

Usage (from repo root, on the cluster):
    # dry run — print all jobs without submitting
    python benchmark/params_search/sweep.py --dry-run

    # grid, PPO only, sigmoid, eta=100
    python benchmark/params_search/sweep.py --algo ppo --prob-fn sigmoid --horizon-eta 100

    # Bayesian, all algos, 40 trials each, sigmoid
    python benchmark/params_search/sweep.py --search bayesian --n-trials 40 --prob-fn sigmoid

    # submit all algos (grid, default settings)
    python benchmark/params_search/sweep.py --algo ppo sac td3
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import submitit

from benchmark.train_rl import TrainingConfig, train

# ── Grid search spaces ────────────────────────────────────────────────────────

PPO_GRID: dict[str, list] = {
    "learning_rate": [5e-5, 1e-4, 2e-4, 3e-4],
    "ppo_ent_coef":  [0.0, 0.01, 0.02],
    "ppo_n_steps":   [256, 512],
    "gamma":         [0.95, 0.99],
    "ppo_n_epochs":  [2, 5, 10],
}

SAC_GRID: dict[str, list] = {
    "learning_rate":              [5e-5, 1e-4, 3e-4, 1e-3],
    "off_policy_batch_size":      [256, 512],
    "off_policy_learning_starts": [500, 1_000, 1_500],
    "gamma":                      [0.95, 0.99],
}

TD3_GRID: dict[str, list] = {
    "learning_rate":              [5e-5, 1e-4, 3e-4, 1e-3],
    "off_policy_batch_size":      [256, 512],
    "td3_action_noise_sigma":     [0.1, 0.2],
    "off_policy_learning_starts": [500, 1_000, 1_500],
    "gamma":                      [0.95, 0.99],
}

GRIDS = {"ppo": PPO_GRID, "sac": SAC_GRID, "td3": TD3_GRID}

# ── Bayesian search spaces ────────────────────────────────────────────────────
# Each entry: (suggest_method, *args, **kwargs)
# suggest_method is called on an optuna.Trial object.

def _suggest_params(trial: Any, algo: str) -> dict[str, Any]:
    """Sample one set of hyper-parameters from an Optuna trial."""
    if algo == "ppo":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
            "ppo_ent_coef":  trial.suggest_float("ppo_ent_coef", 0.0, 0.05),
            "ppo_n_steps":   trial.suggest_categorical("ppo_n_steps", [256, 512, 1024]),
            "gamma":         trial.suggest_categorical("gamma", [0.95, 0.99]),
            "ppo_n_epochs":  trial.suggest_categorical("ppo_n_epochs", [2, 5, 10]),
        }
    if algo == "sac":
        return {
            "learning_rate":              trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
            "off_policy_batch_size":      trial.suggest_categorical("off_policy_batch_size", [256, 512]),
            "off_policy_learning_starts": trial.suggest_categorical("off_policy_learning_starts", [500, 1_000, 1_500]),
            "gamma":                      trial.suggest_categorical("gamma", [0.95, 0.99]),
        }
    if algo == "td3":
        return {
            "learning_rate":              trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
            "off_policy_batch_size":      trial.suggest_categorical("off_policy_batch_size", [256, 512]),
            "td3_action_noise_sigma":     trial.suggest_float("td3_action_noise_sigma", 0.05, 0.3),
            "off_policy_learning_starts": trial.suggest_categorical("off_policy_learning_starts", [500, 1_000, 1_500]),
            "gamma":                      trial.suggest_categorical("gamma", [0.95, 0.99]),
        }
    raise ValueError(f"Unknown algo {algo!r}")


# ── Margaret SLURM parameters ─────────────────────────────────────────────────

SCRATCH_DIR = Path("/scratch/epatard")

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":               2880,
    "slurm_partition":           "normal-best,gpu-best",
    "gpus_per_node":             0,
    "cpus_per_task":             8,
    "tasks_per_node":            1,
    "nodes":                     1,
    "mem_gb":                    8,
    "slurm_signal_delay_s":      120,
    "slurm_array_parallelism":   3,
    "slurm_job_name":            "dynac_sweep",
}

_SUBMITIT_LOG_FOLDER = str(SCRATCH_DIR / "submitit_logs" / "%j")

# ── Naming helpers ────────────────────────────────────────────────────────────

_KEY_ALIASES = {
    "learning_rate":              "lr",
    "ppo_ent_coef":               "ent",
    "ppo_n_steps":                "steps",
    "ppo_n_epochs":               "epochs",
    "ppo_gae_lambda":             "gae",
    "off_policy_batch_size":      "bs",
    "off_policy_learning_starts": "warmup",
    "td3_action_noise_sigma":     "noise",
}


def _make_run_name(overrides: dict[str, Any]) -> str:
    parts = [f"{_KEY_ALIASES.get(k, k)}={v}" for k, v in overrides.items()]
    return "_".join(parts)


def study_name(algo: str, horizon_eta: float, prob_fn: str) -> str:
    return f"{algo}_eta{int(horizon_eta)}_probfn{prob_fn}"


def study_storage_path(algo: str, horizon_eta: float, prob_fn: str) -> Path:
    return SCRATCH_DIR / "optuna" / f"{study_name(algo, horizon_eta, prob_fn)}.log"


# ── Grid job ──────────────────────────────────────────────────────────────────

def _grid_train_job(algo: str, cfg: TrainingConfig, scenario_overrides: dict) -> None:
    train(algo, cfg, scenario_overrides=scenario_overrides)


def _expand_grid(algo: str, total_timesteps: int) -> list[tuple[TrainingConfig, str]]:
    base = TrainingConfig(
        total_timesteps=total_timesteps,
        scratch_dir=SCRATCH_DIR,
        verbose=False,
    )
    grid = GRIDS[algo]
    out = []
    for values in itertools.product(*grid.values()):
        overrides = dict(zip(grid.keys(), values))
        run_name  = _make_run_name(overrides)
        cfg       = replace(base, run_name=run_name, **overrides)
        out.append((cfg, run_name))
    return out


# ── Bayesian job ──────────────────────────────────────────────────────────────

def _bayesian_train_job(
    algo: str,
    base_cfg: TrainingConfig,
    storage_path: str,
    sname: str,
    scenario_overrides: dict,
) -> float:
    """One SLURM trial: ask Optuna → train → tell result."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileBackend(storage_path)
    )
    study = optuna.load_study(study_name=sname, storage=storage)
    trial = study.ask()

    try:
        overrides = _suggest_params(trial, algo)
        run_name  = f"t{trial.number:04d}_" + _make_run_name(overrides)
        cfg       = replace(base_cfg, run_name=run_name, **overrides)

        train(algo, cfg, scenario_overrides=scenario_overrides)

        log_dir = (Path(cfg.scratch_dir) if cfg.scratch_dir else Path(".")) / "logs" / algo / run_name
        data    = np.load(str(log_dir / "evaluations.npz"))
        reward  = float(data["results"].mean(axis=1).max())

        study.tell(trial, reward)
        return reward

    except Exception as exc:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise exc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo", nargs="+", choices=["ppo", "sac", "td3"],
        default=["ppo", "sac", "td3"],
    )
    parser.add_argument(
        "--search", choices=["grid", "bayesian"], default="grid",
        help="Search strategy (default: grid)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Bayesian trials per algo (bayesian mode only, default: 50)",
    )
    parser.add_argument(
        "--horizon-eta", type=float, default=50.0,
        metavar="ETA",
        help="Exponential-horizon parameter η (default: 50)",
    )
    parser.add_argument(
        "--prob-fn", choices=["clip", "sigmoid"], default="clip",
        help="Survival-kernel probability mapping (default: clip)",
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

    scenario_overrides = {
        "horizon_eta":      args.horizon_eta,
        "kernel_squashing": args.prob_fn,
    }

    if args.search == "grid":
        _run_grid(args, scenario_overrides)
    else:
        _run_bayesian(args, scenario_overrides)


def _run_grid(args: Any, scenario_overrides: dict) -> None:
    total = 0
    all_combos: list[tuple[str, TrainingConfig, str]] = []

    for algo in args.algo:
        combos = _expand_grid(algo, args.timesteps)
        total += len(combos)
        if args.dry_run:
            print(f"\n{algo.upper()}: {len(combos)} jobs  "
                  f"[η={args.horizon_eta}, prob_fn={args.prob_fn}]")
            for _, run_name in combos:
                print(f"  {run_name}")
        else:
            all_combos.extend((algo, cfg, run_name) for cfg, run_name in combos)

    if args.dry_run:
        print(f"\nTotal: {total} jobs (dry run, nothing submitted)")
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS)

    for algo in args.algo:
        combos = _expand_grid(algo, args.timesteps)
        n      = len(combos)
        jobs   = executor.map_array(
            _grid_train_job,
            [algo] * n,
            [cfg for cfg, _ in combos],
            [scenario_overrides] * n,
        )
        ids = [j.job_id for j in jobs]
        print(f"[{algo.upper()}] {n} jobs submitted — IDs {ids[0]} … {ids[-1]}")

    print(f"\nTotal: {total} jobs submitted.  Logs → {args.log_folder}")


def _run_bayesian(args: Any, scenario_overrides: dict) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_cfg = TrainingConfig(
        total_timesteps=args.timesteps,
        scratch_dir=SCRATCH_DIR,
        verbose=False,
    )

    if args.dry_run:
        for algo in args.algo:
            sname    = study_name(algo, args.horizon_eta, args.prob_fn)
            storage  = study_storage_path(algo, args.horizon_eta, args.prob_fn)
            print(f"\n{algo.upper()}: {args.n_trials} Bayesian trials  "
                  f"[η={args.horizon_eta}, prob_fn={args.prob_fn}]")
            print(f"  study    : {sname}")
            print(f"  storage  : {storage}")
        print(f"\nTotal: {len(args.algo) * args.n_trials} jobs (dry run, nothing submitted)")
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS)

    for algo in args.algo:
        sname   = study_name(algo, args.horizon_eta, args.prob_fn)
        storage_path = study_storage_path(algo, args.horizon_eta, args.prob_fn)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileBackend(str(storage_path))
        )
        optuna.create_study(
            study_name=sname,
            storage=storage,
            direction="maximize",
            load_if_exists=True,         # resume if already exists
        )

        n     = args.n_trials
        jobs  = executor.map_array(
            _bayesian_train_job,
            [algo]             * n,
            [base_cfg]         * n,
            [str(storage_path)] * n,
            [sname]            * n,
            [scenario_overrides] * n,
        )
        ids = [j.job_id for j in jobs]
        print(f"[{algo.upper()}] {n} Bayesian trials submitted — IDs {ids[0]} … {ids[-1]}")
        print(f"  study   : {sname}")
        print(f"  storage : {storage_path}")


if __name__ == "__main__":
    main()
