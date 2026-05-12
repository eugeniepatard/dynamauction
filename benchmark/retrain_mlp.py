"""
Retrain the MLPValuationHead on the cluster (SLURM via submitit) for a given
squashing function and horizon.

The MLP approximates V0(s, x) = E[reward | win, state s, display x] and is
used by all MPC-family policies. It must be retrained whenever the scenario
changes — in particular when switching the SurvivalKernel squashing from
'clip' (paper default) to 'sigmoid'.

Models are saved to:
    benchmark/models/mlp_valuation_head/                  (clip, default)
    benchmark/models/mlp_valuation_head/squashing=sigmoid/ (sigmoid)

Usage (from repo root, on the Margaret cluster):
    # dry run
    python -m benchmark.retrain_mlp --dry-run

    # train with sigmoid squashing (the main use case for this script)
    python -m benchmark.retrain_mlp --squashing sigmoid

    # train with clip squashing (paper default, same as train_mlp.py)
    python -m benchmark.retrain_mlp --squashing clip

    # custom horizon and architecture
    python -m benchmark.retrain_mlp --squashing sigmoid --horizon-eta 100 \\
        --hidden-dims 128 128 --n-episodes 500 --n-epochs 500
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import submitit

from benchmark.train_rl import SCENARIO_ID, LOGS_DIR
from benchmark.train_mlp import train_mlp, mlp_save_dir

# ── Cluster paths ─────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("/scratch/epatard")

# ── Paper hyperparameters for the MLP valuation head ─────────────────────────

_PAPER_MLP_PARAMS = dict(
    n_episodes  = 500,
    val_frac    = 0.15,
    hidden_dims = [128, 128],
    activation  = "tanh",
    n_epochs    = 500,
    lr          = 1e-3,
    batch_size  = 256,
    seed        = 42,
)

# ── Margaret SLURM parameters ─────────────────────────────────────────────────

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":          480,                    # 8 h (MLP training is fast)
    "slurm_partition":      "normal-best,gpu-best",
    "gpus_per_node":        0,
    "cpus_per_task":        4,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 60,
    "slurm_job_name":       "dynac_retrain_mlp",
}


def _mlp_job(
    squashing:   str,
    horizon_eta: int,
    n_episodes:  int,
    val_frac:    float,
    hidden_dims: List[int],
    activation:  str,
    n_epochs:    int,
    lr:          float,
    batch_size:  int,
    seed:        int,
    tensorboard: bool,
) -> None:
    """Callable submitted to SLURM."""
    scenario_overrides = {
        "horizon_eta":      horizon_eta,
        "kernel_squashing": squashing,
    }

    save_dir = mlp_save_dir(squashing)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir: Optional[Path] = None
    if tensorboard:
        dims_str  = "-".join(str(d) for d in hidden_dims)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name  = (
            f"squashing={squashing}_eta{horizon_eta}"
            f"_act={activation}_h{dims_str}"
            f"_ep{n_episodes}_e{n_epochs}_lr{lr:g}_s{seed}"
            f"_{timestamp}"
        )
        log_dir = LOGS_DIR / "mlp_valuation_head" / run_name

    train_mlp(
        scenario_id=SCENARIO_ID,
        scenario_overrides=scenario_overrides,
        n_episodes=n_episodes,
        val_frac=val_frac,
        hidden_dims=hidden_dims,
        activation=activation,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        save_dir=save_dir,
        log_dir=log_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Scenario ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--squashing", choices=["sigmoid", "clip"], default="sigmoid",
        help="SurvivalKernel probability squashing (default: sigmoid).",
    )
    parser.add_argument(
        "--horizon-eta", type=int, default=50,
        help="Exponential horizon rate η (default: 50).",
    )

    # ── MLP architecture and training ─────────────────────────────────────────
    parser.add_argument("--n-episodes",  type=int,   default=_PAPER_MLP_PARAMS["n_episodes"],
                        help=f"Rollout episodes for data collection (default: {_PAPER_MLP_PARAMS['n_episodes']}).")
    parser.add_argument("--val-frac",    type=float, default=_PAPER_MLP_PARAMS["val_frac"],
                        help=f"Validation fraction (default: {_PAPER_MLP_PARAMS['val_frac']}).")
    parser.add_argument("--hidden-dims", type=int,   nargs="+", default=_PAPER_MLP_PARAMS["hidden_dims"],
                        help=f"Hidden layer widths (default: {_PAPER_MLP_PARAMS['hidden_dims']}).")
    parser.add_argument("--activation",  type=str,   default=_PAPER_MLP_PARAMS["activation"],
                        choices=["tanh", "relu", "leaky_relu"],
                        help=f"Hidden layer activation (default: {_PAPER_MLP_PARAMS['activation']}).")
    parser.add_argument("--n-epochs",    type=int,   default=_PAPER_MLP_PARAMS["n_epochs"],
                        help=f"Training epochs (default: {_PAPER_MLP_PARAMS['n_epochs']}).")
    parser.add_argument("--lr",          type=float, default=_PAPER_MLP_PARAMS["lr"],
                        help=f"Adam learning rate (default: {_PAPER_MLP_PARAMS['lr']}).")
    parser.add_argument("--batch-size",  type=int,   default=_PAPER_MLP_PARAMS["batch_size"],
                        help=f"Mini-batch size (default: {_PAPER_MLP_PARAMS['batch_size']}).")
    parser.add_argument("--seed",        type=int,   default=_PAPER_MLP_PARAMS["seed"],
                        help=f"Random seed (default: {_PAPER_MLP_PARAMS['seed']}).")

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write TensorBoard logs during training.")

    # ── Cluster ───────────────────────────────────────────────────────────────
    parser.add_argument("--local", action="store_true",
                        help="Run directly in-process instead of submitting to SLURM "
                             "(use this on a machine without SLURM).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job config without submitting.")
    parser.add_argument("--log-folder", default=str(SCRATCH_DIR / "submitit_logs" / "%j"),
                        help="submitit log folder template (%%j = SLURM job id).")
    args = parser.parse_args()

    save_dir = mlp_save_dir(args.squashing)

    if args.dry_run:
        print(f"\nsquashing={args.squashing}  horizon_eta={args.horizon_eta}")
        print(f"  save_dir    : {save_dir}")
        print(f"  hidden_dims : {args.hidden_dims}")
        print(f"  activation  : {args.activation}")
        print(f"  n_episodes  : {args.n_episodes}")
        print(f"  n_epochs    : {args.n_epochs}")
        print(f"  lr          : {args.lr}")
        print(f"  batch_size  : {args.batch_size}")
        print(f"  seed        : {args.seed}")
        print(f"  tensorboard : {args.tensorboard}")
        print("\n(dry run — nothing submitted)")
        return

    job_kwargs = dict(
        squashing   = args.squashing,
        horizon_eta = args.horizon_eta,
        n_episodes  = args.n_episodes,
        val_frac    = args.val_frac,
        hidden_dims = args.hidden_dims,
        activation  = args.activation,
        n_epochs    = args.n_epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        seed        = args.seed,
        tensorboard = args.tensorboard,
    )

    if args.local:
        print(f"Running locally (squashing={args.squashing}, eta={args.horizon_eta})")
        print(f"  Model will be saved to: {save_dir}")
        _mlp_job(**job_kwargs)
        return

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS)

    job = executor.submit(_mlp_job, **job_kwargs)
    print(f"Submitted job {job.job_id}  (squashing={args.squashing}, eta={args.horizon_eta})")
    print(f"  Model will be saved to: {save_dir}")


if __name__ == "__main__":
    main()