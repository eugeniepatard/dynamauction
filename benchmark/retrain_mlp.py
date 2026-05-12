"""
Retrain the MLPValuationHead on the cluster (SLURM via submitit) for a given
squashing function and horizon.  Supports hyperparameter sweeps over
architectures, activations, and episode counts — one SLURM job per combination.

The MLP approximates V0(s, x) = E[reward | win, state s, display x] and is
used by all MPC-family policies. It must be retrained whenever the scenario
changes — in particular when switching the SurvivalKernel squashing from
'clip' (paper default) to 'sigmoid'.

Models are saved to:
    benchmark/models/mlp_valuation_head/                   (clip)
    benchmark/models/mlp_valuation_head/squashing=sigmoid/ (sigmoid)

Architecture syntax
-------------------
Each --archs value is a comma-separated list of layer widths:
    --archs 64,64          -> [64, 64]
    --archs 64,64 128,128  -> two jobs: [64,64] and [128,128]

Usage (from repo root)
----------------------
    # Dry run — print all combinations without submitting
    python -m benchmark.retrain_mlp --dry-run \\
        --archs 64,64 128,128 \\
        --activations relu tanh \\
        --n-episodes 200 500

    # Submit sweep to SLURM (one job per combination)
    python -m benchmark.retrain_mlp --squashing sigmoid \\
        --archs 64,64 128,128 64,64,64 \\
        --activations relu tanh leaky_relu \\
        --n-episodes 500

    # Single run on local machine (no SLURM)
    python -m benchmark.retrain_mlp --squashing sigmoid \\
        --archs 128,128 --activations relu --n-episodes 500 \\
        --local --tensorboard
"""
from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import submitit

from benchmark.train_rl import SCENARIO_ID, LOGS_DIR
from benchmark.train_mlp import train_mlp, mlp_save_dir

# ── Cluster paths ─────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("/scratch/epatard")

# ── Paper defaults ────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    archs       = ["128,128"],
    activations = ["tanh"],
    n_episodes  = [500],
    val_frac    = 0.15,
    n_epochs    = 500,
    lr          = 1e-3,
    batch_size  = 256,
    seed        = 42,
)

# ── Margaret SLURM parameters ─────────────────────────────────────────────────

SLURM_PARAMS: dict[str, Any] = {
    "timeout_min":          480,
    "slurm_partition":      "normal-best,gpu-best",
    "gpus_per_node":        0,
    "cpus_per_task":        4,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 60,
    "slurm_job_name":       "dynac_mlp_sweep",
}


def _parse_arch(s: str) -> List[int]:
    """'64,64' -> [64, 64]"""
    try:
        return [int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid architecture {s!r}: expected comma-separated ints, e.g. '128,128'."
        )


def _build_combinations(args: argparse.Namespace):
    """Return list of (hidden_dims, activation, n_episodes) for the sweep."""
    archs = [_parse_arch(a) for a in args.archs]
    return list(itertools.product(archs, args.activations, args.n_episodes))


def _mlp_job(
    squashing:   str,
    horizon_eta: int,
    hidden_dims: List[int],
    activation:  str,
    n_episodes:  int,
    val_frac:    float,
    n_epochs:    int,
    lr:          float,
    batch_size:  int,
    seed:        int,
    tensorboard: bool,
) -> None:
    """Callable submitted to SLURM (or run locally)."""
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
        scenario_id        = SCENARIO_ID,
        scenario_overrides = scenario_overrides,
        n_episodes         = n_episodes,
        val_frac           = val_frac,
        hidden_dims        = hidden_dims,
        activation         = activation,
        n_epochs           = n_epochs,
        lr                 = lr,
        batch_size         = batch_size,
        seed               = seed,
        save_dir           = save_dir,
        log_dir            = log_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Scenario ──────────────────────────────────────────────────────────────
    parser.add_argument("--squashing", choices=["sigmoid", "clip"], default="sigmoid",
                        help="SurvivalKernel squashing (default: sigmoid).")
    parser.add_argument("--horizon-eta", type=int, default=50,
                        help="Exponential horizon rate η (default: 50).")

    # ── Sweep axes ────────────────────────────────────────────────────────────
    parser.add_argument("--archs", nargs="+", default=_DEFAULTS["archs"],
                        metavar="W,W,...",
                        help="Architectures to sweep. Each value is a comma-separated "
                             "list of layer widths. Multiple values = one job per arch. "
                             f"(default: {_DEFAULTS['archs']})")
    parser.add_argument("--activations", nargs="+", default=_DEFAULTS["activations"],
                        choices=["tanh", "relu", "leaky_relu"],
                        help=f"Activations to sweep (default: {_DEFAULTS['activations']}).")
    parser.add_argument("--n-episodes", nargs="+", type=int, default=_DEFAULTS["n_episodes"],
                        metavar="N",
                        help=f"Episode counts to sweep (default: {_DEFAULTS['n_episodes']}).")

    # ── Fixed training params ─────────────────────────────────────────────────
    parser.add_argument("--val-frac",   type=float, default=_DEFAULTS["val_frac"],
                        help=f"Validation fraction (default: {_DEFAULTS['val_frac']}).")
    parser.add_argument("--n-epochs",   type=int,   default=_DEFAULTS["n_epochs"],
                        help=f"Training epochs (default: {_DEFAULTS['n_epochs']}).")
    parser.add_argument("--lr",         type=float, default=_DEFAULTS["lr"],
                        help=f"Adam learning rate (default: {_DEFAULTS['lr']}).")
    parser.add_argument("--batch-size", type=int,   default=_DEFAULTS["batch_size"],
                        help=f"Mini-batch size (default: {_DEFAULTS['batch_size']}).")
    parser.add_argument("--seed",       type=int,   default=_DEFAULTS["seed"],
                        help=f"Random seed (default: {_DEFAULTS['seed']}).")

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write TensorBoard logs (one subdirectory per run).")

    # ── Execution mode ────────────────────────────────────────────────────────
    parser.add_argument("--max-jobs", type=int, default=3,
                        help="Maximum number of SLURM jobs running concurrently "
                             "(default: 3). Ignored with --local.")
    parser.add_argument("--local", action="store_true",
                        help="Run locally (no SLURM). When sweeping, jobs run "
                             "sequentially. Intended for single-combination runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all combinations without running anything.")
    parser.add_argument("--log-folder", default=str(SCRATCH_DIR / "submitit_logs" / "%j"),
                        help="submitit log folder template (%%j = SLURM job id).")

    args = parser.parse_args()

    combinations = _build_combinations(args)
    save_dir     = mlp_save_dir(args.squashing)

    # ── Shared kwargs (constant across sweep) ─────────────────────────────────
    shared = dict(
        squashing   = args.squashing,
        horizon_eta = args.horizon_eta,
        val_frac    = args.val_frac,
        n_epochs    = args.n_epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        seed        = args.seed,
        tensorboard = args.tensorboard,
    )

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{len(combinations)} combination(s)  "
              f"squashing={args.squashing}  eta={args.horizon_eta}\n")
        for i, (hidden_dims, activation, n_episodes) in enumerate(combinations):
            dims_str = "-".join(str(d) for d in hidden_dims)
            print(f"  [{i:2d}]  h={dims_str:12s}  act={activation:10s}  ep={n_episodes}")
        print(f"\n  save_dir : {save_dir}")
        print(f"  n_epochs : {args.n_epochs}  lr={args.lr}  "
              f"batch={args.batch_size}  seed={args.seed}")
        print("\n(dry run — nothing submitted)")
        return

    # ── Local run (sequential) ────────────────────────────────────────────────
    if args.local:
        if len(combinations) > 1:
            print(f"[INFO] Running {len(combinations)} combination(s) sequentially "
                  f"(local mode). Use --dry-run to preview.")
        for hidden_dims, activation, n_episodes in combinations:
            dims_str = "-".join(str(d) for d in hidden_dims)
            print(f"\n--- h={dims_str}  act={activation}  ep={n_episodes} ---")
            _mlp_job(hidden_dims=hidden_dims, activation=activation,
                     n_episodes=n_episodes, **shared)
        return

    # ── SLURM array (max 3 jobs running concurrently) ────────────────────────
    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(**SLURM_PARAMS, slurm_array_parallelism=args.max_jobs)

    # map_array requires a single-argument callable; wrap shared kwargs in closure
    def _run(combo):
        hidden_dims, activation, n_episodes = combo
        _mlp_job(hidden_dims=hidden_dims, activation=activation,
                 n_episodes=n_episodes, **shared)

    jobs = executor.map_array(_run, combinations)

    print(f"\nSubmitted {len(jobs)} job(s) as array  "
          f"(max {args.max_jobs} concurrent)  "
          f"squashing={args.squashing}  eta={args.horizon_eta}")
    for job, (hidden_dims, activation, n_episodes) in zip(jobs, combinations):
        dims_str = "-".join(str(d) for d in hidden_dims)
        print(f"  [{job.job_id}]  h={dims_str}  act={activation}  ep={n_episodes}")

    print(f"\n  Models -> {save_dir}")
    if args.tensorboard:
        print(f"  Logs   -> {LOGS_DIR / 'mlp_valuation_head'}")
        print(f"  View   : tensorboard --logdir {LOGS_DIR / 'mlp_valuation_head'}")



if __name__ == "__main__":
    main()