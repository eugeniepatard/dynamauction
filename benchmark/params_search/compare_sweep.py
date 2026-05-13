"""
Compare hyperparameter sweep results to identify the best configuration per algorithm.

Two modes mirror sweep.py
--------------------------
grid (default)
    Scans benchmark/params_search/logs/<algo>/<run_name>/evaluations.npz
    (written by SB3's EvalCallback) and builds a ranked DataFrame.

bayesian
    Reads directly from the Optuna JournalFileBackend written by sweep.py.
    Requires --horizon-eta and --prob-fn to locate the right study file.
    Supplements with evaluations.npz reward data when available.

Usage:
    python benchmark/params_search/compare_sweep.py                              # grid, all algos
    python benchmark/params_search/compare_sweep.py --algo ppo --top 5          # grid, PPO top 5
    python benchmark/params_search/compare_sweep.py --csv results/sweep.csv     # grid + save CSV

    python benchmark/params_search/compare_sweep.py \\
        --search bayesian --horizon-eta 50 --prob-fn sigmoid                     # bayesian results
    python benchmark/params_search/compare_sweep.py \\
        --search bayesian --horizon-eta 50 --prob-fn sigmoid --top 5 --algo ppo  # top 5 PPO trials
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

SCRATCH_DIR = Path("/scratch/epatard")
LOGS_DIR    = SCRATCH_DIR / "logs"

# ── Grid: result loading ──────────────────────────────────────────────────────

def _load_eval(log_dir: Path) -> Union[Dict[str, Any], None]:
    """Read evaluations.npz, return summary stats or None if missing."""
    npz = log_dir / "evaluations.npz"
    if not npz.exists():
        return None
    data         = np.load(npz)
    results      = data["results"]
    timesteps    = data["timesteps"]
    mean_rewards = results.mean(axis=1)
    return {
        "best_mean_reward":  float(mean_rewards.max()),
        "final_mean_reward": float(mean_rewards[-1]),
        "n_evals":           int(len(timesteps)),
        "timesteps_reached": int(timesteps[-1]),
    }


def _parse_run_name(run_name: str) -> Dict[str, Any]:
    """Parse 'lr=2e-04_ent=0.01_steps=512_gamma=0.99' into a dict."""
    params: dict = {}
    for part in run_name.split("_"):
        m = re.fullmatch(r"([^=]+)=(.+)", part)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    return params


def compare_algo(algo: str, top_n: Optional[int] = None) -> Union[pd.DataFrame, None]:
    """DataFrame of grid-sweep results for one algo, sorted by best reward."""
    algo_dir = LOGS_DIR / algo
    if not algo_dir.exists():
        print(f"[{algo.upper()}] No log directory at {algo_dir}")
        return None

    rows = []
    for run_dir in sorted(algo_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        stats = _load_eval(run_dir)
        if stats is None:
            continue
        row = {"run_name": run_dir.name, **_parse_run_name(run_dir.name), **stats}
        rows.append(row)

    if not rows:
        print(f"[{algo.upper()}] No completed runs found (evaluations.npz missing)")
        return None

    df = (pd.DataFrame(rows)
            .sort_values("best_mean_reward", ascending=False)
            .reset_index(drop=True))
    df.index += 1
    return df.head(top_n) if top_n else df


# ── Bayesian: result loading ──────────────────────────────────────────────────

def compare_algo_bayesian(
    algo: str,
    horizon_eta: float,
    prob_fn: str,
    optuna_dir: Optional[Path] = None,
    top_n: Optional[int] = None,
) -> Union[pd.DataFrame, None]:
    """DataFrame of Bayesian-sweep results read from an Optuna study."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna is required for --search bayesian.  pip install optuna")
        return None

    # Inline the naming helpers to avoid a circular import on the cluster
    _sname = f"{algo}_eta{int(horizon_eta)}_probfn{prob_fn}"
    _scratch = Path("/scratch/epatard")

    if optuna_dir is not None:
        storage_path = optuna_dir / f"{_sname}.db"
    else:
        storage_path = _scratch / "optuna" / f"{_sname}.db"

    if not storage_path.exists():
        print(f"[{algo.upper()}] Optuna study not found at {storage_path}")
        return None

    sname = _sname
    try:
        study = optuna.load_study(study_name=sname, storage=f"sqlite:///{storage_path}")
    except Exception as exc:
        print(f"[{algo.upper()}] Could not load Optuna study: {exc}")
        return None

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print(f"[{algo.upper()}] No completed Bayesian trials in study '{sname}'")
        return None

    rows = []
    for t in trials:
        row: dict = {"trial": t.number}
        row.update(t.params)
        row["best_mean_reward"] = t.value

        # Supplement with full eval stats if the run directory exists
        run_name = f"t{t.number:04d}_" + "_".join(
            f"{k}={v}" for k, v in t.params.items()
        )
        log_dir  = _scratch / "logs" / algo / run_name
        stats    = _load_eval(log_dir)
        if stats:
            row["final_mean_reward"] = stats["final_mean_reward"]
            row["n_evals"]           = stats["n_evals"]
            row["timesteps_reached"] = stats["timesteps_reached"]

        rows.append(row)

    df = (pd.DataFrame(rows)
            .sort_values("best_mean_reward", ascending=False)
            .reset_index(drop=True))
    df.index += 1
    return df.head(top_n) if top_n else df


# ── Display ───────────────────────────────────────────────────────────────────

def _print_table(algo: str, df: pd.DataFrame, mode: str = "grid") -> None:
    print(f"\n{'═' * 70}")
    print(f"  {algo.upper()}  —  {len(df)} runs ranked by best mean eval reward  [{mode}]")
    print(f"{'═' * 70}")

    meta_cols  = {"run_name", "trial", "best_mean_reward", "final_mean_reward",
                  "n_evals", "timesteps_reached"}
    param_cols = [c for c in df.columns if c not in meta_cols]
    keep       = param_cols + [c for c in
                                ["best_mean_reward", "final_mean_reward", "timesteps_reached"]
                                if c in df.columns]
    display    = df[keep]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(display.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

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
        help="Which sweep to read results from (default: grid)",
    )
    # Bayesian-only args (used to locate the right study file)
    parser.add_argument(
        "--horizon-eta", type=float, default=50.0,
        metavar="ETA",
        help="η used during the Bayesian sweep (bayesian mode only, default: 50)",
    )
    parser.add_argument(
        "--prob-fn", choices=["clip", "sigmoid"], default="clip",
        help="prob_fn used during the Bayesian sweep (bayesian mode only, default: clip)",
    )
    parser.add_argument(
        "--optuna-dir", type=Path, default=None,
        metavar="DIR",
        help="Directory containing Optuna .log files (default: /scratch/epatard/optuna/)",
    )
    # Shared args
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only the top N runs per algo (default: all)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        metavar="FILE",
        help="Save results to CSV — one file per algo, e.g. sweep.csv → sweep_ppo.csv",
    )
    args = parser.parse_args()

    for algo in args.algo:
        if args.search == "grid":
            df = compare_algo(algo, top_n=args.top)
        else:
            df = compare_algo_bayesian(
                algo,
                horizon_eta=args.horizon_eta,
                prob_fn=args.prob_fn,
                optuna_dir=args.optuna_dir,
                top_n=args.top,
            )

        if df is None:
            continue

        _print_table(algo, df, mode=args.search)

        if args.csv:
            stem   = Path(args.csv).stem
            suffix = Path(args.csv).suffix or ".csv"
            out    = Path(args.csv).parent / f"{stem}_{algo}{suffix}"
            df.to_csv(out)
            print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
