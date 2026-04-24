"""
Compare hyperparameter sweep results to identify the best configuration per algorithm.

For each algorithm, scans benchmark/logs/<algo>/<run_name>/evaluations.npz
(written by SB3's EvalCallback) and builds a ranked DataFrame.

Usage:
    python benchmark/compare_sweep.py                  # all algos, print table
    python benchmark/compare_sweep.py --algo ppo       # PPO only
    python benchmark/compare_sweep.py --top 5          # top 5 per algo
    python benchmark/compare_sweep.py --csv sweep.csv  # also save CSV per algo
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional, Optional, Union

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).parent
LOGS_DIR      = BENCHMARK_DIR / "logs"

# ── Result loading ────────────────────────────────────────────────────────────

def _load_eval(log_dir: Path) -> Union[Dict[str, Any], None]:
    """
    Read evaluations.npz produced by EvalCallback.

    Returns a dict with summary statistics, or None if the file is missing
    (job still running or crashed).
    """
    npz = log_dir / "evaluations.npz"
    if not npz.exists():
        return None
    data         = np.load(npz)
    results      = data["results"]          # (n_evals, n_eval_episodes)
    timesteps    = data["timesteps"]        # (n_evals,)
    mean_rewards = results.mean(axis=1)
    return {
        "best_mean_reward":  float(mean_rewards.max()),
        "final_mean_reward": float(mean_rewards[-1]),
        "n_evals":           int(len(timesteps)),
        "timesteps_reached": int(timesteps[-1]),
    }


def _parse_run_name(run_name: str) -> Dict[str, Any]:
    """
    Parse 'lr=2e-04_ent=0.01_steps=512_gamma=0.99' into a dict.
    Values are cast to float or int when possible.
    """
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
    """
    Build a DataFrame of sweep results for one algorithm, sorted by best reward.

    Returns None if no completed runs are found.
    """
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

    df = pd.DataFrame(rows).sort_values("best_mean_reward", ascending=False).reset_index(drop=True)
    df.index += 1  # rank starts at 1
    if top_n:
        df = df.head(top_n)
    return df


def _print_table(algo: str, df: pd.DataFrame) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {algo.upper()}  —  {len(df)} runs ranked by best mean eval reward")
    print(f"{'═' * 70}")

    # Separate hyperparameter columns from metadata columns
    meta_cols  = {"run_name", "best_mean_reward", "final_mean_reward",
                  "n_evals", "timesteps_reached"}
    param_cols = [c for c in df.columns if c not in meta_cols]
    display    = df[param_cols + ["best_mean_reward", "final_mean_reward", "timesteps_reached"]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(display.to_string())

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
        "--top", type=int, default=None,
        help="Show only top N runs per algo (default: all)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        metavar="FILE",
        help="Save results to CSV — one file per algo, e.g. sweep_ppo.csv",
    )
    args = parser.parse_args()

    for algo in args.algo:
        df = compare_algo(algo, top_n=args.top)
        if df is None:
            continue
        _print_table(algo, df)
        if args.csv:
            stem = Path(args.csv).stem
            suffix = Path(args.csv).suffix or ".csv"
            out = Path(args.csv).parent / f"{stem}_{algo}{suffix}"
            df.to_csv(out)
            print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
