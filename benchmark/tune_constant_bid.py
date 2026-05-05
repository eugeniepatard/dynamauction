"""
Grid search over the constant bid value for ConstantPolicy.

Evaluates Constant(b) for a grid of bid values in [bid-min, bid-max],
reports average episodic reward per bid, and identifies the best one.
Results are saved to a CSV (one row per policy × seed × step).

Usage (from repo root)
----------------------
    python benchmark/tune_constant_bid.py --horizon-eta 50
    python benchmark/tune_constant_bid.py --horizon-eta 50 --n-bids 20 --n-seeds 30
    python benchmark/tune_constant_bid.py --horizon-eta 50 --bid-min 0.0 --bid-max 5.0
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dynamauction as dac
from dynamauction.baseline.policies import ConstantPolicy
from benchmark.evaluate import EvalConfig, evaluate_policy

SCENARIO_ID = "survival-second-price-exponential"
_RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def tune_constant_bid(
    horizon_eta: int,
    bids: list[float],
    n_seeds: int = 30,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    out = Path(out_dir) if out_dir else _RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    scenario = dac.make(SCENARIO_ID, horizon_eta=horizon_eta)
    cfg = EvalConfig(
        seeds=list(range(n_seeds)),
        scenario_id=SCENARIO_ID,
        scenario_overrides={"horizon_eta": horizon_eta},
    )

    frames = []
    for bid_value in bids:
        policy = ConstantPolicy(bid_value)
        label = f"Constant(b={bid_value:.4g})"
        df = evaluate_policy(policy, scenario, cfg, label)
        df["bid_value"] = bid_value
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv = out / f"{SCENARIO_ID}_eta{horizon_eta}_constant_bid_sweep_{ts}.csv"
    results.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")

    # Summary: mean episodic reward per bid value
    summary = (
        results.sort_values(["bid_value", "seed", "step"])
        .groupby(["bid_value", "seed"])["cum_reward"].last()
        .groupby("bid_value").mean()
        .reset_index()
        .rename(columns={"cum_reward": "mean_episode_reward"})
    )
    best_row = summary.loc[summary["mean_episode_reward"].idxmax()]
    print("\nConstant bid sweep summary (mean episodic reward):")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print(
        f"\nBest bid = {best_row['bid_value']:.4g}  "
        f"(mean episode reward = {best_row['mean_episode_reward']:.4f})"
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--horizon-eta", type=int,   required=True,
                        help="Exponential horizon parameter η.")
    parser.add_argument("--n-bids",      type=int,   default=10,
                        help="Number of bid values evenly spaced in [bid-min, bid-max] (default: 10).")
    parser.add_argument("--bid-min",     type=float, default=0.1)
    parser.add_argument("--bid-max",     type=float, default=2.0)
    parser.add_argument("--n-seeds",     type=int,   default=30)
    parser.add_argument("--out-dir",     type=Path,  default=None,
                        help="Directory for CSV output (default: benchmark/benchmark_results/).")
    args = parser.parse_args()

    bids = list(np.linspace(args.bid_min, args.bid_max, args.n_bids))
    tune_constant_bid(
        horizon_eta=args.horizon_eta,
        bids=bids,
        n_seeds=args.n_seeds,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()