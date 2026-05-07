"""
Grid search over the shading multiplier α for ShadingPolicy.

Evaluates Shading(α) for a grid of α values in [alpha-min, alpha-max],
reports average episodic reward per α, and identifies the best one.
Results are saved to a CSV (one row per policy × seed × step).

Usage (from repo root)
----------------------
    python benchmark/tune_shading_alpha.py --horizon-eta 50
    python benchmark/tune_shading_alpha.py --horizon-eta 50 --n-alphas 20 --n-seeds 30
    python benchmark/tune_shading_alpha.py --horizon-eta 50 --alpha-min 0.0 --alpha-max 1.5
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
from dynamauction.baseline.baseline_algo import SimulationEstimator, ShadingPolicy
from dynamauction.baseline.policies import ConstantPolicy
from benchmark.evaluate import EvalConfig, evaluate_policy

SCENARIO_ID = "survival-second-price-exponential"
_RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def tune_shading_alpha(
    horizon_eta: int,
    alphas: list[float],
    n_seeds: int = 30,
    n_rollouts: int = 10,
    policy_seed: int = 0,
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
    for alpha in alphas:
        policy = ShadingPolicy(
            SimulationEstimator(
                pi0=ConstantPolicy(0.0),
                n_rollouts=n_rollouts,
                seed=policy_seed,
            ),
            alpha=alpha,
        )
        label = f"Shading(α={alpha:.4g})"
        df = evaluate_policy(policy, scenario, cfg, label)
        df["alpha"] = alpha
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv = out / f"{SCENARIO_ID}_eta{horizon_eta}_shading_alpha_sweep_{ts}.csv"
    results.to_csv(csv, index=False)
    print(f"\nResults saved -> {csv}")

    # Summary: mean episodic reward per α
    summary = (
        results.sort_values(["alpha", "seed", "step"])
        .groupby(["alpha", "seed"])["cum_reward"].last()
        .groupby("alpha").mean()
        .reset_index()
        .rename(columns={"cum_reward": "mean_episode_reward"})
    )
    best_row = summary.loc[summary["mean_episode_reward"].idxmax()]
    print("\nα sweep summary (mean episodic reward):")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print(
        f"\nBest α = {best_row['alpha']:.4g}  "
        f"(mean episode reward = {best_row['mean_episode_reward']:.4f})"
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--horizon-eta",  type=int,   required=True,
                        help="Exponential horizon parameter η.")
    parser.add_argument("--n-alphas",     type=int,   default=10,
                        help="Number of α values evenly spaced in [alpha-min, alpha-max] (default: 20).")
    parser.add_argument("--alpha-min",    type=float, default=0.1)
    parser.add_argument("--alpha-max",    type=float, default=1.0)
    parser.add_argument("--n-seeds",      type=int,   default=30)
    parser.add_argument("--n-rollouts",   type=int,   default=10,
                        help="MC rollouts per decision step inside the estimator.")
    parser.add_argument("--policy-seed",  type=int,   default=0)
    parser.add_argument("--out-dir",      type=Path,  default=None,
                        help="Directory for CSV output (default: benchmark/shading_alpha_results/).")
    args = parser.parse_args()

    alphas = list(np.linspace(args.alpha_min, args.alpha_max, args.n_alphas))
    tune_shading_alpha(
        horizon_eta=args.horizon_eta,
        alphas=alphas,
        n_seeds=args.n_seeds,
        n_rollouts=args.n_rollouts,
        policy_seed=args.policy_seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
