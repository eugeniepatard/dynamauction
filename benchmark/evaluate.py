"""
Modular evaluation and benchmarking for RL and planning policies.

Design
------
* All policies satisfy the ``BiddingPolicy`` interface: ``bid(obs_dict) -> float``.
* ``SB3PolicyWrapper`` adapts any trained SB3 model to that interface by
  replicating the ``FlatAuctionObsWrapper`` flattening internally, so the same
  raw-Dict evaluation loop works for both RL and planning agents.
* ``benchmark()`` accepts any ``{name: BiddingPolicy}`` dict — add a new policy
  in one line.

Quick start
-----------
    from benchmark.evaluate import run_full_benchmark, EvalConfig
    df = run_full_benchmark()                          # uses best saved models
    from benchmark.evaluate import score_matrix
    scores = score_matrix(df)                          # {policy: (n_seeds,) array}
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import dynamicauc as dac
from dynamicauc.baseline.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy

from benchmark.train_rl import (
    SCENARIO_ID,
    MODELS_DIR,
    load_model,
    _ALGO_CLS,
)
from dynamicauc.wrappers import SB3PolicyWrapper

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class EvalConfig:
    seeds:             List[int] = field(default_factory=lambda: list(range(30)))
    scenario_id:       str       = SCENARIO_ID
    scenario_overrides: dict     = field(default_factory=dict)


def _run_episode(
    policy:      BiddingPolicy,
    env:         gym.Env,
    seed:        int,
    policy_name: str,
) -> pd.DataFrame:
    obs, _info = env.reset(seed=seed)
    cum_reward  = 0.0
    step        = 0
    done        = False
    rows: List[dict] = []

    while not done:
        bid    = policy.bid(obs)
        action = np.array([bid], dtype=np.float32)
        obs, reward, done, _truncated, info = env.step(action)
        cum_reward += float(reward)
        step       += 1
        rows.append({
            "policy":          policy_name,
            "seed":            seed,
            "step":            obs["step"],
            "t":               float(obs["t"].ravel()[0]),
            "reward":          float(reward),
            "cum_reward":      cum_reward,
            "won":             bool(info["auction"]["won"]),
            "bid":             float(info["auction"]["bid"]),
            "payment":         float(info["auction"]["payment"]),
            "ctx":             float(obs["ctx"].ravel()[0]),
            "expected_reward": float(obs["expected_reward"].ravel()[0]),
            "competing_bid" : float(info["auction"]["competing_bid"]),
        })

    return pd.DataFrame(rows)


def evaluate_policy(
    policy:      BiddingPolicy,
    scenario,
    cfg:         EvalConfig,
    policy_name: str,
) -> pd.DataFrame:
    """
    Evaluate *one* policy over all seeds defined in ``cfg``.

    Returns a tidy per-step DataFrame with columns:
    policy, seed, step, t, reward, cum_reward, won, bid, payment,
    ctx, expected_reward.
    """
    env    = gym.make("DynamicAuction-v0", scenario=scenario)
    frames = [_run_episode(policy, env, s, policy_name) for s in cfg.seeds]
    env.close()
    return pd.concat(frames, ignore_index=True)


def get_planning_policies() -> Dict[str, BiddingPolicy]:
    """
    Return the oracle / hand-crafted planning baselines.

    Add new baselines here to include them automatically in ``run_full_benchmark``.
    """
    return {
        "FiPla":         FiPlaPolicy(),
        "Constant(0.5)": ConstantPolicy(0.5),
        "Random":        RandomPolicy(2.0),
    }


def get_rl_policies(
    scenario,
    algos: List[str] = ("ppo", "sac", "td3"),
) -> Dict[str, SB3PolicyWrapper]:
    """
    Load saved SB3 models and wrap them as ``BiddingPolicy`` objects.

    Missing model files are skipped with a warning so that a partially-trained
    experiment still runs.
    """
    raw_env   = gym.make("DynamicAuction-v0", scenario=scenario)
    obs_space = raw_env.observation_space
    raw_env.close()

    policies: Dict[str, SB3PolicyWrapper] = {}
    for algo in algos:
        model_path = MODELS_DIR / algo / "best_model"
        if not Path(str(model_path) + ".zip").exists():
            print(f"  [WARNING] No saved model for {algo.upper()} at {model_path}.zip — skipping.")
            continue
        obs_rms = None
        obs_rms_path = MODELS_DIR / algo / "obs_rms.pkl"
        if obs_rms_path.exists():
            with open(obs_rms_path, "rb") as f:
                obs_rms = pickle.load(f)
        model = load_model(algo)
        policies[algo.upper()] = SB3PolicyWrapper(model, obs_space, obs_rms=obs_rms)
        print(f"  Loaded {algo.upper()} ← {model_path}.zip")

    return policies


def benchmark(
    policies: Dict[str, BiddingPolicy],
    cfg:      EvalConfig = EvalConfig(),
) -> pd.DataFrame:
    """
    Evaluate a ``{name: BiddingPolicy}`` mapping on the scenario defined by ``cfg``.

    This is the main composition point: pass any mix of RL and planning policies.

    Returns
    -------
    pd.DataFrame
        Tidy per-step frame for all policies and seeds.
    """
    scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)
    frames: List[pd.DataFrame] = []
    for name, policy in policies.items():
        print(f"  Evaluating {name} ({len(cfg.seeds)} seeds) …")
        frames.append(evaluate_policy(policy, scenario, cfg, name))
    return pd.concat(frames, ignore_index=True)


def run_full_benchmark(
    algos:  List[str]       = ("ppo", "sac", "td3"),
    cfg:    EvalConfig      = EvalConfig(),
    save:   bool            = True,
    extra_policies: Optional[Dict[str, BiddingPolicy]] = None,
) -> pd.DataFrame:
    """
    End-to-end benchmark: load all trained RL models + planning baselines,
    evaluate everything, and optionally save results to CSV.

    Parameters
    ----------
    algos : list of str
        RL algorithms to load (must have been trained first via ``train_rl.py``).
    cfg : EvalConfig
        Seeds and scenario config.
    save : bool
        If True, write ``benchmark/results/benchmark_results.csv``.
    extra_policies : dict, optional
        Additional ``{name: BiddingPolicy}`` entries to include (e.g. a new baseline).

    Returns
    -------
    pd.DataFrame
        Per-step results for all policies.
    """
    scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)

    policies: Dict[str, BiddingPolicy] = {}
    policies.update(get_rl_policies(scenario, list(algos)))
    policies.update(get_planning_policies())
    if extra_policies:
        policies.update(extra_policies)

    print(f"\nBenchmarking {len(policies)} policies: {list(policies.keys())}")
    df = benchmark(policies, cfg)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "benchmark_results.csv"
        df.to_csv(out, index=False)
        print(f"\nResults saved → {out}")

    return df


# ── Aggregation helpers ────────────────────────────────────────────────────────

def episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the per-step DataFrame to one row per (policy, seed).

    Columns: policy, seed, episode_reward, n_steps, win_rate, mean_bid, mean_payment.
    """
    return (
        df.groupby(["policy", "seed"], sort=False)
        .agg(
            episode_reward=("reward",  "sum"),
            n_steps       =("step",    "max"),
            win_rate      =("won",     "mean"),
            mean_bid      =("bid",     "mean"),
            mean_payment  =("payment", "mean"),
        )
        .reset_index()
    )


def score_matrix(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build the score matrix expected by **rliable**.

    Returns ``{policy_name: np.ndarray of shape (n_seeds,)}`` where each value
    is the total episode reward for one seed.
    """
    summary = episode_summary(df)
    return {
        policy: summary.loc[summary["policy"] == policy, "episode_reward"].to_numpy()
        for policy in summary["policy"].unique()
    }


# ── CLI entry-point ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the full benchmark.")
    parser.add_argument("--algos",   nargs="+", default=["ppo", "sac", "td3"])
    parser.add_argument("--n-seeds", type=int,  default=30)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = EvalConfig(seeds=list(range(args.n_seeds)))
    run_full_benchmark(algos=args.algos, cfg=cfg, save=not args.no_save)


if __name__ == "__main__":
    main()
