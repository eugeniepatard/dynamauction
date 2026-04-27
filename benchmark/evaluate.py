"""
Evaluation and benchmarking for RL and planning policies.

Quick start
-----------
    from benchmark.evaluate import run_full_benchmark, EvalConfig
    df = run_full_benchmark()
    from benchmark.evaluate import score_matrix
    scores = score_matrix(df)

Selecting baselines (CLI)
-------------------------
    python -m benchmark.evaluate --baselines fipla constant
    python -m benchmark.evaluate --baselines none   # RL only

Available baselines: fipla, constant, random

MPC requires a pre-trained MLPValuationHead — run train_mlp.py first:
    python -m benchmark.train_mlp
    python -m benchmark.evaluate --sim-policies mpc
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
from dynamicauc.baseline import (
    SimulationEstimator,
    GreedyMyopicPolicy,
    ShadingPolicy,
    MPCPolicy,
)

from benchmark.train_rl import (
    SCENARIO_ID,
    MODELS_DIR,
    load_model,
    _ALGO_CLS,
)
from benchmark.train_mlp import load_mlp
from dynamicauc.wrappers import SB3PolicyWrapper

RESULTS_DIR  = Path(__file__).parent / "results"

AVAILABLE_BASELINES: List[str] = ["fipla", "constant", "random"]


def _make_baseline(name: str, constant_bid: float, random_max_bid: float) -> tuple[str, BiddingPolicy]:
    key = name.lower()
    if key == "fipla":
        return "FiPla", FiPlaPolicy()
    if key == "constant":
        return f"Constant({constant_bid})", ConstantPolicy(constant_bid)
    if key == "random":
        return "Random", RandomPolicy(random_max_bid)
    raise ValueError(
        f"Unknown baseline {name!r}. Available: {AVAILABLE_BASELINES}. "
        "Pass a custom BiddingPolicy via extra_policies instead."
    )


@dataclass
class EvalConfig:
    seeds:             List[int] = field(default_factory=lambda: list(range(30)))
    scenario_id:       str       = SCENARIO_ID
    scenario_overrides: dict     = field(default_factory=dict)
    max_bid:           float     = 5.0
    baselines:            List[str] = field(default_factory=lambda: list(AVAILABLE_BASELINES))
    constant_bid:         float     = 0.5
    random_max_bid:       float     = 1.0
    mask_expected_reward: bool      = True
    # Simulation-based planning policies
    include_greedy:        bool           = False
    include_shading:       bool           = False
    include_mpc:           bool           = False
    simulation_n_rollouts: int            = 10
    shading_alpha:         float          = 0.8
    mpc_model_path:        Optional[str]  = None
    mpc_alpha_candidates:  List[float]    = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5])
    mpc_planning_horizon:  float          = 7.0


def _run_episode(
    policy:      BiddingPolicy,
    env:         gym.Env,
    seed:        int,
    policy_name: str,
) -> pd.DataFrame:
    obs, _info = env.reset(seed=seed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
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
            "competing_bid" :  float(info["auction"]["competing_bid"]),
        })

    return pd.DataFrame(rows)


def evaluate_policy(
    policy:      BiddingPolicy,
    scenario,
    cfg:         EvalConfig,
    policy_name: str,
) -> pd.DataFrame:
    env    = gym.make("DynamicAuction-v0", scenario=scenario)
    frames = [_run_episode(policy, env, s, policy_name) for s in cfg.seeds]
    env.close()
    return pd.concat(frames, ignore_index=True)


def get_planning_policies(cfg: EvalConfig) -> Dict[str, BiddingPolicy]:
    if cfg.baselines == ["none"]:
        return {}
    policies: Dict[str, BiddingPolicy] = {}
    for name in cfg.baselines:
        label, policy = _make_baseline(name, cfg.constant_bid, cfg.random_max_bid)
        policies[label] = policy
    return policies


def get_simulation_policies(cfg: EvalConfig) -> Dict[str, BiddingPolicy]:
    policies: Dict[str, BiddingPolicy] = {}

    if cfg.include_greedy:
        policies["GreedyMyopic"] = GreedyMyopicPolicy(
            SimulationEstimator(
                pi0=ConstantPolicy(0.0),
                n_rollouts=cfg.simulation_n_rollouts,
            )
        )

    if cfg.include_shading:
        policies[f"Shading(α={cfg.shading_alpha})"] = ShadingPolicy(
            SimulationEstimator(
                pi0=ConstantPolicy(0.0),
                n_rollouts=cfg.simulation_n_rollouts,
            ),
            alpha=cfg.shading_alpha,
        )

    if cfg.include_mpc:
        mlp = load_mlp(cfg.scenario_id, cfg.mpc_model_path)
        policies["MPC"] = MPCPolicy(
            v0=mlp,
            alpha_candidates=cfg.mpc_alpha_candidates,
            alpha_cont=1.0,
            n_rollouts=cfg.simulation_n_rollouts,
            planning_horizon=cfg.mpc_planning_horizon,
        )

    return policies


def get_rl_policies(
    scenario,
    algos: List[str] = ("ppo", "sac", "td3"),
    mask_expected_reward: bool = True,
) -> Dict[str, SB3PolicyWrapper]:
    raw_env   = gym.make("DynamicAuction-v0", scenario=scenario)
    obs_space = raw_env.observation_space
    raw_env.close()

    exclude = ["expected_reward"] if mask_expected_reward else []

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
        policies[algo.upper()] = SB3PolicyWrapper(
            model, obs_space, obs_rms=obs_rms, exclude_keys=exclude
        )
        print(f"  Loaded {algo.upper()} ← {model_path}.zip")

    return policies


def benchmark(
    policies: Dict[str, BiddingPolicy],
    cfg:      EvalConfig = EvalConfig(),
) -> pd.DataFrame:
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
    scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)

    policies: Dict[str, BiddingPolicy] = {}
    policies.update(get_rl_policies(scenario, list(algos), cfg.mask_expected_reward))
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg))
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
    summary = episode_summary(df)
    return {
        policy: summary.loc[summary["policy"] == policy, "episode_reward"].to_numpy()
        for policy in summary["policy"].unique()
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Available baselines: {', '.join(AVAILABLE_BASELINES)}\n"
            "Pass --baselines none to skip baselines entirely.\n"
            "\nExamples:\n"
            "  python -m benchmark.evaluate\n"
            "  python -m benchmark.evaluate --algos ppo sac --baselines fipla\n"
            "  python -m benchmark.evaluate --baselines none\n"
            "  python -m benchmark.evaluate --sim-policies mpc  # needs train_mlp.py first\n"
        ),
    )
    parser.add_argument("--algos",           nargs="+", default=["ppo", "sac", "td3"])
    parser.add_argument("--baselines",       nargs="+", default=AVAILABLE_BASELINES, metavar="BASELINE")
    parser.add_argument("--constant-bid",    type=float, default=0.5)
    parser.add_argument("--random-max-bid",  type=float, default=1.0)
    parser.add_argument("--n-seeds",         type=int,   default=30)
    parser.add_argument("--no-save",         action="store_true")
    parser.add_argument("--no-mask-expected-reward", action="store_true")
    parser.add_argument(
        "--sim-policies", nargs="*", default=["all"],
        choices=["greedy", "shading", "mpc", "all"],
        metavar="POLICY",
    )
    parser.add_argument("--n-rollouts",  type=int,   default=10)
    parser.add_argument("--alpha",       type=float, default=0.8)
    parser.add_argument("--mpc-model",   type=str,   default=None,
                        help="Path to a MLPValuationHead .pt file (default: auto-resolved from models dir).")
    args = parser.parse_args()

    cfg = EvalConfig(
        seeds=list(range(args.n_seeds)),
        baselines=args.baselines,
        constant_bid=args.constant_bid,
        random_max_bid=args.random_max_bid,
        mask_expected_reward=not args.no_mask_expected_reward,
        include_greedy="greedy" in args.sim_policies or "all" in args.sim_policies,
        include_shading="shading" in args.sim_policies or "all" in args.sim_policies,
        include_mpc="mpc" in args.sim_policies or "all" in args.sim_policies,
        simulation_n_rollouts=args.n_rollouts,
        shading_alpha=args.alpha,
        mpc_model_path=args.mpc_model,
    )

    run_full_benchmark(algos=args.algos, cfg=cfg, save=not args.no_save)


if __name__ == "__main__":
    main()
