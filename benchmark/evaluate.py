"""
Evaluation and benchmarking for RL and planning policies (sequential).

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

For parallel execution on a multi-core machine or cluster, use
benchmark.full_benchmark.parallel_evaluate instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import gymnasium as gym

import dynamicauc as dac
from dynamicauc.baseline.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy
from dynamicauc.baseline import (
    SimulationEstimator,
    GreedyMyopicPolicy,
    ShadingPolicy,
    MPCOraclePolicy,
    MPCPolicy,
    MPCTimeBinnedPolicy,
)

from benchmark.train_rl import (
    SCENARIO_ID,
    MODELS_DIR,
    load_model,
)
from benchmark.train_mlp import load_mlp
from dynamicauc.wrappers import SB3PolicyWrapper

RESULTS_DIR = Path(__file__).parent / "benchmark_results"

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
    policy_seed:                int            = 0
    include_greedy:              bool           = False
    include_shading:             bool           = False
    include_mpc:                 bool           = False
    include_mpc_oracle:          bool           = False
    include_mpc_timebin:         bool           = False
    mpc_n_bins:                  List[int]      = field(default_factory=lambda: [2])
    simulation_n_rollouts:       int            = 10
    shading_alphas:              List[float]    = field(default_factory=lambda: [0.8])
    mpc_model_paths:             List[Optional[str]] = field(default_factory=lambda: [None])
    mpc_alpha_candidates:        List[float]    = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5])
    mpc_planning_horizons:       List[float]    = field(default_factory=lambda: [7.0])
    # Used by parallel_evaluate.benchmark(); ignored here.
    n_workers:                   Optional[int]  = None


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

        alpha_chosen = float("nan")
        if hasattr(policy, "alpha_log") and policy.alpha_log:
            last = policy.alpha_log[-1]
            # MPCTimeBinned logs tuples (one α per bin); record the executed first-bin α
            alpha_chosen = float(last[0]) if isinstance(last, (tuple, list)) else float(last)

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
            "competing_bid":   float(info["auction"]["competing_bid"]),
            "alpha_chosen":    alpha_chosen,
        })

    return pd.DataFrame(rows)


def evaluate_policy(
    policy:      BiddingPolicy,
    scenario,
    cfg:         EvalConfig,
    policy_name: str,
) -> pd.DataFrame:
    env    = gym.make("DynamicAuction-v0", scenario=scenario)
    frames = []
    for s in tqdm(cfg.seeds, desc=f"Evaluating {policy_name}", unit="seed"):
        frames.append(_run_episode(policy, env, s, policy_name))
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


def _mpc_label(path: Optional[str], horizon: float, n_models: int, n_horizons: int, prefix: str = "MPC") -> str:
    parts = [f"H={horizon:g}"]
    if n_models > 1:
        if path is None:
            parts.append("auto")
        else:
            stem = Path(path).stem
            parts.append(stem.split("_", 1)[-1] if "_" in stem else stem)
    return prefix + (f"({','.join(parts)})" if parts else "")


def get_simulation_policies(cfg: EvalConfig) -> Dict[str, BiddingPolicy]:
    policies: Dict[str, BiddingPolicy] = {}

    if cfg.include_greedy:
        policies["GreedyMyopic"] = GreedyMyopicPolicy(
            SimulationEstimator(
                pi0=ConstantPolicy(0.0),
                n_rollouts=cfg.simulation_n_rollouts,
                seed=cfg.policy_seed,
            )
        )

    if cfg.include_shading:
        for alpha in cfg.shading_alphas:
            label = f"Shading(α={alpha:g})"
            policies[label] = ShadingPolicy(
                SimulationEstimator(
                    pi0=ConstantPolicy(0.0),
                    n_rollouts=cfg.simulation_n_rollouts,
                    seed=cfg.policy_seed,
                ),
                alpha=alpha,
            )

    if cfg.include_mpc:
        n_models   = len(cfg.mpc_model_paths)
        n_horizons = len(cfg.mpc_planning_horizons)
        for path in cfg.mpc_model_paths:
            mlp = load_mlp(cfg.scenario_id, path)
            for horizon in cfg.mpc_planning_horizons:
                label = _mpc_label(path, horizon, n_models, n_horizons)
                policies[label] = MPCPolicy(
                    v0=mlp,
                    alpha_candidates=cfg.mpc_alpha_candidates,
                    alpha_cont=1.0,
                    n_rollouts=cfg.simulation_n_rollouts,
                    planning_horizon=horizon,
                    seed=cfg.policy_seed,
                )

    if cfg.include_mpc_oracle:
        n_models   = len(cfg.mpc_model_paths)
        n_horizons = len(cfg.mpc_planning_horizons)
        for path in cfg.mpc_model_paths:
            mlp = load_mlp(cfg.scenario_id, path)
            for horizon in cfg.mpc_planning_horizons:
                label = _mpc_label(path, horizon, n_models, n_horizons, prefix="MPCOracle")
                policies[label] = MPCOraclePolicy(
                    v0=mlp,
                    alpha_candidates=cfg.mpc_alpha_candidates,
                    alpha_cont=1.0,
                    n_rollouts=cfg.simulation_n_rollouts,
                    planning_horizon=horizon,
                    seed=cfg.policy_seed,
                )

    if cfg.include_mpc_timebin:
        n_models   = len(cfg.mpc_model_paths)
        n_horizons = len(cfg.mpc_planning_horizons)
        for path in cfg.mpc_model_paths:
            mlp = load_mlp(cfg.scenario_id, path)
            for horizon in cfg.mpc_planning_horizons:
                for k in cfg.mpc_n_bins:
                    parts = [f"H={horizon:g}", f"K={k}"]
                    if n_models > 1:
                        stem = Path(path).stem if path else "auto"
                        parts.append(stem.split("_", 1)[-1] if "_" in stem else stem)
                    label = "MPCTimeBinned" + (f"({','.join(parts)})" if parts else "")
                    policies[label] = MPCTimeBinnedPolicy(
                        v0=mlp,
                        alpha_candidates=cfg.mpc_alpha_candidates,
                        n_bins=k,
                        n_rollouts=cfg.simulation_n_rollouts,
                        planning_horizon=horizon,
                        seed=cfg.policy_seed,
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
    if not policies:
        raise ValueError("No policies to benchmark. Check your --algos, --baselines, and --sim-policies flags.")

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
        out = RESULTS_DIR / f"{cfg.scenario_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df.to_csv(out, index=False)
        print(f"\nResults saved → {out}")

    return df


def episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["policy", "seed", "step"])
        .groupby(["policy", "seed"], sort=False)
        .agg(
            episode_reward=("cum_reward", "last"),
            n_steps       =("step",       "max"),
            win_rate      =("won",        "mean"),
            mean_bid      =("bid",        "mean"),
            mean_payment  =("payment",    "mean"),
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
        description="Run the full benchmark (sequential).",
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
    parser.add_argument("--algos",           nargs="+", default=["ppo", "sac", "td3"], metavar="ALGO",
                        help="RL algorithms to evaluate. Use 'none' to skip all.")
    parser.add_argument("--baselines",       nargs="+", default=AVAILABLE_BASELINES, metavar="BASELINE")
    parser.add_argument("--policy-seed",    type=float, default=0.0, help="Seed for any policy components that require randomness (e.g. MPC rollouts).")
    parser.add_argument("--constant-bid",    type=float, default=0.5)
    parser.add_argument("--random-max-bid",  type=float, default=1.0)
    parser.add_argument("--n-seeds",         type=int,   default=30)
    parser.add_argument("--no-save",         action="store_true")
    parser.add_argument("--no-mask-expected-reward", action="store_true")
    parser.add_argument(
        "--sim-policies", nargs="*", default=["all"],
        choices=["greedy", "shading", "mpc", "mpc-oracle", "mpc-timebin", "all", "none"],
        metavar="POLICY",
    )
    parser.add_argument("--n-rollouts",   type=int,   default=10)
    parser.add_argument("--alphas",        type=float, nargs="+", default=[0.8],
                        help="Shading factors α to evaluate (one Shading policy per value).")
    parser.add_argument("--mpc-models",   type=str,   nargs="+", default=[None],
                        help="Paths to MLPValuationHead .pt files (default: auto-resolved).")
    parser.add_argument("--mpc-horizons", type=float, nargs="+", default=[7.0],
                        help="Planning horizons H to evaluate for MPC.")
    args = parser.parse_args()

    cfg = EvalConfig(
        seeds=list(range(args.n_seeds)),
        baselines=args.baselines,
        constant_bid=args.constant_bid,
        random_max_bid=args.random_max_bid,
        policy_seed=args.policy_seed,
        mask_expected_reward=not args.no_mask_expected_reward,
        include_greedy="greedy" in args.sim_policies or "all" in args.sim_policies,
        include_shading="shading" in args.sim_policies or "all" in args.sim_policies,
        include_mpc="mpc" in args.sim_policies or "all" in args.sim_policies,
        include_mpc_oracle="mpc-oracle" in args.sim_policies or "all" in args.sim_policies,
        simulation_n_rollouts=args.n_rollouts,
        include_mpc_timebin="mpc-timebin" in args.sim_policies or "all" in args.sim_policies,
        shading_alphas=args.alphas,
        mpc_model_paths=args.mpc_models,
        mpc_planning_horizons=args.mpc_horizons,
    )

    algos = [] if args.algos == ["none"] else args.algos
    run_full_benchmark(algos=algos, cfg=cfg, save=not args.no_save)


if __name__ == "__main__":
    main()
