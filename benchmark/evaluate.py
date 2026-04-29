"""
Evaluation and benchmarking for RL and planning policies (sequential).

Parameters are loaded from benchmark/conf/eval.yaml (override with --config).
RL model paths can be set in the YAML (rl_models section) or passed as CLI flags.

Quick start
-----------
    from benchmark.evaluate import run_full_benchmark, EvalConfig
    df = run_full_benchmark()
    from benchmark.evaluate import score_matrix
    scores = score_matrix(df)

CLI usage
---------
    # run with all defaults from eval.yaml
    python -m benchmark.evaluate --horizon-eta 50

    # with RL models (override eval.yaml paths)
    python -m benchmark.evaluate --horizon-eta 50 \\
        --ppo-model /path/to/ppo/best_model

    # select specific simulation policies
    python -m benchmark.evaluate --horizon-eta 50 --sim-policies greedy shading

    # planning baselines only (no RL, no simulation)
    python -m benchmark.evaluate --horizon-eta 50 --sim-policies none --no-rl

Available baselines: fipla, constant, random

MPC requires a pre-trained MLPValuationHead — run train_mlp.py first:
    python -m benchmark.train_mlp

For parallel execution on a multi-core machine or cluster, use
benchmark.full_benchmark.parallel_evaluate.
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
import yaml
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

from benchmark.train_rl import SCENARIO_ID, load_model
from benchmark.train_mlp import load_mlp
from dynamicauc.wrappers import SB3PolicyWrapper

RESULTS_DIR   = Path(__file__).parent / "benchmark_results"
_DEFAULT_CONF = Path(__file__).parent / "conf" / "eval.yaml"

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
    seeds:                List[int]              = field(default_factory=lambda: list(range(30)))
    scenario_id:          str                    = SCENARIO_ID
    scenario_overrides:   dict                   = field(default_factory=dict)
    max_bid:              float                  = 5.0
    baselines:            List[str]              = field(default_factory=lambda: list(AVAILABLE_BASELINES))
    constant_bid:         float                  = 0.5
    random_max_bid:       float                  = 1.0
    mask_expected_reward: bool                   = True
    # RL models: maps algo name → path without .zip (None to skip)
    rl_model_paths:       Dict[str, Optional[str]] = field(default_factory=dict)
    # Simulation-based planning policies
    policy_seed:          int                    = 0
    include_greedy:       bool                   = False
    include_shading:      bool                   = False
    include_mpc:          bool                   = False
    include_mpc_oracle:   bool                   = False
    include_mpc_timebin:  bool                   = False
    mpc_n_bins:           List[int]              = field(default_factory=lambda: [2])
    simulation_n_rollouts: int                   = 10
    shading_alphas:       List[float]            = field(default_factory=lambda: [0.8])
    mpc_model_paths:      List[Optional[str]]    = field(default_factory=lambda: [None])
    mpc_alpha_candidates: List[float]            = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5])
    mpc_planning_horizons: List[float]           = field(default_factory=lambda: [7.0])
    # Used by parallel_evaluate.benchmark(); ignored here.
    n_workers:            Optional[int]          = None


def load_eval_conf(cfg_path: Optional[Path] = None) -> dict:
    """Load the eval YAML, defaulting to benchmark/conf/eval.yaml."""
    path = Path(cfg_path) if cfg_path else _DEFAULT_CONF
    with open(path) as f:
        return yaml.safe_load(f)


def cfg_from_yaml(conf: dict) -> EvalConfig:
    """Build an EvalConfig from a parsed eval YAML dict."""
    eval_conf    = conf.get("eval", {})
    shading_conf = conf.get("shading", {})
    mpc_conf     = conf.get("mpc", {})
    policy_conf  = conf.get("policies", {})
    rl_conf      = conf.get("rl_models", {}) or {}

    rl_model_paths = {k: v for k, v in rl_conf.items() if v is not None}
    mlp_model = mpc_conf.get("mlp_model")   # None → auto-resolved by load_mlp

    return EvalConfig(
        seeds=list(range(eval_conf.get("n_seeds", 30))),
        scenario_id=conf.get("scenario_id", SCENARIO_ID),
        policy_seed=eval_conf.get("policy_seed", 0),
        baselines=eval_conf.get("baselines", ["fipla"]),
        shading_alphas=shading_conf.get("alphas", [0.2, 0.4, 0.6, 0.8]),
        mpc_planning_horizons=mpc_conf.get("horizons", [0.0, 4.0, 7.0, 10.0, 15.0, 25.0]),
        mpc_alpha_candidates=mpc_conf.get("alpha_candidates", [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]),
        mpc_n_bins=mpc_conf.get("n_bins", [2, 3, 5, 10]),
        mpc_model_paths=[mlp_model],
        include_greedy=policy_conf.get("include_greedy", False),
        include_shading=policy_conf.get("include_shading", False),
        include_mpc=policy_conf.get("include_mpc", False),
        include_mpc_oracle=policy_conf.get("include_mpc_oracle", False),
        include_mpc_timebin=policy_conf.get("include_mpc_timebin", False),
        rl_model_paths=rl_model_paths,
    )


# ── core evaluation ───────────────────────────────────────────────────────────

def _run_episode(policy: BiddingPolicy, env: gym.Env, seed: int, policy_name: str) -> pd.DataFrame:
    obs, _info = env.reset(seed=seed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    cum_reward = 0.0
    step       = 0
    done       = False
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
            "competing_bid":   float(info["auction"]["competing_bid"]),
        })

    return pd.DataFrame(rows)


def evaluate_policy(policy: BiddingPolicy, scenario, cfg: EvalConfig, policy_name: str) -> pd.DataFrame:
    env    = gym.make("DynamicAuction-v0", scenario=scenario)
    frames = []
    for s in tqdm(cfg.seeds, desc=f"Evaluating {policy_name}", unit="seed"):
        frames.append(_run_episode(policy, env, s, policy_name))
    env.close()
    return pd.concat(frames, ignore_index=True)


# ── policy factories ──────────────────────────────────────────────────────────

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
            policies[f"Shading(α={alpha:g})"] = ShadingPolicy(
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
                policies[_mpc_label(path, horizon, n_models, n_horizons)] = MPCPolicy(
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
                policies[_mpc_label(path, horizon, n_models, n_horizons, prefix="MPCOracle")] = MPCOraclePolicy(
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
    model_paths: Dict[str, Optional[str]],
    mask_expected_reward: bool = True,
) -> Dict[str, SB3PolicyWrapper]:
    """Load RL policies from explicit model paths.

    Args:
        model_paths: maps algo name (e.g. "ppo") to path without .zip extension.
            obs_rms.pkl is expected in the same directory. Pass None to skip.
    """
    if not model_paths:
        return {}

    raw_env   = gym.make("DynamicAuction-v0", scenario=scenario)
    obs_space = raw_env.observation_space
    raw_env.close()

    exclude   = ["expected_reward"] if mask_expected_reward else []
    policies: Dict[str, SB3PolicyWrapper] = {}

    for algo, path_str in model_paths.items():
        if path_str is None:
            continue
        model_path = Path(path_str)
        zip_path   = Path(str(model_path) + ".zip")
        if not zip_path.exists():
            print(f"  [{algo.upper()}] model not found at {zip_path} — skipping.")
            continue
        obs_rms      = None
        obs_rms_path = model_path.parent / "obs_rms.pkl"
        if obs_rms_path.exists():
            with open(obs_rms_path, "rb") as f:
                obs_rms = pickle.load(f)
        model = load_model(algo, path=model_path)
        policies[algo.upper()] = SB3PolicyWrapper(
            model, obs_space, obs_rms=obs_rms, exclude_keys=exclude
        )
        print(f"  Loaded {algo.upper()} ← {zip_path}")

    return policies


# ── benchmark runners ─────────────────────────────────────────────────────────

def benchmark(policies: Dict[str, BiddingPolicy], cfg: EvalConfig = EvalConfig()) -> pd.DataFrame:
    if not policies:
        raise ValueError("No policies to benchmark.")
    scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)
    frames: List[pd.DataFrame] = []
    for name, policy in policies.items():
        print(f"  Evaluating {name} ({len(cfg.seeds)} seeds) …")
        frames.append(evaluate_policy(policy, scenario, cfg, name))
    return pd.concat(frames, ignore_index=True)


def run_full_benchmark(
    cfg: EvalConfig = EvalConfig(),
    save: bool = True,
    extra_policies: Optional[Dict[str, BiddingPolicy]] = None,
) -> pd.DataFrame:
    """Evaluate all policies defined in *cfg* and return a tidy DataFrame.

    Example
    -------
    >>> from benchmark.evaluate import run_full_benchmark, EvalConfig
    >>> from dynamicauc.baseline.policies import BiddingPolicy
    >>>
    >>> class MyPolicy(BiddingPolicy):
    ...     def bid(self, obs): return obs["expected_reward"].item() * 0.8
    >>>
    >>> df = run_full_benchmark(
    ...     cfg=EvalConfig(seeds=list(range(30))),
    ...     extra_policies={"MyPolicy": MyPolicy()},
    ... )
    """
    scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)

    policies: Dict[str, BiddingPolicy] = {}
    policies.update(get_rl_policies(scenario, cfg.rl_model_paths, cfg.mask_expected_reward))
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


# ── analysis helpers ──────────────────────────────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full benchmark (sequential).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Available baselines: {', '.join(AVAILABLE_BASELINES)}\n"
            "Simulation policies: greedy, shading, mpc, mpc-oracle, mpc-timebin, all, none\n"
            "\nExamples:\n"
            "  python -m benchmark.evaluate --horizon-eta 50\n"
            "  python -m benchmark.evaluate --horizon-eta 50 --ppo-model /path/to/best_model\n"
            "  python -m benchmark.evaluate --horizon-eta 50 --sim-policies greedy shading\n"
            "  python -m benchmark.evaluate --horizon-eta 50 --sim-policies none --no-rl\n"
        ),
    )
    parser.add_argument("--config",      type=Path, default=None,
                        help="Path to eval YAML config (default: benchmark/conf/eval.yaml).")
    parser.add_argument("--horizon-eta", type=float, default=None,
                        help="Exponential horizon parameter η (sets scenario_overrides.horizon_eta).")
    parser.add_argument("--n-seeds",     type=int,   default=None,
                        help="Number of evaluation seeds (overrides eval.yaml).")
    parser.add_argument("--policy-seed", type=int,   default=None,
                        help="Seed for simulation-based policy components (overrides eval.yaml).")
    parser.add_argument("--baselines",   nargs="+",  default=None, metavar="BASELINE",
                        help="Baselines to include (overrides eval.yaml). Pass 'none' to skip all.")
    parser.add_argument(
        "--sim-policies", nargs="*", default=None,
        choices=["greedy", "shading", "mpc", "mpc-oracle", "mpc-timebin", "all", "none"],
        metavar="POLICY",
        help="Simulation policies to enable (overrides eval.yaml include_* flags).",
    )
    parser.add_argument("--ppo-model",   type=Path, default=None,
                        help="Path to PPO best_model (without .zip). Overrides eval.yaml.")
    parser.add_argument("--sac-model",   type=Path, default=None,
                        help="Path to SAC best_model (without .zip). Overrides eval.yaml.")
    parser.add_argument("--td3-model",   type=Path, default=None,
                        help="Path to TD3 best_model (without .zip). Overrides eval.yaml.")
    parser.add_argument("--no-rl",       action="store_true",
                        help="Skip all RL policies.")
    parser.add_argument("--mlp-model",   type=Path, default=None,
                        help="Path to the MLPValuationHead .pt file (overrides eval.yaml mpc.mlp_model).")
    parser.add_argument("--no-save",     action="store_true")
    parser.add_argument("--no-mask-expected-reward", action="store_true")
    args = parser.parse_args()

    # 1. load YAML defaults
    conf = load_eval_conf(args.config)
    cfg  = cfg_from_yaml(conf)

    # 2. apply CLI overrides
    if args.horizon_eta is not None:
        cfg.scenario_overrides["horizon_eta"] = args.horizon_eta
    if args.n_seeds is not None:
        cfg.seeds = list(range(args.n_seeds))
    if args.policy_seed is not None:
        cfg.policy_seed = args.policy_seed
    if args.baselines is not None:
        cfg.baselines = [] if args.baselines == ["none"] else args.baselines
    if args.no_mask_expected_reward:
        cfg.mask_expected_reward = False
    if args.mlp_model is not None:
        cfg.mpc_model_paths = [str(args.mlp_model)]

    # 2a. simulation policy overrides
    if args.sim_policies is not None:
        all_on  = "all"  in args.sim_policies
        all_off = "none" in args.sim_policies
        def _flag(name: str) -> bool:
            return (not all_off) and (all_on or name in args.sim_policies)
        cfg.include_greedy      = _flag("greedy")
        cfg.include_shading     = _flag("shading")
        cfg.include_mpc         = _flag("mpc")
        cfg.include_mpc_oracle  = _flag("mpc-oracle")
        cfg.include_mpc_timebin = _flag("mpc-timebin")

    # 2b. RL model overrides
    if args.no_rl:
        cfg.rl_model_paths = {}
    else:
        for algo, path in [("ppo", args.ppo_model), ("sac", args.sac_model), ("td3", args.td3_model)]:
            if path is not None:
                cfg.rl_model_paths[algo] = str(path)

    run_full_benchmark(cfg=cfg, save=not args.no_save)


if __name__ == "__main__":
    main()
