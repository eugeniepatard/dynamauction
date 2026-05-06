"""
Evaluation and benchmarking for RL and planning policies (sequential, single process).

Output: per-step DataFrame with columns
    policy, seed, step, t, reward, cum_reward, won, bid, payment,
    ctx, expected_reward, competing_bid, alpha_chosen

Python API
----------
    from benchmark.evaluate import run_full_benchmark, score_matrix, EvalConfig

    df     = run_full_benchmark()           # all policies, default config
    scores = score_matrix(df)              # {policy: np.ndarray} for rliable

    # Custom seeds / scenario
    df = run_full_benchmark(
        algos=["ppo", "sac"],
        cfg=EvalConfig(seeds=list(range(1000)), scenario_overrides={"horizon_eta": 50}),
    )

    # Custom policy
    df = run_full_benchmark(extra_policies={"MyPolicy": MyPolicy()})

    # MPC (requires train_mlp.py first)
    df = run_full_benchmark(algos=[], cfg=EvalConfig(
        include_mpc=True, mpc_planning_horizons=[0.0, 7.0, 15.0],
    ))

CLI (run from repo root)
------------------------
    # Paper experiment: all policies, 1000 seeds, η=50
    python -m benchmark.evaluate --horizon-eta 50 --n-seeds 1000 \
        --sim-policies greedy shading mpc mpc-oracle mpc-timebin

    # Baselines only (no RL, no MPC)
    python -m benchmark.evaluate --algos none --sim-policies none

    # RL only
    python -m benchmark.evaluate --baselines none --sim-policies none --horizon-eta 50

    # Single RL algorithm against FiPla
    python -m benchmark.evaluate --algos ppo --baselines fipla --sim-policies none

    # Shading α sweep
    python -m benchmark.evaluate --algos none --sim-policies shading \
        --alphas 0.2 0.4 0.6 0.8 1.0 1.2

    # MPC with several planning horizons (needs train_mlp.py first)
    python -m benchmark.evaluate --algos none --sim-policies mpc \
        --mpc-horizons 0 4 7 10 15 25

    # All MPC variants together
    python -m benchmark.evaluate --algos none \
        --sim-policies mpc mpc-oracle mpc-timebin --mpc-horizons 7 15

    # Custom MLP weights path
    python -m benchmark.evaluate --sim-policies mpc \
        --mpc-models benchmark/models/mlp_valuation_head/my_model.pt

    # Explicit model paths (override the automatic eta-based resolution)
    python -m benchmark.evaluate --horizon-eta 100 \
        --ppo-model benchmark/models/ppo/eta100/lr=0.0003_.../best_model \
        --sac-model benchmark/models/sac/eta100/lr=5e-05_.../best_model \
        --td3-model benchmark/models/td3/eta100/lr=0.0001_.../best_model

    # Mix: explicit PPO, auto-resolved SAC and TD3
    python -m benchmark.evaluate --horizon-eta 100 \
        --ppo-model benchmark/models/ppo/eta100/lr=0.0003_.../best_model

    # Dry run (no CSV saved)
    python -m benchmark.evaluate --no-save

Results → benchmark/benchmark_results/<scenario_id>_eta<η>_<timestamp>.csv
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

import dynamauction as dac
from dynamauction.baseline.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy
from dynamauction.baseline.baseline_algo import (
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
from dynamauction.wrappers import SB3PolicyWrapper

# ── Paths ─────────────────────────────────────────────────────────────────────
# SCENARIO_ID and MODELS_DIR are imported from train_rl (set via benchmark/conf/train.yaml).
RESULTS_DIR = Path(__file__).parent / "benchmark_results"

# ── Pretrained model paths (shipped with the repo) ────────────────────────────
# Models are organised as benchmark/models/<algo>/eta<η>/<hyperparams>/best_model.zip.
# _find_model_path() globs for best_model.zip inside the eta-specific directory so
# the exact hyperparameter folder name does not need to be hardcoded here.
# MLP valuation head : produced by train_mlp.py. Required for MPC, MPCOracle, MPCTimeBinned.
_DEFAULT_MLP = MODELS_DIR / "mlp_valuation_head" / "survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt"


def _find_model_path(algo: str, eta: Optional[int]) -> Optional[Path]:
    """Return the best_model stem path (without .zip) for *algo* and horizon *eta*.

    Search order:
    1. ``MODELS_DIR/<algo>/eta<eta>/*/best_model.zip``  (shipped or freshly trained)
    2. ``MODELS_DIR/<algo>/eta50/*/best_model.zip``     (fallback to eta=50 weights)
    """
    for search_eta in ([eta, 50] if eta is not None and eta != 50 else [50]):
        search_dir = MODELS_DIR / algo / f"eta{search_eta}"
        if search_dir.exists():
            hits = sorted(search_dir.glob("*/best_model.zip"))
            if hits:
                return hits[0].with_suffix("")   # strip .zip → stem path
    return None

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_N_SEEDS              = 30
DEFAULT_ALGOS                = ["ppo", "sac", "td3"]
DEFAULT_BASELINES            = ["fipla", "constant", "random"]
DEFAULT_MAX_BID              = 5.0
DEFAULT_CONSTANT_BID         = 0.5
DEFAULT_RANDOM_MAX_BID       = 1.0
DEFAULT_N_ROLLOUTS           = 10
DEFAULT_SHADING_ALPHAS       = [0.8]
DEFAULT_MPC_HORIZONS         = [7.0]
DEFAULT_MPC_ALPHA_CANDIDATES = [0.6, 0.8, 1.0, 1.2, 1.5]
DEFAULT_MPC_N_BINS           = [2]
DEFAULT_HORIZON_ETA          = 50

AVAILABLE_BASELINES: List[str] = DEFAULT_BASELINES


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
    seeds:               List[int] = field(default_factory=lambda: list(range(DEFAULT_N_SEEDS)))
    scenario_id:         str       = SCENARIO_ID
    scenario_overrides:  dict      = field(default_factory=dict)
    max_bid:             float     = DEFAULT_MAX_BID
    baselines:           List[str] = field(default_factory=lambda: list(DEFAULT_BASELINES))
    constant_bid:        float     = DEFAULT_CONSTANT_BID
    random_max_bid:      float     = DEFAULT_RANDOM_MAX_BID
    mask_expected_reward: bool     = True
    # Simulation-based planning policies
    policy_seed:             int            = 0
    include_greedy:          bool           = False
    include_shading:         bool           = False
    include_mpc:             bool           = False
    include_mpc_oracle:      bool           = False
    include_mpc_timebin:     bool           = False
    mpc_n_bins:              List[int]      = field(default_factory=lambda: list(DEFAULT_MPC_N_BINS))
    simulation_n_rollouts:   int            = DEFAULT_N_ROLLOUTS
    shading_alphas:          List[float]    = field(default_factory=lambda: list(DEFAULT_SHADING_ALPHAS))
    mpc_model_paths:         List[Optional[str]] = field(default_factory=lambda: [None])
    mpc_alpha_candidates:    List[float]    = field(default_factory=lambda: list(DEFAULT_MPC_ALPHA_CANDIDATES))
    mpc_planning_horizons:   List[float]    = field(default_factory=lambda: list(DEFAULT_MPC_HORIZONS))
    # Used by parallel_evaluate.benchmark(); ignored here.
    n_workers:               Optional[int]  = None


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
    env    = gym.make("DynamAuction-v0", scenario=scenario)
    frames = []
    for s in tqdm(cfg.seeds, desc=f"Evaluating {policy_name}", unit="seed"):
        frames.append(_run_episode(policy, env, s, policy_name))
    env.close()
    return pd.concat(frames, ignore_index=True)


def get_planning_policies(cfg: EvalConfig) -> Dict[str, BiddingPolicy]:
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
    eta: Optional[int] = None,
    model_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, SB3PolicyWrapper]:
    """
    Parameters
    ----------
    model_paths : dict, optional
        Explicit ``{algo: path}`` overrides (without .zip extension).
        Keys not present fall back to the automatic eta-based resolution.
        Example: ``{"ppo": "benchmark/models/ppo/eta100/.../best_model"}``
    """
    raw_env   = gym.make("DynamAuction-v0", scenario=scenario)
    obs_space = raw_env.observation_space
    raw_env.close()

    exclude     = ["expected_reward"] if mask_expected_reward else []
    model_paths = model_paths or {}

    policies: Dict[str, SB3PolicyWrapper] = {}
    for algo in algos:
        if algo in model_paths:
            model_path = Path(model_paths[algo])
            print(f"  Using explicit path for {algo.upper()} ← {model_path}.zip")
        else:
            model_path = _find_model_path(algo, eta)
            if model_path is None:
                print(f"  [WARNING] No saved model for {algo.upper()} (eta={eta}) — skipping.")
                continue
            if eta is not None and f"eta{eta}" not in str(model_path):
                print(f"  [INFO] No eta{eta} model for {algo.upper()}, falling back to eta50 weights.")
        obs_rms = None
        obs_rms_path = Path(str(model_path).replace("best_model", "obs_rms")).with_suffix(".pkl")
        if obs_rms_path.exists():
            with open(obs_rms_path, "rb") as f:
                obs_rms = pickle.load(f)
        model = load_model(algo, path=model_path)
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
    model_paths: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)

    eta = cfg.scenario_overrides.get("horizon_eta")

    policies: Dict[str, BiddingPolicy] = {}
    policies.update(get_rl_policies(
        scenario, list(algos), cfg.mask_expected_reward,
        eta=eta, model_paths=model_paths,
    ))
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg))
    if extra_policies:
        policies.update(extra_policies)

    print(f"\nBenchmarking {len(policies)} policies: {list(policies.keys())}")
    df = benchmark(policies, cfg)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        eta = cfg.scenario_overrides.get("horizon_eta", "default")
        out = RESULTS_DIR / f"{cfg.scenario_id}_eta{eta}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
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
            "  python -m benchmark.evaluate --horizon-eta 50\n"
            "  python -m benchmark.evaluate --algos ppo sac --baselines fipla\n"
            "  python -m benchmark.evaluate --baselines none\n"
            "  python -m benchmark.evaluate --sim-policies mpc  # needs train_mlp.py first\n"
        ),
    )
    # ── Policy selection ──────────────────────────────────────────────────────
    parser.add_argument("--algos",        nargs="+", default=DEFAULT_ALGOS,     metavar="ALGO",
                        help="RL algorithms to evaluate. Use 'none' to skip all.")
    parser.add_argument("--baselines",    nargs="+", default=DEFAULT_BASELINES, metavar="BASELINE",
                        help="Simple baselines to include. Use 'none' to skip all.")
    parser.add_argument("--sim-policies", nargs="+", default=["none"],          metavar="POLICY",
                        choices=["greedy", "shading", "mpc", "mpc-oracle", "mpc-timebin", "all", "none"],
                        help="Simulation-based planning policies. Use 'all' to include all.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--horizon-eta",             type=int,   default=DEFAULT_HORIZON_ETA,
                        help="Exponential horizon rate η.")
    parser.add_argument("--n-seeds",                 type=int,   default=DEFAULT_N_SEEDS)
    parser.add_argument("--no-mask-expected-reward", action="store_true",
                        help="Expose expected_reward to RL policies (disabled by default).")
    parser.add_argument("--no-save",                 action="store_true",
                        help="Skip writing the results CSV.")

    # ── Simple baseline parameters ────────────────────────────────────────────
    parser.add_argument("--constant-bid",   type=float, default=DEFAULT_CONSTANT_BID,
                        help="Fixed bid value for ConstantPolicy.")
    parser.add_argument("--random-max-bid", type=float, default=DEFAULT_RANDOM_MAX_BID,
                        help="Upper bound for RandomPolicy bids.")

    # ── Simulation-based policy parameters ───────────────────────────────────
    parser.add_argument("--policy-seed",  type=int,   default=DEFAULT_N_ROLLOUTS)
    parser.add_argument("--n-rollouts",   type=int,   default=DEFAULT_N_ROLLOUTS,
                        help="Monte Carlo rollouts per decision step.")
    parser.add_argument("--alphas",       type=float, nargs="+", default=DEFAULT_SHADING_ALPHAS,
                        help="Shading factors α (one ShadingPolicy per value).")

    # ── RL model path overrides ───────────────────────────────────────────────
    parser.add_argument("--ppo-model", type=str, default=None, metavar="PATH",
                        help="Explicit path to PPO best_model (without .zip). "
                             "Overrides the automatic eta-based resolution.")
    parser.add_argument("--sac-model", type=str, default=None, metavar="PATH",
                        help="Explicit path to SAC best_model (without .zip).")
    parser.add_argument("--td3-model", type=str, default=None, metavar="PATH",
                        help="Explicit path to TD3 best_model (without .zip).")

    # ── MPC parameters ────────────────────────────────────────────────────────
    parser.add_argument("--mpc-horizons", type=float, nargs="+", default=DEFAULT_MPC_HORIZONS,
                        help="Planning horizons H for MPC variants.")
    parser.add_argument("--mpc-models",   type=str,   nargs="+", default=[_DEFAULT_MLP],
                        help="Paths to MLPValuationHead .pt files.")
    args = parser.parse_args()

    algos     = [] if "none" in args.algos     else args.algos
    baselines = [] if "none" in args.baselines else args.baselines
    sim_pols  = [] if "none" in args.sim_policies else args.sim_policies

    model_paths = {k: v for k, v in
                   [("ppo", args.ppo_model), ("sac", args.sac_model), ("td3", args.td3_model)]
                   if v is not None}

    cfg = EvalConfig(
        seeds=list(range(args.n_seeds)),
        scenario_overrides={"horizon_eta": args.horizon_eta},
        baselines=baselines,
        constant_bid=args.constant_bid,
        random_max_bid=args.random_max_bid,
        policy_seed=args.policy_seed,
        mask_expected_reward=not args.no_mask_expected_reward,
        simulation_n_rollouts=args.n_rollouts,
        shading_alphas=args.alphas,
        mpc_model_paths=args.mpc_models,
        mpc_planning_horizons=args.mpc_horizons,
        include_greedy      ="greedy"      in sim_pols or "all" in sim_pols,
        include_shading     ="shading"     in sim_pols or "all" in sim_pols,
        include_mpc         ="mpc"         in sim_pols or "all" in sim_pols,
        include_mpc_oracle  ="mpc-oracle"  in sim_pols or "all" in sim_pols,
        include_mpc_timebin ="mpc-timebin" in sim_pols or "all" in sim_pols,
    )

    run_full_benchmark(algos=algos, cfg=cfg, save=not args.no_save, model_paths=model_paths or None)


if __name__ == "__main__":
    main()
