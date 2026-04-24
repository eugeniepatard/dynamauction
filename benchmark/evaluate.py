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

Selecting baselines (CLI)
-------------------------
    python -m benchmark.evaluate --baselines fipla constant
    python -m benchmark.evaluate --baselines fipla --constant-bid 0.3 --random-max-bid 2.0
    python -m benchmark.evaluate --baselines none   # RL policies only

Available baselines: fipla, constant, random

Adding a custom policy (Python API only — cannot be done from the CLI)
-----------------------------------------------------------------------
The CLI cannot accept arbitrary Python objects, so custom policies must
be passed via the Python API.  Write a small script, e.g. my_eval.py:

    from benchmark.evaluate import run_full_benchmark, EvalConfig
    from dynamicauc.baseline.policies import BiddingPolicy

    class MyPolicy(BiddingPolicy):
        def bid(self, obs):
            return obs["expected_reward"].item() * 0.8

    df = run_full_benchmark(extra_policies={"MyPolicy": MyPolicy()})

Then run:  python my_eval.py
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
    MLPValuationHead,
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
from dynamicauc.wrappers import SB3PolicyWrapper

RESULTS_DIR  = Path(__file__).parent / "results"
MLP_DIR      = MODELS_DIR / "mlp_valuation_head"  # one .pt per scenario_id

AVAILABLE_BASELINES: List[str] = ["fipla", "constant", "random"]


def _make_baseline(name: str, constant_bid: float, random_max_bid: float) -> tuple[str, BiddingPolicy]:
    """Return (display_name, policy) for a named baseline."""
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
    mask_expected_reward: bool      = True   # must match the flag used during training
    # Simulation-based planning policies
    include_greedy:       bool            = False
    include_shading:      bool            = False
    include_mpc:          bool            = False
    simulation_n_rollouts: int            = 10
    shading_alpha:        float           = 0.8
    mpc_model_path:       Optional[str]   = None  # path to a saved MLPValuationHead .pt; trained on the fly if None
    mpc_train_episodes:   int             = 100   # episodes used when training MLPValuationHead from scratch
    mpc_alpha_candidates: List[float]     = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5])
    mpc_planning_horizon: float           = 7.0  # si H=0 => redevient shading à vérifier


def _run_episode(
    policy:      BiddingPolicy,
    env:         gym.Env,
    seed:        int,
    policy_name: str,
) -> pd.DataFrame:
    obs, _info = env.reset(seed=seed)
    # Planning policies (Greedy, Shading, MPC) need a reference to the live
    # env so they can deepcopy it for rollouts.  Ignored by simple policies.
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


def get_planning_policies(cfg: EvalConfig) -> Dict[str, BiddingPolicy]:
    """
    Return the selected baseline policies as a ``{display_name: policy}`` dict.

    Which baselines to include is controlled by ``cfg.baselines``
    (a list of names from ``AVAILABLE_BASELINES``).  Pass an empty list
    or ``["none"]`` to skip baselines entirely.
    """
    if cfg.baselines == ["none"]:
        return {}
    policies: Dict[str, BiddingPolicy] = {}
    for name in cfg.baselines:
        label, policy = _make_baseline(name, cfg.constant_bid, cfg.random_max_bid)
        policies[label] = policy
    return policies


def train_mlp_valuation_head(
    cfg: EvalConfig,
    scenario=None,
    save_path: Optional[Path] = None,
) -> MLPValuationHead:
    """
    Train an ``MLPValuationHead`` on FiPla rollouts and save it to disk.

    The model is saved under ``benchmark/models/mlp_valuation_head/<scenario_id>.pt``
    by default, or to ``save_path`` if provided.  Pass the returned path (or
    ``cfg.mpc_model_path``) to future calls to avoid retraining.

    Parameters
    ----------
    cfg : EvalConfig
        Uses ``cfg.scenario_id``, ``cfg.scenario_overrides``, and
        ``cfg.mpc_train_episodes``.
    scenario : optional
        Pre-built scenario object.  Built from ``cfg`` if not provided.
    save_path : Path, optional
        Override the default save location.
    """
    if scenario is None:
        scenario = dac.make(cfg.scenario_id, **cfg.scenario_overrides)

    out = Path(save_path) if save_path else MLP_DIR / f"{cfg.scenario_id}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Training MLPValuationHead ({cfg.mpc_train_episodes} episodes) …")
    mlp = MLPValuationHead.from_scenario(scenario, n_episodes=cfg.mpc_train_episodes)
    mlp.save(out)
    print(f"  Saved → {out}")
    return mlp


def _load_or_train_mlp(cfg: EvalConfig, scenario) -> MLPValuationHead:
    """Load MLP from cfg.mpc_model_path, default path, or train from scratch."""
    if cfg.mpc_model_path is not None:
        return MLPValuationHead.load(cfg.mpc_model_path)

    default = MLP_DIR / f"{cfg.scenario_id}.pt"
    if default.exists():
        print(f"  Loading cached MLPValuationHead from {default}")
        return MLPValuationHead.load(default)

    return train_mlp_valuation_head(cfg, scenario)


def get_simulation_policies(cfg: EvalConfig, scenario=None) -> Dict[str, BiddingPolicy]:
    """
    Build simulation-based planning policies (GreedyMyopic, Shading, MPC)
    according to the flags in ``cfg``.  Returns an empty dict if none are enabled.

    For MPC, loads the valuation head from ``cfg.mpc_model_path`` if provided,
    then from the default cache path, otherwise trains one from scratch and saves it.
    """
    policies: Dict[str, BiddingPolicy] = {}

    if cfg.include_greedy:
        policies["GreedyMyopic"] = GreedyMyopicPolicy(
            SimulationEstimator(
                pi0=ConstantPolicy(0.0),   # lose all future auctions → isolates value of this single win
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
        if scenario is None:
            raise ValueError(
                "MPCPolicy needs a scenario. Call via run_full_benchmark() or pass scenario."
            )
        mlp = _load_or_train_mlp(cfg, scenario)
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
    """
    Load saved SB3 models and wrap them as ``BiddingPolicy`` objects.

    ``mask_expected_reward`` must match the flag used during training: when
    True, ``expected_reward`` is excluded from the flattened obs so the
    evaluation input matches what the model saw during training.

    Missing model files are skipped with a warning so that a partially-trained
    experiment still runs.
    """
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
    policies.update(get_rl_policies(scenario, list(algos), cfg.mask_expected_reward))
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg, scenario))
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
            "  python -m benchmark.evaluate --baselines constant --constant-bid 0.3\n"
        ),
    )
    parser.add_argument("--algos",           nargs="+", default=["ppo", "sac", "td3"],
                        help="RL algorithms to evaluate (must be trained first).")
    parser.add_argument("--baselines",       nargs="+", default=AVAILABLE_BASELINES,
                        metavar="BASELINE",
                        help=f"Baseline policies to include. Available: {AVAILABLE_BASELINES}. Use 'none' to skip.")
    parser.add_argument("--constant-bid",    type=float, default=0.5,
                        help="Fixed bid value for the Constant baseline (default: 0.5).")
    parser.add_argument("--random-max-bid",  type=float, default=1.0,
                        help="Upper bound for the Random baseline (default: 1.0).")
    parser.add_argument("--n-seeds",         type=int,   default=30,
                        help="Number of evaluation seeds (default: 30).")
    parser.add_argument("--no-save",              action="store_true",
                        help="Do not write results to CSV.")
    parser.add_argument("--no-mask-expected-reward", action="store_true",
                        help="Do not exclude expected_reward from RL obs (use only for models "
                             "trained WITHOUT MaskExpectedRewardWrapper).")
    parser.add_argument(
        "--sim-policies", nargs="*", default=["all"],
        choices=["greedy", "shading", "mpc", "all"],
        metavar="POLICY",
        help="Simulation-based policies to include: greedy, shading, mpc, all. E.g. --sim-policies greedy shading",
    )
    parser.add_argument("--n-rollouts",  type=int,   default=10,   help="Monte Carlo rollouts for simulation policies (default: 10).")
    parser.add_argument("--alpha",       type=float, default=0.8,  help="Shading factor α for ShadingPolicy (default: 0.8).")
    parser.add_argument(
        "--mpc-model", type=str, default=None,
        help=(
            "Path to a MLPValuationHead .pt file.  "
            "With --train-mlp-only: save to this path.  "
            "With --sim-policies mpc or --sim-policies all: load from this path (trains and saves to default location if omitted)."
        ),
    )
    parser.add_argument("--mpc-train-ep",   type=int, default=100, help="Episodes to train MLPValuationHead from scratch (default: 100).")
    parser.add_argument("--train-mlp-only", action="store_true",   help="Train and save MLPValuationHead then exit without evaluating.")
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
        mpc_train_episodes=args.mpc_train_ep,
    )

    if args.train_mlp_only:
        # --mpc-model overrides the save path; default used if omitted
        train_mlp_valuation_head(cfg, save_path=args.mpc_model)
        return

    run_full_benchmark(algos=args.algos, cfg=cfg, save=not args.no_save)


if __name__ == "__main__":
    main()
