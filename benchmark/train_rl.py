"""
RL training for bidding policy benchmarks.

Trains PPO, SAC, TD3 on the 'survival-second-price-exponential' scenario using
Stable Baselines3.  Model weights are saved to benchmark/models/<algo>/,
TensorBoard logs to benchmark/logs/<algo>/.

Quick start
-----------
    from benchmark.train_rl import train, train_all, TrainingConfig
    model = train("ppo")
    models = train_all(["ppo", "sac", "td3"], TrainingConfig(total_timesteps=200_000))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import dynamicauc as dac

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent
MODELS_DIR    = BENCHMARK_DIR / "models"
LOGS_DIR      = BENCHMARK_DIR / "logs"

SCENARIO_ID   = "survival-second-price-exponential"


# ── Observation wrapper ────────────────────────────────────────────────────────

class FlatAuctionObsWrapper(gym.ObservationWrapper):
    """
    Flatten the Dict observation space into a 1-D float32 vector.

    The ``step`` field is Discrete(100000); FlattenObservation would one-hot
    encode it to 100K dims.  Here it is kept as a single float scalar instead.
    All Box fields are concatenated in insertion order.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        orig = env.observation_space
        self._keys: List[str] = list(orig.spaces.keys())

        dim = sum(
            1 if isinstance(sp, spaces.Discrete) else int(np.prod(sp.shape))
            for sp in orig.spaces.values()
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        parts = []
        for key in self._keys:
            v  = obs[key]
            sp = self.env.observation_space.spaces[key]
            if isinstance(sp, spaces.Discrete):
                parts.append(np.array([float(v)], dtype=np.float32))
            else:
                parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts)


# ── Action wrapper ─────────────────────────────────────────────────────────────

class BidScaleWrapper(gym.ActionWrapper):
    """
    Restrict the bid action space to [0, max_bid].

    The underlying env has action_space = Box([0, 1e12]), which is valid but
    too wide for stable RL training.  Since expected bids under the survival
    kernel are in [0, 1], max_bid=5 provides a generous safe margin.
    The wrapper passes the bid through unchanged (just clips).
    """

    def __init__(self, env: gym.Env, max_bid: float = 5.0) -> None:
        super().__init__(env)
        self._max_bid = max_bid
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(max_bid),
            shape=(1,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, 0.0, self._max_bid).astype(np.float32)


# ── Training configuration ─────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    total_timesteps: int   = 500_000
    n_envs:          int   = 4
    seed:            int   = 42
    learning_rate:   float = 3e-4
    gamma:           float = 0.99
    max_bid:         float = 5.0

    # PPO
    ppo_n_steps:    int = 1024
    ppo_batch_size: int = 128
    ppo_n_epochs:   int = 10

    # SAC / TD3
    off_policy_batch_size:     int = 256
    off_policy_learning_starts: int = 1_000

    # Evaluation during training
    eval_freq:       int = 10_000
    n_eval_episodes: int = 20


# ── Algorithm registry ─────────────────────────────────────────────────────────

_ALGO_CLS: Dict[str, type] = {"ppo": PPO, "sac": SAC, "td3": TD3}


def _build_model(algo: str, env, cfg: TrainingConfig, log_dir: Path):
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tensorboard_log=str(log_dir),
    )
    if algo == "ppo":
        return PPO(
            **common,
            n_steps=cfg.ppo_n_steps,
            batch_size=cfg.ppo_batch_size,
            n_epochs=cfg.ppo_n_epochs,
        )
    if algo == "sac":
        return SAC(
            **common,
            batch_size=cfg.off_policy_batch_size,
            learning_starts=cfg.off_policy_learning_starts,
        )
    if algo == "td3":
        return TD3(
            **common,
            batch_size=cfg.off_policy_batch_size,
            learning_starts=cfg.off_policy_learning_starts,
        )
    raise ValueError(f"Unknown algo {algo!r}. Choose from {list(_ALGO_CLS)}")


def _env_factory(scenario, max_bid: float) -> gym.Env:
    env = gym.make("DynamicAuction-v0", scenario=scenario)
    env = FlatAuctionObsWrapper(env)
    env = BidScaleWrapper(env, max_bid)
    env = Monitor(env)
    return env


# ── Public API ─────────────────────────────────────────────────────────────────

def train(
    algo: str,
    cfg: TrainingConfig = TrainingConfig(),
    scenario_overrides: Optional[dict] = None,
) -> object:
    """
    Train a single RL algorithm on the survival-second-price-exponential scenario.

    Parameters
    ----------
    algo : str
        One of ``"ppo"``, ``"sac"``, ``"td3"`` (case-insensitive).
    cfg : TrainingConfig
        Hyper-parameters and training budget.
    scenario_overrides : dict, optional
        Key-value pairs forwarded to ``dac.make()`` to override scenario defaults.

    Returns
    -------
    model : SB3 model
        Trained model.  Also saved to ``benchmark/models/<algo>/``.
    """
    algo = algo.lower()
    if algo not in _ALGO_CLS:
        raise ValueError(f"Unknown algo {algo!r}. Choose from {list(_ALGO_CLS)}")

    scenario = dac.make(SCENARIO_ID, **(scenario_overrides or {}))

    log_dir   = LOGS_DIR   / algo
    model_dir = MODELS_DIR / algo
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        lambda: _env_factory(scenario, cfg.max_bid),
        n_envs=cfg.n_envs,
        seed=cfg.seed,
    )
    eval_env = make_vec_env(
        lambda: _env_factory(scenario, cfg.max_bid),
        n_envs=1,
        seed=cfg.seed + 9_999,
    )

    model = _build_model(algo, train_env, cfg, log_dir)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    print(f"\n{'='*60}")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Timesteps : {cfg.total_timesteps:,}")
    print(f"  Scenario  : {SCENARIO_ID}")
    print(f"  Models  → {model_dir}")
    print(f"  Logs    → {log_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=eval_cb,
        tb_log_name=algo,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"[{algo.upper()}] Final model → {final_path}.zip")
    print(f"[{algo.upper()}] Best  model → {model_dir / 'best_model'}.zip")

    return model


def train_all(
    algos: List[str] = ("ppo", "sac", "td3"),
    cfg: TrainingConfig = TrainingConfig(),
    scenario_overrides: Optional[dict] = None,
) -> Dict[str, object]:
    """Train all listed algorithms sequentially. Returns ``{algo: model}``."""
    return {a: train(a, cfg, scenario_overrides) for a in algos}


def load_model(algo: str, path: Optional[str | Path] = None) -> object:
    """
    Load a saved SB3 model from disk.

    Defaults to ``benchmark/models/<algo>/best_model.zip`` (saved by EvalCallback).
    Pass ``path`` to load from a custom location (without the ``.zip`` extension).
    """
    algo = algo.lower()
    cls  = _ALGO_CLS.get(algo)
    if cls is None:
        raise ValueError(f"Unknown algo {algo!r}. Choose from {list(_ALGO_CLS)}")
    if path is None:
        path = MODELS_DIR / algo / "best_model"
    return cls.load(str(path))


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL bidding policies.")
    parser.add_argument("--algos", nargs="+", default=["ppo", "sac", "td3"],
                        help="Algorithms to train (default: ppo sac td3)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    cfg = TrainingConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    train_all(args.algos, cfg)
