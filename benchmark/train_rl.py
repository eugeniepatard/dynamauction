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

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import dynamicauc as dac

import hydra
from omegaconf import DictConfig


BENCHMARK_DIR = Path(__file__).parent
MODELS_DIR    = BENCHMARK_DIR / "models"
LOGS_DIR      = BENCHMARK_DIR / "logs"

SCENARIO_ID   = "survival-second-price-exponential"

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
    env = dac.FlatAuctionObsWrapper(env)
    env = dac.BidScaleWrapper(env, max_bid)
    env = Monitor(env)
    return env

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

def _cfg_to_training_config(cfg: DictConfig) -> TrainingConfig:
    """Build a TrainingConfig from the Hydra DictConfig."""
    t   = cfg.training
    a   = cfg.algo
    kw  = dict(
        total_timesteps=t.total_timesteps,
        n_envs=t.n_envs,
        seed=t.seed,
        max_bid=t.max_bid,
        eval_freq=t.eval_freq,
        n_eval_episodes=t.n_eval_episodes,
        learning_rate=a.learning_rate,
        gamma=a.gamma,
    )
    if a.name == "ppo":
        kw.update(ppo_n_steps=a.n_steps, ppo_batch_size=a.batch_size,
                  ppo_n_epochs=a.n_epochs)
    else:
        kw.update(off_policy_batch_size=a.batch_size,
                  off_policy_learning_starts=a.learning_starts)
    return TrainingConfig(**kw)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Train one algorithm with the config resolved by Hydra.

    Single run (default algo = ppo):
        python -m benchmark.train_rl

    Select algorithm:
        python -m benchmark.train_rl algo=sac
        python -m benchmark.train_rl algo=td3 training.total_timesteps=1000000

    Train all algorithms in parallel (one process per algo):
        python -m benchmark.train_rl 'algo=ppo,sac,td3' --multirun
    """
    training_cfg = _cfg_to_training_config(cfg)
    scenario_overrides = dict(cfg.scenario_overrides) if cfg.scenario_overrides else {}
    train(cfg.algo.name, training_cfg, scenario_overrides)


if __name__ == "__main__":
    main()
