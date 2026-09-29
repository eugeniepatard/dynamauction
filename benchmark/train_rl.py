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

Train with Hydra CLI
-------------------

    Single run (default algo = ppo):
        python -m benchmark.train_rl

    Select algorithm:
        python -m benchmark.train_rl algo=sac
        python -m benchmark.train_rl algo=td3 training.total_timesteps=1000000

    Train all algorithms in parallel (one process per algo):
        python -m benchmark.train_rl 'algo=ppo,sac,td3' --multirun

Visualize training curves with TensorBoard
-------------------------------------------
    tensorboard --logdir benchmark/logs/ (during training in another terminal)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import dynamauction as dac
from dynamauction.primitives import FixedHorizon

import hydra
from omegaconf import DictConfig


BENCHMARK_DIR = Path(__file__).parent
MODELS_DIR    = BENCHMARK_DIR / "models"
LOGS_DIR      = BENCHMARK_DIR / "logs"

SCENARIO_ID   = "survival-second-price-exponential"

@dataclass
class TrainingConfig:
    total_timesteps: int   = 500_000
    use_vec_normalize: bool = True
    fixed_horizon: Optional[float] = None   # set via Python API; Hydra users control horizon via scenario_overrides
    run_name:   str            = ""    # optional subdirectory suffix used by sweep jobs to avoid output collisions
    scratch_dir: Optional[Path] = None  # if set, models/logs go here instead of benchmark/models|logs
    n_envs:          int   = 4
    seed:            int   = 42
    learning_rate:   float = 3e-4
    gamma:           float = 0.99
    max_bid:         float = 5.0
    mask_expected_reward: bool = True   # remove oracle signal from RL obs (fair vs FiPla)

    # PPO
    ppo_n_steps:      int   = 1024
    ppo_batch_size:   int   = 128
    ppo_n_epochs:     int   = 10
    ppo_gae_lambda:   float = 0.90
    ppo_ent_coef:     float = 0.0

    # SAC / TD3
    off_policy_batch_size:      int   = 256
    off_policy_learning_starts: int   = 1_000

    # TD3
    td3_action_noise_sigma: float = 0.1   # std in normalized [-1, 1] action space (~10% of bid range)

    # Evaluation during training
    eval_freq:       int = 10_000
    n_eval_episodes: int = 20

    verbose: bool = True

_ALGO_CLS: Dict[str, type] = {"ppo": PPO, "sac": SAC, "td3": TD3}


class _SyncNormEvalCallback(EvalCallback):
    """EvalCallback that syncs VecNormalize obs statistics from train to eval env."""

    def __init__(self, eval_env, train_env: VecNormalize, **kwargs):
        super().__init__(eval_env, **kwargs)
        self._train_venv = train_env

    def _on_step(self) -> bool:
        if isinstance(self._train_venv, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = deepcopy(self._train_venv.obs_rms)
        return super()._on_step()


def _build_model(algo: str, env, cfg: TrainingConfig, log_dir: Path):
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1 if cfg.verbose else 0,
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
            gae_lambda=cfg.ppo_gae_lambda,
            ent_coef=cfg.ppo_ent_coef,
        )
    if algo == "sac":
        return SAC(
            **common,
            batch_size=cfg.off_policy_batch_size,
            learning_starts=cfg.off_policy_learning_starts,
        )
    if algo == "td3":
        sigma = cfg.td3_action_noise_sigma
        return TD3(
            **common,
            batch_size=cfg.off_policy_batch_size,
            learning_starts=cfg.off_policy_learning_starts,
            action_noise=NormalActionNoise(mean=np.zeros(1), sigma=sigma * np.ones(1)),
        )
    raise ValueError(f"Unknown algo {algo!r}. Choose from {list(_ALGO_CLS)}")


def _env_factory(scenario, max_bid: float, mask_expected_reward: bool = True) -> gym.Env:
    env = gym.make("DynamAuction-v0", scenario=scenario)
    if mask_expected_reward:
        env = dac.MaskExpectedRewardWrapper(env)
    env = dac.FlatAuctionObsWrapper(env)
    env = dac.BidScaleWrapper(env, max_bid)
    env = Monitor(env)
    return env

def train(
    algo: str,
    cfg: TrainingConfig = TrainingConfig(),
    scenario_overrides: Optional[dict] = None,
    scenario_id: Optional[str] = None,
) -> object:
    """
    Train a single RL algorithm.

    Parameters
    ----------
    algo : str
        One of ``"ppo"``, ``"sac"``, ``"td3"`` (case-insensitive).
    cfg : TrainingConfig
        Hyper-parameters and training budget.
    scenario_overrides : dict, optional
        Key-value pairs forwarded to ``dac.make()`` to override scenario defaults.
    scenario_id : str, optional
        Scenario preset name.  Defaults to ``survival-second-price-exponential``.

    Returns
    -------
    model : SB3 model
        Trained model.  Also saved to ``benchmark/models/<algo>/``.
    """
    algo = algo.lower()
    if algo not in _ALGO_CLS:
        raise ValueError(f"Unknown algo {algo!r}. Choose from {list(_ALGO_CLS)}")

    sid = scenario_id if scenario_id is not None else SCENARIO_ID
    scenario = dac.make(sid, **(scenario_overrides or {}))
    if cfg.fixed_horizon is not None:
        scenario.horizon_generator = FixedHorizon(cfg.fixed_horizon)

    base      = Path(cfg.scratch_dir) if cfg.scratch_dir else BENCHMARK_DIR
    subdir    = Path(cfg.run_name) if cfg.run_name else Path(".")
    log_dir   = base / "logs"   / algo / subdir
    model_dir = base / "models" / algo / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        lambda: _env_factory(scenario, cfg.max_bid, cfg.mask_expected_reward),
        n_envs=cfg.n_envs,
        seed=cfg.seed,
    )
    eval_env = make_vec_env(
        lambda: _env_factory(scenario, cfg.max_bid, cfg.mask_expected_reward),
        n_envs=1,
        seed=cfg.seed + 9_999,
    )

    if cfg.use_vec_normalize:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, training=False)

    model = _build_model(algo, train_env, cfg, log_dir)

    EvalCbCls = _SyncNormEvalCallback if cfg.use_vec_normalize else EvalCallback

    common_kwargs = dict(
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    if cfg.use_vec_normalize:
        common_kwargs["train_env"] = train_env

    eval_cb = EvalCbCls(eval_env, **common_kwargs)

    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"  Algorithm : {algo.upper()}")
        print(f"  Timesteps : {cfg.total_timesteps:,}")
        print(f"  Scenario  : {sid}")
        print(f"  Horizon   : {'fixed=' + str(cfg.fixed_horizon) if cfg.fixed_horizon else 'exponential'}")
        print(f"  VecNorm   : {cfg.use_vec_normalize}")
        print(f"  MaskER    : {cfg.mask_expected_reward}  (expected_reward hidden from RL obs)")
        print(f"  Models  -> {model_dir}")
        print(f"  Logs    -> {log_dir}")
        print(f"{'='*60}\n")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=eval_cb,
        tb_log_name=algo,
        progress_bar=cfg.verbose,
        reset_num_timesteps=True,
    )

    if cfg.use_vec_normalize:
        with open(model_dir / "obs_rms.pkl", "wb") as f:
            pickle.dump(train_env.obs_rms, f)

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    if cfg.verbose:
        print(f"[{algo.upper()}] Final model -> {final_path}.zip")
        print(f"[{algo.upper()}] Best  model -> {model_dir / 'best_model'}.zip")

    return model


def train_all(
    algos: List[str] = ("ppo", "sac", "td3"),
    cfg: TrainingConfig = TrainingConfig(),
    scenario_overrides: Optional[dict] = None,
) -> Dict[str, object]:
    """Train all listed algorithms sequentially. Returns ``{algo: model}``."""
    models = {}
    for algo in tqdm(list(algos), desc="Algorithms", unit="algo", position=0, leave=True):
        models[algo] = train(algo, cfg, scenario_overrides)
    return models


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
        use_vec_normalize=t.use_vec_normalize,
        mask_expected_reward=t.mask_expected_reward,
        n_envs=t.n_envs,
        seed=t.seed,
        max_bid=t.max_bid,
        eval_freq=t.eval_freq,
        n_eval_episodes=t.n_eval_episodes,
        verbose=t.verbose,
        learning_rate=a.learning_rate,
        gamma=a.gamma,
    )
    if a.name == "ppo":
        kw.update(ppo_n_steps=a.n_steps, ppo_batch_size=a.batch_size,
                  ppo_n_epochs=a.n_epochs, ppo_gae_lambda=a.gae_lambda,
                  ppo_ent_coef=a.ent_coef)
    else:
        kw.update(off_policy_batch_size=a.batch_size,
                  off_policy_learning_starts=a.learning_starts)
        if a.name == "td3":
            kw["td3_action_noise_sigma"] = a.action_noise_sigma
    return TrainingConfig(**kw)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    
    training_cfg = _cfg_to_training_config(cfg)
    scenario_overrides = dict(cfg.scenario_overrides) if cfg.scenario_overrides else {}
    train(cfg.algo.name, training_cfg, scenario_overrides)


if __name__ == "__main__":
    main()
