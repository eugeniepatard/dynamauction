"""
Train PPO with a larger MLP [256, 256] ReLU (same as SAC default) using the
best hyperparameters from the eta=50 sweep, to test whether the small default
[64, 64] Tanh architecture was limiting PPO's performance.

Best eta=50 config:  lr=3e-4, ent=0.02, steps=512, gamma=0.95, epochs=5
Architecture change: net_arch=[256, 256], activation=ReLU  (vs [64,64] Tanh)

Output: /scratch/epatard/models/ppo/lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5_mlp256/
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import submitit
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRATCH_DIR    = Path("/scratch/epatard")
RUN_NAME       = "lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5_mlp256"
TOTAL_TS       = 5_000_000
N_ENVS         = 4
SEED           = 42
SCENARIO_ID    = "survival-second-price-exponential"

_SLURM: dict[str, Any] = {
    "timeout_min":          2880,
    "slurm_partition":      "normal,gpu",
    "gpus_per_node":        0,
    "cpus_per_task":        8,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 120,
    "slurm_job_name":       "dynac_ppo256",
}


def train_ppo_large() -> None:
    from copy import deepcopy
    import gymnasium as gym
    import dynamicauc as dac
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize

    log_dir   = SCRATCH_DIR / "logs"   / "ppo" / RUN_NAME
    model_dir = SCRATCH_DIR / "models" / "ppo" / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    scenario = dac.make(SCENARIO_ID)

    def _make_env():
        env = gym.make("DynamicAuction-v0", scenario=scenario)
        env = dac.MaskExpectedRewardWrapper(env)
        env = dac.FlatAuctionObsWrapper(env)
        env = dac.BidScaleWrapper(env, 5.0)
        return Monitor(env)

    train_env = make_vec_env(_make_env, n_envs=N_ENVS, seed=SEED)
    eval_env  = make_vec_env(_make_env, n_envs=1,      seed=SEED + 9_999)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=SEED,
        learning_rate=3e-4,
        gamma=0.95,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gae_lambda=0.90,
        ent_coef=0.02,
        tensorboard_log=str(log_dir),
        policy_kwargs={"net_arch": [256, 256],
                       "activation_fn": torch.nn.ReLU},
    )

    class _SyncNormEvalCallback(EvalCallback):
        def __init__(self, eval_env_, train_env_, obs_rms_path_, **kw):
            super().__init__(eval_env_, **kw)
            self._train_venv   = train_env_
            self._obs_rms_path = obs_rms_path_
        def _on_step(self) -> bool:
            if isinstance(self._train_venv, VecNormalize) and \
               isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = deepcopy(self._train_venv.obs_rms)
            prev_best = self.best_mean_reward
            result    = super()._on_step()
            if self.best_mean_reward > prev_best:
                with open(self._obs_rms_path, "wb") as f:
                    pickle.dump(self._train_venv.obs_rms, f)
            return result

    eval_cb = _SyncNormEvalCallback(
        eval_env,
        train_env_=train_env,
        obs_rms_path_=model_dir / "obs_rms.pkl",
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TS, callback=eval_cb, reset_num_timesteps=True)

    with open(model_dir / "obs_rms.pkl", "wb") as f:
        pickle.dump(train_env.obs_rms, f)
    model.save(str(model_dir / "final_model"))
    print(f"Done → {model_dir}")


if __name__ == "__main__":
    executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "ppo256" / "%j")
    )
    executor.update_parameters(**_SLURM)
    job = executor.submit(train_ppo_large)
    print(f"Submitted → job {job.job_id}")
    print(f"Model  → {SCRATCH_DIR}/models/ppo/{RUN_NAME}/")
    print(f"Logs   → {SCRATCH_DIR}/logs/ppo/{RUN_NAME}/")
    print("Check  : squeue -u epatard")
