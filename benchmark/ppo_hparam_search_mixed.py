"""
PPO hyperparameter grid search for mixed-second-price-exponential-eta50.

Same grid as ppo_hparam_search.py (saturation), adapted for the immediate-reward
survival kernel to find the best PPO configuration for the RL diagnostic experiment.

Grid (5 × 3 × 3 × 3 = 135 trials):
  ent_coef      [0.01, 0.05, 0.1, 0.2, 0.5]
  learning_rate [1e-4, 3e-4, 1e-3]
  n_steps       [512, 1024, 2048]
  gae_lambda    [0.90, 0.95, 0.99]

Each trial trains PPO for 1M steps on mixed-second-price-exponential-eta50
(eta=50, mask_expected_reward=True — same observation space as canonical delayed).

Submit:
  sbatch benchmark/run_scripts/run_ppo_hparam_search_mixed.sh

Run one trial locally:
  TRIAL_ID=0 python benchmark/ppo_hparam_search_mixed.py
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dynamauction as dac
from benchmark.train_rl import (
    TrainingConfig,
    _SyncNormEvalCallback,
    _build_model,
    _env_factory,
)

# ── Hyperparameter grid ──────────────────────────────────────────────────────
GRID = {
    "ent_coef":      [0.01, 0.05, 0.1, 0.2, 0.5],
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps":       [512, 1024, 2048],
    "gae_lambda":    [0.90, 0.95, 0.99],
}
COMBOS   = list(itertools.product(*GRID.values()))
N_TRIALS = len(COMBOS)   # 135

# ── Constants ────────────────────────────────────────────────────────────────
SCENARIO_ID      = "mixed-second-price-exponential-eta50"
HORIZON_ETA      = 50
TOTAL_TIMESTEPS  = 1_000_000
N_ENVS           = 4
BATCH_SIZE       = 128
N_EVAL_STEPS     = 10_000
MAX_BID          = 5.0


def get_hparams(trial_id: int) -> dict:
    if trial_id >= N_TRIALS:
        raise ValueError(f"trial_id={trial_id} out of range [0, {N_TRIALS - 1}]")
    return dict(zip(GRID.keys(), COMBOS[trial_id]))


def train_and_evaluate(trial_id: int, scratch_dir: Path) -> dict:
    hp       = get_hparams(trial_id)
    run_name = f"hparam_search_mixed/trial_{trial_id:03d}"

    cfg = TrainingConfig(
        total_timesteps      = TOTAL_TIMESTEPS,
        use_vec_normalize    = True,
        mask_expected_reward = True,
        n_envs               = N_ENVS,
        seed                 = 42,
        max_bid              = MAX_BID,
        eval_freq            = 100_000,
        n_eval_episodes      = 20,
        verbose              = False,
        scratch_dir          = scratch_dir,
        run_name             = run_name,
        learning_rate        = hp["learning_rate"],
        ppo_ent_coef         = hp["ent_coef"],
        ppo_n_steps          = hp["n_steps"],
        ppo_batch_size       = BATCH_SIZE,
        ppo_n_epochs         = 10,
        ppo_gae_lambda       = hp["gae_lambda"],
    )

    model_dir = scratch_dir / "models" / "ppo" / run_name
    log_dir   = scratch_dir / "logs"   / "ppo" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scenario = dac.make(SCENARIO_ID)

    def _make_env():
        return _env_factory(scenario, MAX_BID, mask_expected_reward=True)

    train_env = VecNormalize(
        make_vec_env(_make_env, n_envs=N_ENVS, seed=cfg.seed),
        norm_obs=True, norm_reward=True,
    )
    eval_env = VecNormalize(
        make_vec_env(_make_env, n_envs=1, seed=cfg.seed + 9_999),
        norm_obs=True, norm_reward=False, training=False,
    )

    model = _build_model("ppo", train_env, cfg, log_dir)

    eval_cb = _SyncNormEvalCallback(
        eval_env,
        train_env=train_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(cfg.eval_freq // N_ENVS, 1),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_cb,
        tb_log_name="ppo",
        reset_num_timesteps=True,
    )

    obs_rms_path = model_dir / "obs_rms.pkl"
    with open(obs_rms_path, "wb") as f:
        pickle.dump(train_env.obs_rms, f)

    # ── Post-training evaluation ─────────────────────────────────────────────
    eval_vn = VecNormalize(
        make_vec_env(_make_env, n_envs=1, seed=0),
        norm_obs=True, norm_reward=False, training=False,
    )
    with open(obs_rms_path, "rb") as f:
        eval_vn.obs_rms = pickle.load(f)

    best = PPO.load(str(model_dir / "best_model"), env=eval_vn)

    obs = eval_vn.reset()
    ep_rewards: list[float] = []

    for _ in range(N_EVAL_STEPS):
        action, _ = best.predict(obs, deterministic=True)
        obs, _, done, info = eval_vn.step(action)
        if done[0] and "episode" in info[0]:
            ep_rewards.append(info[0]["episode"]["r"])

    mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0

    result = {
        "trial_id": trial_id,
        "hparams":  hp,
        "metrics": {
            "mean_reward": mean_reward,
            "n_episodes":  len(ep_rewards),
        },
    }
    print(
        f"[trial {trial_id:03d}]  "
        f"ent_coef={hp['ent_coef']:.3f}  lr={hp['learning_rate']:.0e}  "
        f"n_steps={hp['n_steps']}  gae={hp['gae_lambda']:.2f}  │  "
        f"reward={mean_reward:.3f}  n_ep={len(ep_rewards)}"
    )
    return result


def main() -> None:
    trial_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID",
                                     sys.argv[1] if len(sys.argv) > 1 else "0"))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", "/scratch/epatard/dynamicauc"))

    out_dir  = scratch_dir / "ppo_hparam_search_mixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"trial_{trial_id:03d}.json"

    if out_file.exists():
        print(f"[trial {trial_id}] Already done ({out_file}), skipping.")
        return

    result = train_and_evaluate(trial_id, scratch_dir=scratch_dir)

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[trial {trial_id}] Saved → {out_file}")


if __name__ == "__main__":
    main()
