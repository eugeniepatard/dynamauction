"""
PPO hyperparameter grid search for saturation-second-price-uniform-eta50.

Designed to run as a SLURM job array where SLURM_ARRAY_TASK_ID selects the
hyperparameter combination.  Each trial trains PPO for 1 M steps, then
evaluates whether the policy escaped the zero-bid degenerate optimum.

Grid (5 × 3 × 3 × 3 = 135 trials):
  ent_coef      [0.01, 0.05, 0.1, 0.2, 0.5]
  learning_rate [1e-4, 3e-4, 1e-3]
  n_steps       [512, 1024, 2048]
  gae_lambda    [0.90, 0.95, 0.99]

Non-degenerate metric
  mean_bid        average bid across all evaluation steps
  frac_nonzero    fraction of bids > DEGENERATE_THRESHOLD (0.05)
  mean_reward     mean episode reward over collected episodes
  is_degenerate   True when mean_bid < DEGENERATE_THRESHOLD

Submit:
  sbatch --array=0-134 benchmark/run_scripts/run_ppo_hparam_search.sh

Run one trial locally:
  TRIAL_ID=0 python benchmark/ppo_hparam_search.py
  python benchmark/ppo_hparam_search.py 42
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dynamauction as dac
from benchmark.train_rl import (
    BENCHMARK_DIR,
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
SCENARIO_ID           = "saturation-second-price-uniform-eta50"
TOTAL_TIMESTEPS       = 1_000_000
N_ENVS                = 4
BATCH_SIZE            = 128   # divides n_steps * N_ENVS for all grid n_steps values
N_EVAL_STEPS          = 10_000  # steps collected in post-training evaluation
DEGENERATE_THRESHOLD  = 0.05    # bids below this count as zero (collapsed policy)
MAX_BID               = 5.0


def get_hparams(trial_id: int) -> dict:
    """Return the hyperparameter dict for a given trial index."""
    if trial_id >= N_TRIALS:
        raise ValueError(f"trial_id={trial_id} out of range [0, {N_TRIALS - 1}]")
    return dict(zip(GRID.keys(), COMBOS[trial_id]))


def train_and_evaluate(trial_id: int, scratch_dir: Path) -> dict:
    hp       = get_hparams(trial_id)
    run_name = f"hparam_search/trial_{trial_id:03d}"

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
        # Searched hyperparameters
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

    # ── Post-training evaluation: non-degenerate check ───────────────────────
    # Load best model in a fresh normalised env and collect bids + rewards.
    eval_vn = VecNormalize(
        make_vec_env(_make_env, n_envs=1, seed=0),
        norm_obs=True, norm_reward=False, training=False,
    )
    with open(obs_rms_path, "rb") as f:
        eval_vn.obs_rms = pickle.load(f)

    best = PPO.load(str(model_dir / "best_model"), env=eval_vn)

    obs = eval_vn.reset()
    raw_actions: list[float] = []
    ep_rewards:  list[float] = []

    for _ in range(N_EVAL_STEPS):
        action, _ = best.predict(obs, deterministic=True)
        # action shape: (n_envs=1, action_dim=1); value in [-1, 1]
        raw_actions.append(float(action[0, 0]))
        obs, _, done, info = eval_vn.step(action)
        if done[0] and "episode" in info[0]:
            ep_rewards.append(info[0]["episode"]["r"])

    actions_arr = np.array(raw_actions)
    # BidScaleWrapper maps [-1, 1] → [0, MAX_BID]
    bids = (actions_arr + 1.0) / 2.0 * MAX_BID

    mean_bid      = float(bids.mean())
    frac_nonzero  = float((bids > DEGENERATE_THRESHOLD).mean())
    mean_reward   = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    is_degenerate = mean_bid < DEGENERATE_THRESHOLD

    result = {
        "trial_id": trial_id,
        "hparams":  hp,
        "metrics": {
            "mean_bid":      mean_bid,
            "frac_nonzero":  frac_nonzero,
            "mean_reward":   mean_reward,
            "n_episodes":    len(ep_rewards),
            "is_degenerate": is_degenerate,
        },
    }
    print(
        f"[trial {trial_id:03d}]  "
        f"ent_coef={hp['ent_coef']:.3f}  lr={hp['learning_rate']:.0e}  "
        f"n_steps={hp['n_steps']}  gae={hp['gae_lambda']:.2f}  │  "
        f"mean_bid={mean_bid:.3f}  frac_nz={frac_nonzero:.2f}  "
        f"reward={mean_reward:.3f}  degenerate={is_degenerate}"
    )
    return result


def main() -> None:
    trial_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID",
                                     sys.argv[1] if len(sys.argv) > 1 else "0"))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", "/scratch/epatard/dynamicauc"))

    out_dir  = scratch_dir / "ppo_hparam_search"
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
