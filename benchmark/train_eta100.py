"""
Train the best Phase-1 model for each algorithm from scratch on eta=100.

Why from scratch and not fine-tuning
─────────────────────────────────────
  The Phase-1 sweep used the default scenario horizon distribution
  ExponentialHorizon(eta=50).  At eta=100 the mean horizon is twice as long,
  so bids near the end of a campaign carry very different value.  The obs_rms
  statistics and the value functions are no longer calibrated, making
  fine-tuning unreliable.  A full retraining on eta=100 from scratch is the
  only way to get a fair apples-to-apples comparison with the MPC baselines
  (which are also evaluated on eta=100).

Chosen configurations
─────────────────────
  These are the single best configuration per algorithm from the Phase-1
  eta=50 sweep (ranked by best_mean_reward over 5 M timesteps):

    TD3  lr=0.0001  bs=512  noise=0.1  warmup=500  gamma=0.99  → best 3.655
    SAC  lr=5e-05   bs=512  warmup=500 gamma=0.95              → best 2.784
    PPO  lr=0.0003  ent=0.02 steps=512 gamma=0.95  epochs=5   → best 3.595

  All other hyper-parameters (n_envs, seed, eval_freq, VecNormalize, …) match
  Phase-1 exactly so results are comparable.

Training budget
───────────────
  total_timesteps = 5 000 000  — same as Phase-1 sweep.

Checkpoint / preemption handling
─────────────────────────────────
  TrainEta100Job is a callable class with a checkpoint() method so that
  SLURM preemption resumes training instead of restarting from scratch.
  A CheckpointCallback saves model weights (+ replay buffer for TD3/SAC,
  + VecNormalize stats) every CHECKPOINT_FREQ steps.  On restart the job
  detects the latest checkpoint and resumes with reset_num_timesteps=False,
  so the total training budget is always honoured.

Output paths
────────────
  Models      : /scratch/epatard/eta100/models/<algo>/<run_name>/
  Logs        : /scratch/epatard/eta100/logs/<algo>/<run_name>/
  Checkpoints : /scratch/epatard/eta100/checkpoints/<algo>/<run_name>/
  (Separate from eta=50 artefacts — nothing is overwritten.)

Usage (from repo root, on the cluster)
───────────────────────────────────────
    python benchmark/train_eta100.py --dry-run   # inspect, don't submit
    python benchmark/train_eta100.py             # submit all 3 jobs
    python benchmark/train_eta100.py --algo td3  # single algo
"""
from __future__ import annotations

import argparse
import pickle
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import submitit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRATCH_DIR    = Path("/scratch/epatard")
SCRATCH_ETA100 = SCRATCH_DIR / "eta100"

HORIZON_ETA      = 100
TOTAL_TIMESTEPS  = 5_000_000
CHECKPOINT_FREQ  = 100_000   # save a resumable checkpoint every 100k steps

# ── Training constants (match Phase-1 sweep) ──────────────────────────────────

N_ENVS          = 4
SEED            = 42
MAX_BID         = 5.0
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 20

# ── Best configuration per algorithm (from Phase-1 eta=50 sweep) ──────────────

_BEST: dict[str, tuple[str, dict]] = {
    "td3": (
        "lr=0.0001_bs=512_noise=0.1_warmup=500_gamma=0.99",
        dict(
            learning_rate=1e-4,
            off_policy_batch_size=512,
            td3_action_noise_sigma=0.1,
            off_policy_learning_starts=500,
            gamma=0.99,
        ),
    ),
    "sac": (
        "lr=5e-05_bs=512_warmup=500_gamma=0.99",
        dict(
            learning_rate=5e-5,
            off_policy_batch_size=512,
            off_policy_learning_starts=500,
            gamma=0.99,
        ),
    ),
    "ppo": (
        "lr=0.0003_ent=0.02_steps=512_gamma=0.99_epochs=5",
        dict(
            learning_rate=3e-4,
            ppo_ent_coef=0.02,
            ppo_n_steps=512,
            gamma=0.99,
            ppo_n_epochs=5,
        ),
    ),
}

# ── SLURM parameters ──────────────────────────────────────────────────────────

_SLURM: dict[str, Any] = {
    "timeout_min":          2880,
    "slurm_partition":      "normal,gpu",   # normal (not normal-best) → fewer preemptions
    "gpus_per_node":        0,
    "cpus_per_task":        8,
    "tasks_per_node":       1,
    "nodes":                1,
    "mem_gb":               8,
    "slurm_signal_delay_s": 120,            # 120 s warning before SIGTERM → time to checkpoint
    "slurm_job_name":       "dynac_eta100",
}

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _latest_checkpoint(ckpt_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    Return (model_path, vecnorm_path) for the highest-step checkpoint, or
    (None, None) if no checkpoint exists.
    model_path has no .zip suffix (SB3 convention for load()).
    """
    if not ckpt_dir.exists():
        return None, None
    zips = list(ckpt_dir.glob("rl_model_*_steps.zip"))
    if not zips:
        return None, None

    def _parse_steps(p: Path) -> int:
        try:
            return int(p.stem.split("_")[2])
        except (IndexError, ValueError):
            return -1

    latest   = max(zips, key=_parse_steps)
    model_p  = latest.with_suffix("")                          # strip .zip
    vecnorm_p = ckpt_dir / (latest.stem + "_vecnormalize.pkl")
    return model_p, vecnorm_p if vecnorm_p.exists() else None


# ── Callable job class with checkpoint() support ──────────────────────────────

class TrainEta100Job:
    """
    Submitit-compatible job class.  checkpoint() is called by submitit when
    SLURM sends a preemption signal; it simply returns a new DelayedSubmission
    so the job is requeued.  On the next run, __call__ detects the latest
    CheckpointCallback snapshot and resumes training from that point.
    """

    def __call__(self, algo: str, run_name: str, overrides: dict) -> None:
        import gymnasium as gym
        import dynamicauc as dac
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.vec_env import VecNormalize

        _ALGO_CLS = {"ppo": PPO, "sac": SAC, "td3": TD3}

        log_dir   = SCRATCH_ETA100 / "logs"        / algo / run_name
        model_dir = SCRATCH_ETA100 / "models"      / algo / run_name
        ckpt_dir  = SCRATCH_ETA100 / "checkpoints" / algo / run_name
        for d in (log_dir, model_dir, ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Environment factory ───────────────────────────────────────────────
        scenario = dac.make("survival-second-price-exponential",
                            horizon_eta=HORIZON_ETA)

        def _make_env() -> gym.Env:
            env = gym.make("DynamicAuction-v0", scenario=scenario)
            env = dac.MaskExpectedRewardWrapper(env)
            env = dac.FlatAuctionObsWrapper(env)
            env = dac.BidScaleWrapper(env, MAX_BID)
            return Monitor(env)

        train_env = make_vec_env(_make_env, n_envs=N_ENVS, seed=SEED)
        eval_env  = make_vec_env(_make_env, n_envs=1,      seed=SEED + 9_999)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False,
                                 training=False)

        # ── Detect and restore checkpoint ─────────────────────────────────────
        ckpt_model, ckpt_vecnorm = _latest_checkpoint(ckpt_dir)
        resuming = ckpt_model is not None

        if resuming:
            print(f"[{algo.upper()}/{run_name}] Resuming from {ckpt_model}")
            if ckpt_vecnorm:
                train_env = VecNormalize.load(str(ckpt_vecnorm), train_env.venv)
                train_env.training = True
                train_env.norm_reward = True
            model = _ALGO_CLS[algo].load(str(ckpt_model), env=train_env)
            print(f"  Resuming at timestep {model.num_timesteps:,} / {TOTAL_TIMESTEPS:,}")
        else:
            print(f"[{algo.upper()}/{run_name}] Fresh start")
            cfg = replace(
                __import__("benchmark.train_rl", fromlist=["TrainingConfig"]).TrainingConfig(
                    total_timesteps=TOTAL_TIMESTEPS,
                    scratch_dir=SCRATCH_ETA100,
                    verbose=False,
                ),
                run_name=run_name,
                **overrides,
            )
            from benchmark.train_rl import _build_model
            model = _build_model(algo, train_env, cfg, log_dir)

        # TD3: re-attach action noise (not serialised in .zip)
        if algo == "td3":
            sigma = overrides.get("td3_action_noise_sigma", 0.1)
            model.action_noise = NormalActionNoise(
                mean=np.zeros(1), sigma=sigma * np.ones(1)
            )

        # ── Callbacks ─────────────────────────────────────────────────────────
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
                # Save obs_rms whenever a new best_model.zip is written
                if self.best_mean_reward > prev_best:
                    with open(self._obs_rms_path, "wb") as fh:
                        pickle.dump(self._train_venv.obs_rms, fh)
                return result

        eval_cb = _SyncNormEvalCallback(
            eval_env,
            train_env_=train_env,
            obs_rms_path_=model_dir / "obs_rms.pkl",
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        )

        ckpt_cb = CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path=str(ckpt_dir),
            name_prefix="rl_model",
            save_replay_buffer=True,   # TD3/SAC: preserve replay buffer across preemptions
            save_vecnormalize=True,    # preserve obs normalisation stats
            verbose=1,
        )

        # ── Train ─────────────────────────────────────────────────────────────
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=not resuming,
        )

        # ── Persist final artefacts ───────────────────────────────────────────
        with open(model_dir / "obs_rms.pkl", "wb") as f:
            pickle.dump(train_env.obs_rms, f)
        model.save(str(model_dir / "final_model"))
        print(f"[{algo.upper()}/{run_name}] Done → {model_dir}")

    def checkpoint(self, algo: str, run_name: str, overrides: dict) -> submitit.helpers.DelayedSubmission:
        """
        Called by submitit on SLURM preemption (SIGTERM received).
        CheckpointCallback has already saved the latest model snapshot, so we
        simply requeue the same job — __call__ will detect and restore it.
        """
        return submitit.helpers.DelayedSubmission(self, algo, run_name, overrides)


# ── CLI / SLURM submission ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo", nargs="+", choices=list(_BEST), default=list(_BEST),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "eta100" / "%j")
    )
    executor.update_parameters(**_SLURM)

    jobs_to_submit = [(algo, *_BEST[algo]) for algo in args.algo]
    job_obj        = TrainEta100Job()

    if args.dry_run:
        for algo, run_name, overrides in jobs_to_submit:
            ckpt_model, _ = _latest_checkpoint(
                SCRATCH_ETA100 / "checkpoints" / algo / run_name
            )
            status = f"resume from {ckpt_model}" if ckpt_model else "fresh start"
            print(f"  [DRY RUN] {algo.upper()} — {run_name} ({status})")
        print(f"\nTotal: {len(jobs_to_submit)} jobs (dry run — nothing submitted)")
        return

    jobs = executor.map_array(
        job_obj,
        [algo       for algo, *_        in jobs_to_submit],
        [run_name   for _, run_name, _  in jobs_to_submit],
        [overrides  for _, _, overrides in jobs_to_submit],
    )
    ids = [j.job_id for j in jobs]
    for (algo, run_name, _), jid in zip(jobs_to_submit, ids):
        print(f"  [{algo.upper()}] {run_name}  → job {jid}")
    print(f"\nTotal submitted: {len(ids)} jobs")
    print("Check status   : squeue -u epatard")
    print(f"Models → {SCRATCH_ETA100 / 'models'}")


if __name__ == "__main__":
    main()
