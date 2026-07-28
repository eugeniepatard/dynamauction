"""
Continue training of the 4 Phase-1 runs with real potential to beat the paper best.

Candidate selection
───────────────────
  Out of 22 still-improving TD3 runs and 55 PPO runs, only 4 had a Phase-1
  best reward close enough (<0.1 gap) to the current paper-best to realistically
  surpass it with extended training:

  TD3 (paper best = 3.655)
    • lr=5e-05_bs=512_noise=0.2_warmup=1000_gamma=0.99  — Phase-1 best 3.578, norm_slope 0.28
    • lr=5e-05_bs=256_noise=0.1_warmup=1500_gamma=0.95  — Phase-1 best 3.573, norm_slope 0.45

  PPO (paper best = 3.595)
    • lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5  — Phase-1 best 3.595, norm_slope 0.19
      (this IS the current paper-best model — still improving, so worth extending)
    • lr=0.0003_ent=0.01_steps=512_gamma=0.99_epochs=10 — Phase-1 best 3.551, norm_slope 0.20

  All other 73 originally-flagged runs had gaps ≥ 0.3 to the paper best and were
  dropped to avoid wasting cluster time before the paper deadline.

Convergence stopping criterion
──────────────────────────────
  CONVERGENCE_WINDOW    = 40 evals   — sliding window (~400 000 timesteps at
                                        eval_freq = 10 000, n_envs = 4)
  CONV_THRESHOLD        = 0.02       — normalised slope below which we declare
                                        the curve flat (< 2 % of the Phase-2
                                        reward range gained per 5 M steps)
  MIN_PHASE2_EVALS      = 40         — minimum new evals before the stopper
                                        activates (avoids premature stopping on
                                        transient flat patches early in Phase 2)
  MAX_EXTRA_TIMESTEPS   = 15_000_000 — hard safety cap (3× original Phase-1
                                        budget)

Model loading
─────────────
  • Weights     : best_model.zip  (highest-reward Phase-1 checkpoint).
                  Backed up to best_model_phase1.zip before the first Phase-2
                  write, so Phase-1 data is never lost.
  • Obs scaling : obs_rms.pkl is restored into the VecNormalize wrapper so that
                  observation normalisation is consistent with the saved policy.
                  The eval env inherits the same statistics at each eval step
                  via _SyncNormEvalCallback (same mechanism as Phase 1).
  • Replay buf  : NOT restored — SB3 does not save the replay buffer by default.
                  TD3 restarts with an empty buffer but inherits the optimised
                  policy and Q-networks, so performance recovers quickly
                  (~learning_starts steps).
  • Timestep    : reset_num_timesteps=False — the SB3 counter continues from the
                  checkpoint's num_timesteps, keeping Phase-2 eval timestamps
                  contiguous with Phase-1.
  • best_model.zip is only overwritten by Phase-2 if a strictly higher mean
                  reward is achieved (EvalCallback.best_mean_reward is seeded
                  from the Phase-1 backup).
  • evaluations.npz is overwritten with Phase-2 data; the Phase-1 file is
                  backed up as evaluations_phase1.npz.

Usage (from repo root, on the cluster)
───────────────────────────────────────
    python benchmark/continue_training.py --dry-run     # inspect, don't submit
    python benchmark/continue_training.py               # submit TD3 + PPO
    python benchmark/continue_training.py --algo td3    # TD3 only
    python benchmark/continue_training.py --algo ppo    # PPO only
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import submitit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("/scratch/epatard")
LOGS_DIR    = SCRATCH_DIR / "logs"
MODELS_DIR  = SCRATCH_DIR / "models"
SCENARIO_ID = "survival-second-price-exponential"

# ── Curated candidates (runs within 0.1 of the paper best, still improving) ──

_CANDIDATES: dict[str, list[str]] = {
    "td3": [
        "lr=5e-05_bs=512_noise=0.2_warmup=1000_gamma=0.99",  # best 3.578, gap 0.08
        "lr=5e-05_bs=256_noise=0.1_warmup=1500_gamma=0.95",  # best 3.573, gap 0.08
    ],
    "ppo": [
        "lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5",  # best 3.595, gap ≈0 (paper model)
        "lr=0.0003_ent=0.01_steps=512_gamma=0.99_epochs=10", # best 3.551, gap 0.04
    ],
}

# ── Phase-2 convergence stopper ───────────────────────────────────────────────

CONVERGENCE_WINDOW  = 40        # number of Phase-2 evals in the sliding window
CONV_THRESHOLD      = 0.02      # normalised slope below which training stops
MIN_PHASE2_EVALS    = 40        # min new evals before the stopper is active
MAX_EXTRA_TIMESTEPS = 15_000_000  # hard cap per job (3× original budget)

# ── Training constants (must match Phase-1 sweep) ─────────────────────────────

EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 20
N_ENVS          = 4
SEED            = 42
MAX_BID         = 5.0

# ── SLURM parameters ─────────────────────────────────────────────────────────

_SLURM: dict[str, Any] = {
    "timeout_min":             2880,               # 48 h
    "slurm_partition":         "normal-best,gpu-best",
    "gpus_per_node":           0,
    "cpus_per_task":           8,
    "tasks_per_node":          1,
    "nodes":                   1,
    "mem_gb":                  8,
    "slurm_signal_delay_s":    120,
    "slurm_array_parallelism": 3,
    "slurm_job_name":          "dynac_continue",
}

# ── Callbacks ─────────────────────────────────────────────────────────────────

def _make_callbacks():
    """Import SB3 inside the worker (safe for both local and SLURM execution)."""
    from copy import deepcopy
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize

    class _SyncNormEvalCallback(EvalCallback):
        """Syncs VecNormalize obs statistics from train → eval env before each eval."""
        def __init__(self, eval_env, train_env, **kwargs):
            super().__init__(eval_env, **kwargs)
            self._train_venv = train_env

        def _on_step(self) -> bool:
            if isinstance(self._train_venv, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = deepcopy(self._train_venv.obs_rms)
            return super()._on_step()

    class _ContinueEvalCallback(_SyncNormEvalCallback):
        """
        Extends _SyncNormEvalCallback with a convergence stopper for Phase-2.

        Stops training when the sliding-window normalised slope of the Phase-2
        eval curve drops below CONV_THRESHOLD, ensuring training halts only once
        the reward has genuinely plateaued.
        """
        def __init__(self, eval_env, train_env, conv_window, conv_threshold,
                     min_phase2_evals, **kwargs):
            super().__init__(eval_env, train_env=train_env, **kwargs)
            self._conv_window     = conv_window
            self._conv_threshold  = conv_threshold
            self._min_phase2_evals = min_phase2_evals

        def _on_step(self) -> bool:
            result = super()._on_step()
            n = len(self.evaluations_timesteps)
            if n < max(self._min_phase2_evals, self._conv_window):
                return result

            recent_r  = np.array(self.evaluations_results[-self._conv_window:]).mean(axis=1)
            recent_ts = np.array(self.evaluations_timesteps[-self._conv_window:])
            slope, *_ = scipy_stats.linregress(recent_ts, recent_r)
            all_r     = np.array(self.evaluations_results).mean(axis=1)
            norm_slope = slope * 5_000_000 / (all_r.max() - all_r.min() + 1e-9)

            if abs(norm_slope) < self._conv_threshold:
                print(f"[ConvergenceStopper] norm_slope={norm_slope:.4f} < "
                      f"{self._conv_threshold} after {n} Phase-2 evals — stopping.")
                return False
            return result

    return _ContinueEvalCallback


# ── Job function (runs inside each SLURM task) ────────────────────────────────

def continue_job(algo: str, run_name: str) -> None:
    """Load Phase-1 best_model.zip and train until convergence or the safety cap."""
    import gymnasium as gym
    import dynamicauc as dac
    from stable_baselines3 import PPO, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import VecNormalize

    _ALGO_CLS = {"ppo": PPO, "td3": TD3}

    log_dir   = LOGS_DIR   / algo / run_name
    model_dir = MODELS_DIR / algo / run_name

    # ── Back up Phase-1 artifacts (idempotent) ────────────────────────────────
    for src in [model_dir / "best_model.zip", log_dir / "evaluations.npz"]:
        dst = src.with_stem(src.stem + "_phase1")
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    # ── Seed best_mean_reward from Phase-1 so the callback does not trivially ─
    # ── overwrite best_model.zip on the very first Phase-2 evaluation        ──
    phase1_npz = log_dir / "evaluations_phase1.npz"
    phase1_best_reward = -np.inf
    if phase1_npz.exists():
        d = np.load(phase1_npz)
        phase1_best_reward = float(d["results"].mean(axis=1).max())

    # ── Environment factory (matches Phase-1 exactly) ─────────────────────────
    scenario = dac.make(SCENARIO_ID)

    def _make_env(max_bid: float) -> gym.Env:
        env = gym.make("DynamicAuction-v0", scenario=scenario)
        env = dac.MaskExpectedRewardWrapper(env)
        env = dac.FlatAuctionObsWrapper(env)
        env = dac.BidScaleWrapper(env, max_bid)
        return Monitor(env)

    train_env = make_vec_env(lambda: _make_env(MAX_BID), n_envs=N_ENVS, seed=SEED)
    eval_env  = make_vec_env(lambda: _make_env(MAX_BID), n_envs=1, seed=SEED + 9_999)

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, training=False)

    # ── Restore Phase-1 observation normalisation statistics ──────────────────
    obs_rms_path = model_dir / "obs_rms.pkl"
    if obs_rms_path.exists():
        with open(obs_rms_path, "rb") as f:
            train_env.obs_rms = pickle.load(f)

    # ── Load Phase-1 best weights ─────────────────────────────────────────────
    model = _ALGO_CLS[algo].load(str(model_dir / "best_model"), env=train_env)

    # TD3 action noise is not serialised in .zip — re-attach from the run name
    if algo == "td3":
        sigma = next(
            (float(p.split("=")[1]) for p in run_name.split("_") if p.startswith("noise=")),
            0.1,
        )
        model.action_noise = NormalActionNoise(
            mean=np.zeros(1), sigma=sigma * np.ones(1)
        )

    # ── Build convergence-stopping callback ───────────────────────────────────
    _ContinueEvalCallback = _make_callbacks()
    callback = _ContinueEvalCallback(
        eval_env,
        train_env=train_env,
        conv_window=CONVERGENCE_WINDOW,
        conv_threshold=CONV_THRESHOLD,
        min_phase2_evals=MIN_PHASE2_EVALS,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback.best_mean_reward = phase1_best_reward

    # ── Phase-2 training ──────────────────────────────────────────────────────
    print(f"\n[{algo.upper()}/{run_name}] Phase-2 start "
          f"(phase1_best={phase1_best_reward:.3f}, "
          f"num_timesteps={model.num_timesteps:,})")

    model.learn(
        total_timesteps=MAX_EXTRA_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=False,
    )

    # ── Persist updated normalisation stats and final weights ─────────────────
    with open(obs_rms_path, "wb") as f:
        pickle.dump(train_env.obs_rms, f)

    final_path = model_dir / "final_model_phase2"
    model.save(str(final_path))
    print(f"[{algo.upper()}/{run_name}] Phase-2 done → {final_path}.zip")


# ── CLI / SLURM submission ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo", nargs="+", choices=["td3", "ppo"], default=["td3", "ppo"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "continue" / "%j")
    )
    executor.update_parameters(**_SLURM)

    all_ids: list[str] = []
    for algo in args.algo:
        runs = _CANDIDATES.get(algo, [])
        print(f"\n[{algo.upper()}] {len(runs)} candidate runs")
        if not runs:
            print("  — none.")
            continue
        if args.dry_run:
            for r in runs:
                print(f"  [DRY RUN] continue_job(algo={algo!r}, run_name={r!r})")
            continue
        jobs = executor.map_array(continue_job, [algo] * len(runs), runs)
        ids  = [j.job_id for j in jobs]
        print(f"  Submitted {len(jobs)} jobs — IDs {ids[0]} … {ids[-1]}")
        all_ids.extend(ids)

    n_total = sum(len(v) for k, v in _CANDIDATES.items() if k in args.algo)
    if args.dry_run:
        print(f"\nTotal: {n_total} jobs (dry run — nothing submitted)")
    else:
        print(f"\nTotal submitted: {len(all_ids)} jobs")
        print("Check status   : squeue -u epatard")


if __name__ == "__main__":
    main()
