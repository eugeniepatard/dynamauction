"""
Recompute obs_rms.pkl for best_model.zip by rolling out the saved policy.

The problem this fixes
──────────────────────
  SB3 saves obs_rms.pkl at the END of training, but best_model.zip is saved
  at the point of peak eval reward.  When training collapses after the peak
  (as happens with PPO on eta=100), the end-of-training obs statistics no
  longer match the peak policy, causing poor benchmark performance.

  This script loads best_model.zip, rolls it out deterministically for
  ROLLOUT_STEPS steps to accumulate fresh VecNormalize statistics that are
  consistent with the saved policy, then overwrites obs_rms.pkl.

Models to fix
─────────────
  eta=50  : all three algos are stable (no collapse) — obs_rms.pkl already
             correct, but we recompute as a sanity check if requested.
  eta=100 : PPO collapsed after the peak → must recompute.
            TD3 / SAC have not finished training yet.

Usage (from repo root, on the cluster)
───────────────────────────────────────
    python benchmark/recompute_obs_rms.py --dry-run       # show what would run
    python benchmark/recompute_obs_rms.py                 # fix PPO eta=100 only
    python benchmark/recompute_obs_rms.py --all           # all finished models
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import submitit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRATCH_DIR    = Path("/scratch/epatard")
SCRATCH_ETA100 = SCRATCH_DIR / "eta100"

# ── Job constants ─────────────────────────────────────────────────────────────

ROLLOUT_STEPS = 200_000   # steps per env; enough for obs_rms to converge
N_ENVS        = 4
SEED          = 42
MAX_BID       = 5.0

# ── Targets: (models_root, algo, run_name, horizon_eta) ───────────────────────

_TARGETS: dict[str, tuple[Path, str, str, int]] = {
    "ppo_eta100": (
        SCRATCH_ETA100 / "models",
        "ppo",
        "lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5",
        100,
    ),
    "ppo_eta50": (
        SCRATCH_DIR / "models",
        "ppo",
        "lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5",
        50,
    ),
    "sac_eta50": (
        SCRATCH_DIR / "models",
        "sac",
        "lr=5e-05_bs=512_warmup=500_gamma=0.95",
        50,
    ),
    "td3_eta50": (
        SCRATCH_DIR / "models",
        "td3",
        "lr=0.0001_bs=512_noise=0.1_warmup=500_gamma=0.99",
        50,
    ),
}

# ── SLURM parameters ──────────────────────────────────────────────────────────

_SLURM: dict[str, Any] = {
    "timeout_min":             240,                 # 4 h — rollout only, no training
    "slurm_partition":         "normal-best,gpu-best",
    "gpus_per_node":           0,
    "cpus_per_task":           8,
    "tasks_per_node":          1,
    "nodes":                   1,
    "mem_gb":                  8,
    "slurm_signal_delay_s":    60,
    "slurm_array_parallelism": 4,
    "slurm_job_name":          "dynac_obs_rms",
}

# ── Job function (runs inside each SLURM task) ────────────────────────────────

def recompute_job(models_root: Path, algo: str, run_name: str, horizon_eta: int) -> None:
    """Roll out best_model.zip and rewrite obs_rms.pkl with fresh statistics."""
    import gymnasium as gym
    import dynamicauc as dac
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize

    _ALGO_CLS = {"ppo": PPO, "sac": SAC, "td3": TD3}

    model_dir   = models_root / algo / run_name
    model_path  = model_dir / "best_model"
    obs_rms_out = model_dir / "obs_rms.pkl"

    if not model_path.with_suffix(".zip").exists():
        print(f"[{algo.upper()} eta={horizon_eta}] best_model.zip not found — skipping.")
        return

    print(f"\n[{algo.upper()} eta={horizon_eta}] Rolling out best_model "
          f"for {ROLLOUT_STEPS:,} steps ({ROLLOUT_STEPS * N_ENVS:,} total) …")

    scenario = dac.make("survival-second-price-exponential", horizon_eta=horizon_eta)

    def _make_env():
        env = gym.make("DynamicAuction-v0", scenario=scenario)
        env = dac.MaskExpectedRewardWrapper(env)
        env = dac.FlatAuctionObsWrapper(env)
        env = dac.BidScaleWrapper(env, MAX_BID)
        return Monitor(env)

    # Fresh VecNormalize in training mode — statistics accumulate from rollout
    vec_env = make_vec_env(_make_env, n_envs=N_ENVS, seed=SEED)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=True)

    model = _ALGO_CLS[algo].load(str(model_path), env=vec_env)

    obs = vec_env.reset()
    for step in range(ROLLOUT_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        if step % 50_000 == 0:
            print(f"  step {step:>7,} / {ROLLOUT_STEPS:,}  "
                  f"obs_mean={vec_env.obs_rms.mean.mean():.4f}")

    vec_env.close()

    # Back up old obs_rms and save new one
    old_backup = obs_rms_out.with_stem("obs_rms_old")
    if obs_rms_out.exists() and not old_backup.exists():
        shutil.copy(obs_rms_out, old_backup)
        print(f"  Old obs_rms backed up → {old_backup.name}")

    with open(obs_rms_out, "wb") as f:
        pickle.dump(vec_env.obs_rms, f)
    print(f"  obs_rms.pkl rewritten → {obs_rms_out}")


# ── CLI / SLURM submission ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all",     action="store_true",
                        help="Recompute for all finished models (eta=50 and eta=100).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = list(_TARGETS.keys()) if args.all else ["ppo_eta100"]
    targets = [(key, *_TARGETS[key]) for key in keys]

    print(f"\n[recompute_obs_rms] {len(targets)} target(s)")
    for key, models_root, algo, run_name, eta in targets:
        tag = f"{algo.upper()} eta={eta}"
        model_zip = models_root / algo / run_name / "best_model.zip"
        status = "OK" if model_zip.exists() else "MISSING best_model.zip — will skip"
        if args.dry_run:
            print(f"  [DRY RUN] recompute_job({tag}, {run_name})  [{status}]")
        else:
            print(f"  {tag}  [{status}]")

    if args.dry_run:
        print(f"\nTotal: {len(targets)} jobs (dry run — nothing submitted)")
        return

    executor = submitit.AutoExecutor(
        folder=str(SCRATCH_DIR / "submitit_logs" / "obs_rms" / "%j")
    )
    executor.update_parameters(**_SLURM)

    jobs = executor.map_array(
        recompute_job,
        [models_root for _, models_root, *_ in targets],
        [algo        for _, _, algo, *_      in targets],
        [run_name    for _, _, _, run_name, _ in targets],
        [eta         for _, _, _, _, eta      in targets],
    )
    ids = [j.job_id for j in jobs]
    for (key, _, algo, run_name, eta), jid in zip(targets, ids):
        print(f"  [{algo.upper()} eta={eta}] {run_name}  → job {jid}")
    print(f"\nTotal submitted: {len(ids)} jobs")
    print("Check status   : squeue -u epatard")


if __name__ == "__main__":
    main()
