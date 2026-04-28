"""
Evaluate the best PPO, SAC, TD3 models against all planning baselines.

Fill in the three model path variables below before running.

Usage (from repo root)
----------------------
    python benchmark/full_benchmark/run_rl.py --horizon-eta 50
    python benchmark/full_benchmark/run_rl.py --horizon-eta 100
"""
from __future__ import annotations

import sys
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gymnasium as gym

from benchmark.evaluate import EvalConfig, get_planning_policies, get_simulation_policies
from benchmark.full_benchmark.parallel_evaluate import benchmark
from benchmark.train_rl import load_model, SCENARIO_ID
from dynamicauc.wrappers import SB3PolicyWrapper

# ── Fill these in before submitting ──────────────────────────────────────────
# Paths to best_model (without .zip extension).
# obs_rms.pkl is expected in the same directory as best_model.zip.
# Example: Path("/scratch/epatard/models/ppo/lr=3e-4_steps=1024/best_model")

PPO_MODEL_PATH: Optional[Path] = None  # TODO: set your PPO best_model path
SAC_MODEL_PATH: Optional[Path] = None  # TODO: set your SAC best_model path
TD3_MODEL_PATH: Optional[Path] = None  # TODO: set your TD3 best_model path

# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MLP_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "mlp_valuation_head"
    / "survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt"
)

MPC_HORIZONS         = [0.0, 7.0, 10.0, 15.0, 25.0]
SHADING_ALPHAS       = [0.2, 0.4, 0.6, 0.8]
MPC_ALPHA_CANDIDATES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
MPC_N_BINS           = [2, 3, 5, 10]

_SCRATCH_RESULTS = Path("/scratch/epatard/full_benchmark/results_rl")
_LOCAL_RESULTS   = Path(__file__).resolve().parent / "results_rl"

_MODEL_PATHS = {"ppo": PPO_MODEL_PATH, "sac": SAC_MODEL_PATH, "td3": TD3_MODEL_PATH}


def _default_out_dir() -> Path:
    return _SCRATCH_RESULTS if Path("/scratch/epatard").exists() else _LOCAL_RESULTS


def _load_rl_policy(algo: str, path: Optional[Path], obs_space) -> Optional[SB3PolicyWrapper]:
    """Load a model from *path* (or skip if None / file missing)."""
    if path is None:
        print(f"  [{algo.upper()}] path not set — skipping.")
        return None
    path = Path(path)
    zip_path = Path(str(path) + ".zip")
    if not zip_path.exists():
        print(f"  [{algo.upper()}] model not found at {zip_path} — skipping.")
        return None

    model = load_model(algo, path=path)

    obs_rms = None
    obs_rms_path = path.parent / "obs_rms.pkl"
    if obs_rms_path.exists():
        with open(obs_rms_path, "rb") as f:
            obs_rms = pickle.load(f)

    print(f"  Loaded {algo.upper()} ← {zip_path}")
    return SB3PolicyWrapper(model, obs_space, obs_rms=obs_rms, exclude_keys=["expected_reward"])


def run_rl_job(
    horizon_eta: int,
    n_seeds: int = 30,
    out_dir: Optional[Path] = None,
    mlp_model: Optional[Path] = None,
    n_workers: Optional[int] = None,
) -> None:
    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    import dynamicauc as dac
    scenario = dac.make(SCENARIO_ID, horizon_eta=horizon_eta)
    raw_env  = gym.make("DynamicAuction-v0", scenario=scenario)
    obs_space = raw_env.observation_space
    raw_env.close()

    rl_policies = {}
    for algo, path in _MODEL_PATHS.items():
        policy = _load_rl_policy(algo, path, obs_space)
        if policy is not None:
            rl_policies[algo.upper()] = policy

    if not rl_policies:
        raise RuntimeError(
            "No RL model paths are set. Edit the PPO/SAC/TD3_MODEL_PATH variables at the top of run_rl.py."
        )

    model_path = Path(mlp_model) if mlp_model else _DEFAULT_MLP_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"MLP model not found: {model_path}\n"
            "Transfer it from your local machine with scp (see module docstring in run.py)."
        )

    cfg = EvalConfig(
        seeds=list(range(n_seeds)),
        scenario_id=SCENARIO_ID,
        scenario_overrides={"horizon_eta": horizon_eta},
        baselines=["fipla"],
        include_greedy=True,
        include_shading=True,
        shading_alphas=SHADING_ALPHAS,
        include_mpc=True,
        include_mpc_oracle=True,
        include_mpc_timebin=True,
        mpc_model_paths=[str(model_path)],
        mpc_planning_horizons=MPC_HORIZONS,
        mpc_alpha_candidates=MPC_ALPHA_CANDIDATES,
        mpc_n_bins=MPC_N_BINS,
        include_mpc_delta=True,
        n_workers=n_workers,
        policy_seed=0,
    )

    policies = {}
    policies.update(rl_policies)
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg))

    print(
        f"\n=== horizon_eta={horizon_eta} | {len(policies)} policies: "
        f"{list(policies.keys())} ==="
    )
    df = benchmark(policies, cfg)

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv = out / f"{SCENARIO_ID}_rl_eta{horizon_eta}_{ts}.csv"
    df.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--horizon-eta", type=int, required=True,
        help="Exponential horizon parameter η.",
    )
    parser.add_argument("--n-seeds",   type=int,  default=30)
    parser.add_argument("--out-dir",   type=Path, default=None)
    parser.add_argument("--mlp-model", type=Path, default=None)
    parser.add_argument("--policy-seed",    type=int, default=0, help="Seed for any policy components that require randomness (e.g. MPC rollouts).")
    parser.add_argument("--n-workers", type=int,  default=None,
                        help="Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count). Use 1 for sequential.")
    args = parser.parse_args()
    run_rl_job(args.horizon_eta, args.n_seeds, args.out_dir, args.mlp_model, args.n_workers)


if __name__ == "__main__":
    main()
