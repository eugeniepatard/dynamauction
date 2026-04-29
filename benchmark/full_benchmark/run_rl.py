"""
Evaluate PPO, SAC, TD3 models against the FiPla baseline.

Set the model path variables below or pass them on the command line.
Planning policies (Shading, MPC, …) are run separately via run.py.

Usage (from repo root)
----------------------
    python benchmark/full_benchmark/run_rl.py --horizon-eta 50 \\
        --ppo-model /scratch/epatard/models/ppo/best_model
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

import dynamicauc as dac
from benchmark.evaluate import EvalConfig, get_planning_policies
from benchmark.full_benchmark.parallel_evaluate import benchmark
from benchmark.train_rl import load_model, SCENARIO_ID
from dynamicauc.wrappers import SB3PolicyWrapper

# ── Fill these in before submitting ──────────────────────────────────────────
PPO_MODEL_PATH: Optional[Path] = None
SAC_MODEL_PATH: Optional[Path] = None
TD3_MODEL_PATH: Optional[Path] = None
# ─────────────────────────────────────────────────────────────────────────────

_SCRATCH_RESULTS = Path("/scratch/epatard/full_benchmark/results")
_LOCAL_RESULTS   = Path(__file__).resolve().parent / "results"


def _default_out_dir() -> Path:
    return _SCRATCH_RESULTS if Path("/scratch/epatard").exists() else _LOCAL_RESULTS


def _load_rl_policy(algo: str, path: Optional[Path], obs_space) -> Optional[SB3PolicyWrapper]:
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
    n_workers: Optional[int] = None,
    ppo_model: Optional[Path] = None,
    sac_model: Optional[Path] = None,
    td3_model: Optional[Path] = None,
) -> None:
    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    scenario  = dac.make(SCENARIO_ID, horizon_eta=horizon_eta)
    raw_env   = gym.make("DynamicAuction-v0", scenario=scenario)
    obs_space = raw_env.observation_space
    raw_env.close()

    model_paths = {
        "ppo": ppo_model or PPO_MODEL_PATH,
        "sac": sac_model or SAC_MODEL_PATH,
        "td3": td3_model or TD3_MODEL_PATH,
    }

    rl_policies = {}
    for algo, path in model_paths.items():
        policy = _load_rl_policy(algo, path, obs_space)
        if policy is not None:
            rl_policies[algo.upper()] = policy

    if not rl_policies:
        raise RuntimeError(
            "No RL model paths are set. Edit PPO/SAC/TD3_MODEL_PATH at the top "
            "or pass --ppo-model / --sac-model / --td3-model."
        )

    cfg = EvalConfig(
        seeds=list(range(n_seeds)),
        scenario_id=SCENARIO_ID,
        scenario_overrides={"horizon_eta": horizon_eta},
        baselines=["fipla"],
        n_workers=n_workers,
    )

    policies = {}
    policies.update(rl_policies)
    policies.update(get_planning_policies(cfg))

    print(
        f"\n=== eta={horizon_eta} | RL | {len(policies)} policies: "
        f"{list(policies.keys())} ==="
    )
    df = benchmark(policies, cfg)

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv = out / f"{SCENARIO_ID}_eta{horizon_eta}_rl_{ts}.csv"
    df.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--horizon-eta", type=int, required=True,
                        help="Exponential horizon parameter η.")
    parser.add_argument("--n-seeds",   type=int,  default=30)
    parser.add_argument("--out-dir",   type=Path, default=None)
    parser.add_argument("--n-workers", type=int,  default=None,
                        help="Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count).")
    parser.add_argument("--ppo-model", type=Path, default=None,
                        help="Path to PPO best_model (without .zip). Overrides module-level variable.")
    parser.add_argument("--sac-model", type=Path, default=None,
                        help="Path to SAC best_model (without .zip). Overrides module-level variable.")
    parser.add_argument("--td3-model", type=Path, default=None,
                        help="Path to TD3 best_model (without .zip). Overrides module-level variable.")
    args = parser.parse_args()
    run_rl_job(
        horizon_eta=args.horizon_eta,
        n_seeds=args.n_seeds,
        out_dir=args.out_dir,
        n_workers=args.n_workers,
        ppo_model=args.ppo_model,
        sac_model=args.sac_model,
        td3_model=args.td3_model,
    )


if __name__ == "__main__":
    main()
