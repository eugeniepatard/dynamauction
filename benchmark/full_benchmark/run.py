"""
Full benchmark — planning and RL policies.

Job types
---------
  greedy_fipla  — FiPla + GreedyMyopic            (1 job per horizon_eta)
  shading       — Shading × alphas                 (1 job per horizon_eta)
  mpc           — MPC(H)                           (1 job per horizon_eta × mpc_horizon)
  mpc_oracle    — MPCOracle(H)                     (1 job per horizon_eta × mpc_horizon)
  mpc_timebin   — MPCTimeBinned(H, K)              (1 job per horizon_eta × mpc_horizon × n_bins)
  rl            — PPO / SAC / TD3 vs FiPla         (1 job per horizon_eta)

Usage (from repo root)
----------------------
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type greedy_fipla
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type shading
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type mpc --mpc-horizon 7.0
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type mpc_timebin --mpc-horizon 7.0 --n-bins 5
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type rl \\
        --ppo-model /scratch/epatard/models/ppo/best_model \\
        --sac-model /scratch/epatard/models/sac/best_model
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type rl \\
        --algos ppo --ppo-model /scratch/epatard/models/ppo/best_model
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
from benchmark.evaluate import EvalConfig, get_planning_policies, get_simulation_policies
from benchmark.full_benchmark.parallel_evaluate import benchmark
from benchmark.train_rl import load_model, SCENARIO_ID
from dynamicauc.wrappers import SB3PolicyWrapper

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_DEFAULT_MLP_MODEL = (
    _MODELS_DIR
    / "mlp_valuation_head"
    / "survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt"
)

_DEFAULT_PPO_MODEL = _MODELS_DIR / "ppo" / "lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5" / "best_model"
_DEFAULT_SAC_MODEL = _MODELS_DIR / "sac" / "lr=5e-05_bs=512_warmup=500_gamma=0.95"             / "best_model"
_DEFAULT_TD3_MODEL = _MODELS_DIR / "td3" / "lr=0.0001_bs=512_noise=0.1_warmup=500_gamma=0.99"  / "best_model"

MPC_HORIZONS         = [10.0]
SHADING_ALPHAS       = [0.7]
MPC_ALPHA_CANDIDATES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MPC_N_BINS           = [2]

JOB_TYPES = ("greedy_fipla", "shading", "mpc", "mpc_oracle", "mpc_timebin", "rl")

_SCRATCH_RESULTS = Path("/scratch/epatard/full_benchmark/results")
_LOCAL_RESULTS   = Path(__file__).resolve().parent / "results"


def _default_out_dir() -> Path:
    return _SCRATCH_RESULTS if Path("/scratch/epatard").exists() else _LOCAL_RESULTS


def _resolve_mlp(mlp_model: Optional[Path]) -> str:
    path = Path(mlp_model) if mlp_model else _DEFAULT_MLP_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"MLP model not found: {path}\n"
            "Transfer it from your local machine with scp."
        )
    return str(path)


_ALGO_DEFAULTS = {"ppo": _DEFAULT_PPO_MODEL, "sac": _DEFAULT_SAC_MODEL, "td3": _DEFAULT_TD3_MODEL}


def _load_rl_policy(algo: str, path: Optional[Path], obs_space) -> Optional[SB3PolicyWrapper]:
    if path is None:
        path = _ALGO_DEFAULTS.get(algo)
    if path is None:
        return None
    path = Path(path)
    zip_path = Path(str(path) + ".zip")
    if not zip_path.exists():
        print(f"  [{algo.upper()}] model not found at {zip_path} — skipping.")
        return None
    model   = load_model(algo, path=path)
    obs_rms = None
    obs_rms_path = path.parent / "obs_rms.pkl"
    if obs_rms_path.exists():
        with open(obs_rms_path, "rb") as f:
            obs_rms = pickle.load(f)
    print(f"  Loaded {algo.upper()} ← {zip_path}")
    return SB3PolicyWrapper(model, obs_space, obs_rms=obs_rms, exclude_keys=["expected_reward"])


def run_job(
    horizon_eta: int,
    job_type: str = "greedy_fipla",
    # planning options
    mpc_horizon: float = 0.0,
    n_bins: int = 2,
    mlp_model: Optional[Path] = None,
    # RL options
    ppo_model: Optional[Path] = None,
    sac_model: Optional[Path] = None,
    td3_model: Optional[Path] = None,
    algos: Optional[list[str]] = None,
    # common
    n_seeds: int = 30,
    out_dir: Optional[Path] = None,
    n_workers: Optional[int] = None,
) -> None:
    """Run one benchmark job — planning or RL.

    For RL jobs, pass at least one of ppo_model / sac_model / td3_model.
    ``algos`` restricts which RL algorithms are evaluated (default: all with a valid path).
    """
    if job_type not in JOB_TYPES:
        raise ValueError(f"Unknown job_type {job_type!r}. Choose from: {JOB_TYPES}")

    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    base = dict(
        seeds=list(range(n_seeds)),
        scenario_id=SCENARIO_ID,
        scenario_overrides={"horizon_eta": horizon_eta},
        mpc_alpha_candidates=MPC_ALPHA_CANDIDATES,
        n_workers=n_workers,
        policy_seed=0,
    )

    # ── RL job ────────────────────────────────────────────────────────────────
    if job_type == "rl":
        scenario  = dac.make(SCENARIO_ID, horizon_eta=horizon_eta)
        raw_env   = gym.make("DynamicAuction-v0", scenario=scenario)
        obs_space = raw_env.observation_space
        raw_env.close()

        algo_paths = {"ppo": ppo_model, "sac": sac_model, "td3": td3_model}
        if algos:
            algo_paths = {k: v for k, v in algo_paths.items() if k in algos}

        rl_policies = {}
        for algo, path in algo_paths.items():
            policy = _load_rl_policy(algo, path, obs_space)
            if policy is not None:
                rl_policies[algo.upper()] = policy

        if not rl_policies:
            raise RuntimeError(
                "No RL models loaded. Pass at least one of "
                "--ppo-model / --sac-model / --td3-model."
            )

        cfg      = EvalConfig(**base, baselines=[])
        policies = rl_policies
        tag      = "_rl"

    # ── Planning jobs ─────────────────────────────────────────────────────────
    elif job_type == "greedy_fipla":
        cfg      = EvalConfig(**base, baselines=["fipla"], include_greedy=True)
        policies = {**get_planning_policies(cfg), **get_simulation_policies(cfg)}
        tag      = "_greedy_fipla"

    elif job_type == "shading":
        cfg      = EvalConfig(**base, baselines=[], include_shading=True,
                              shading_alphas=SHADING_ALPHAS)
        policies = {**get_planning_policies(cfg), **get_simulation_policies(cfg)}
        tag      = "_shading"

    elif job_type == "mpc":
        cfg      = EvalConfig(**base, baselines=[], include_mpc=True,
                              mpc_model_paths=[_resolve_mlp(mlp_model)],
                              mpc_planning_horizons=[mpc_horizon])
        policies = {**get_planning_policies(cfg), **get_simulation_policies(cfg)}
        tag      = f"_mpc_H{mpc_horizon:g}"

    elif job_type == "mpc_oracle":
        cfg      = EvalConfig(**base, baselines=[], include_mpc_oracle=True,
                              mpc_model_paths=[_resolve_mlp(mlp_model)],
                              mpc_planning_horizons=[mpc_horizon])
        policies = {**get_planning_policies(cfg), **get_simulation_policies(cfg)}
        tag      = f"_mpcoracle_H{mpc_horizon:g}"

    else:  # mpc_timebin
        cfg      = EvalConfig(**base, baselines=[], include_mpc_timebin=True,
                              mpc_model_paths=[_resolve_mlp(mlp_model)],
                              mpc_planning_horizons=[mpc_horizon],
                              mpc_n_bins=[n_bins])
        policies = {**get_planning_policies(cfg), **get_simulation_policies(cfg)}
        tag      = f"_mpctimebin_H{mpc_horizon:g}_K{n_bins}"

    print(
        f"\n=== eta={horizon_eta} | type={job_type} | "
        f"{len(policies)} policies: {list(policies.keys())} ==="
    )
    df  = benchmark(policies, cfg)
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv = out / f"{SCENARIO_ID}_eta{horizon_eta}{tag}_{ts}.csv"
    df.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")


def run_rl_job(
    horizon_eta: int,
    n_seeds: int = 30,
    out_dir: Optional[Path] = None,
    n_workers: Optional[int] = None,
    ppo_model: Optional[Path] = None,
    sac_model: Optional[Path] = None,
    td3_model: Optional[Path] = None,
) -> None:
    """Backward-compatible wrapper used by submit.py."""
    run_job(
        horizon_eta=horizon_eta,
        job_type="rl",
        n_seeds=n_seeds,
        out_dir=out_dir,
        n_workers=n_workers,
        ppo_model=ppo_model,
        sac_model=sac_model,
        td3_model=td3_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--horizon-eta", type=int, required=True,
                        help="Exponential horizon parameter η.")
    parser.add_argument(
        "--job-type", choices=JOB_TYPES, default="greedy_fipla",
        help="Policy group to run.",
    )

    # Planning options
    g_plan = parser.add_argument_group("planning options")
    g_plan.add_argument("--mpc-horizon", type=float, default=0.0,
                        help="MPC planning horizon H (mpc / mpc_oracle / mpc_timebin).")
    g_plan.add_argument("--n-bins",      type=int,   default=2,
                        help="Number of time bins K (mpc_timebin).")
    g_plan.add_argument("--mlp-model",   type=Path,  default=None,
                        help="Path to the MLPValuationHead .pt file (default: auto-resolved).")

    # RL options
    g_rl = parser.add_argument_group("RL options (job-type=rl)")
    g_rl.add_argument("--algos",     nargs="+", choices=["ppo", "sac", "td3"],
                      default=None,
                      help="RL algorithms to evaluate (default: all with a provided model path).")
    g_rl.add_argument("--ppo-model", type=Path, default=None,
                      help=f"Path to PPO best_model (without .zip). Default: {_DEFAULT_PPO_MODEL}")
    g_rl.add_argument("--sac-model", type=Path, default=None,
                      help=f"Path to SAC best_model (without .zip). Default: {_DEFAULT_SAC_MODEL}")
    g_rl.add_argument("--td3-model", type=Path, default=None,
                      help=f"Path to TD3 best_model (without .zip). Default: {_DEFAULT_TD3_MODEL}")

    # Common options
    g_common = parser.add_argument_group("common options")
    g_common.add_argument("--n-seeds",   type=int,  default=30)
    g_common.add_argument("--out-dir",   type=Path, default=None,
                          help="Directory for CSV results (default: scratch on cluster, local otherwise).")
    g_common.add_argument("--n-workers", type=int,  default=None,
                          help="Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count).")

    args = parser.parse_args()
    run_job(
        horizon_eta=args.horizon_eta,
        job_type=args.job_type,
        mpc_horizon=args.mpc_horizon,
        n_bins=args.n_bins,
        mlp_model=args.mlp_model,
        ppo_model=args.ppo_model,
        sac_model=args.sac_model,
        td3_model=args.td3_model,
        algos=args.algos,
        n_seeds=args.n_seeds,
        out_dir=args.out_dir,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
