"""
Full benchmark (planning policies) — one policy group per invocation.

Job types
---------
  greedy_fipla  — FiPla + GreedyMyopic            (1 job per horizon_eta)
  shading       — Shading × 4 alphas               (1 job per horizon_eta)
  mpc           — MPC(H) + MPCOracle(H)            (1 job per horizon_eta × mpc_horizon)
  mpc_timebin   — MPCTimeBinned(H, K)              (1 job per horizon_eta × mpc_horizon × n_bins)

Usage (from repo root)
----------------------
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type greedy_fipla
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type shading
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type mpc --mpc-horizon 7.0
    python benchmark/full_benchmark/run.py --horizon-eta 50 --job-type mpc_timebin --mpc-horizon 7.0 --n-bins 5
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.evaluate import EvalConfig, get_planning_policies, get_simulation_policies
from benchmark.full_benchmark.parallel_evaluate import benchmark

SCENARIO_ID = "survival-second-price-exponential"

_DEFAULT_MLP_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "mlp_valuation_head"
    / "survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt"
)

MPC_HORIZONS         = [0.0, 4.0, 7.0, 10.0]
SHADING_ALPHAS       = [0.2, 0.4, 0.6, 0.8]
MPC_ALPHA_CANDIDATES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
MPC_N_BINS           = [2, 5, 10]

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


def run_job(
    horizon_eta: int,
    job_type: str = "greedy_fipla",
    mpc_horizon: float = 0.0,
    n_bins: int = 2,
    n_seeds: int = 30,
    out_dir: Optional[Path] = None,
    mlp_model: Optional[Path] = None,
    n_workers: Optional[int] = None,
) -> None:
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

    if job_type == "greedy_fipla":
        cfg = EvalConfig(
            **base,
            baselines=["fipla"],
            include_greedy=True,
        )
    elif job_type == "shading":
        cfg = EvalConfig(
            **base,
            baselines=[],
            include_shading=True,
            shading_alphas=SHADING_ALPHAS,
        )
    elif job_type == "mpc":
        cfg = EvalConfig(
            **base,
            baselines=[],
            include_mpc=True,
            include_mpc_oracle=True,
            mpc_model_paths=[_resolve_mlp(mlp_model)],
            mpc_planning_horizons=[mpc_horizon],
        )
    else:  # mpc_timebin
        cfg = EvalConfig(
            **base,
            baselines=[],
            include_mpc_timebin=True,
            mpc_model_paths=[_resolve_mlp(mlp_model)],
            mpc_planning_horizons=[mpc_horizon],
            mpc_n_bins=[n_bins],
        )

    policies = {}
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg))

    print(
        f"\n=== eta={horizon_eta} | type={job_type} | H={mpc_horizon:g} | K={n_bins} | "
        f"{len(policies)} policies: {list(policies.keys())} ==="
    )
    df = benchmark(policies, cfg)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if job_type == "mpc":
        tag = f"_mpc_H{mpc_horizon:g}"
    elif job_type == "mpc_timebin":
        tag = f"_mpctimebin_H{mpc_horizon:g}_K{n_bins}"
    elif job_type == "shading":
        tag = "_shading"
    else:
        tag = "_greedy_fipla"
    csv = out / f"{SCENARIO_ID}_eta{horizon_eta}{tag}_{ts}.csv"
    df.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--horizon-eta", type=int, required=True,
                        help="Exponential horizon parameter η.")
    parser.add_argument(
        "--job-type", choices=JOB_TYPES, default="greedy_fipla",
        help="Policy group to run in this job.",
    )
    parser.add_argument(
        "--mpc-horizon", type=float, default=0.0,
        help="MPC planning horizon H (used for mpc and mpc_timebin job types).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=2,
        help="Number of time bins K (used for mpc_timebin job type).",
    )
    parser.add_argument("--n-seeds",   type=int,  default=30)
    parser.add_argument("--out-dir",   type=Path, default=None,
                        help="Directory for CSV results (default: scratch on cluster, local otherwise).")
    parser.add_argument("--mlp-model", type=Path, default=None,
                        help="Path to the MLPValuationHead .pt file (default: auto-resolved).")
    parser.add_argument("--n-workers", type=int,  default=None,
                        help="Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count). Use 1 for sequential.")
    args = parser.parse_args()
    run_job(
        horizon_eta=args.horizon_eta,
        job_type=args.job_type,
        mpc_horizon=args.mpc_horizon,
        n_bins=args.n_bins,
        n_seeds=args.n_seeds,
        out_dir=args.out_dir,
        mlp_model=args.mlp_model,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
