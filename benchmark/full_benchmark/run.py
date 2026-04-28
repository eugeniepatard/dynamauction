"""
Full benchmark: FiPla, GreedyMyopic, Shading×4, MPC×6, MPCOracle×6, MPCTimeBinned×24
for a given horizon_eta scenario parameter.

Usage (from repo root)
----------------------
    # local run
    python benchmark/full_benchmark/run.py --horizon-eta 50
    python benchmark/full_benchmark/run.py --horizon-eta 100

    # specify a custom results directory
    python benchmark/full_benchmark/run.py --horizon-eta 50 --out-dir /scratch/epatard/full_benchmark/results

    # use a different MLP model
    python benchmark/full_benchmark/run.py --horizon-eta 50 --mlp-model /path/to/model.pt

   
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

MPC_HORIZONS         = [0.0, 4.0, 7.0, 10.0, 15.0, 25.0]
SHADING_ALPHAS       = [0.2, 0.4, 0.6, 0.8]
MPC_ALPHA_CANDIDATES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
MPC_N_BINS           = [2, 3, 5, 10]

_SCRATCH_RESULTS = Path("/scratch/epatard/full_benchmark/results")
_LOCAL_RESULTS   = Path(__file__).resolve().parent / "results"


def _default_out_dir() -> Path:
    return _SCRATCH_RESULTS if Path("/scratch/epatard").exists() else _LOCAL_RESULTS


def run_job(
    horizon_eta: int,
    n_seeds: int = 30,
    out_dir: Optional[Path] = None,
    mlp_model: Optional[Path] = None,
    n_workers: Optional[int] = None,
) -> None:
    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    model_path = Path(mlp_model) if mlp_model else _DEFAULT_MLP_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"MLP model not found: {model_path}\n"
            "Transfer it from your local machine with scp (see module docstring)."
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
        n_workers=n_workers,
        policy_seed=0,
    )

    policies = {}
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg))

    print(
        f"\n=== horizon_eta={horizon_eta} | {len(policies)} policies: "
        f"{list(policies.keys())} ==="
    )
    df = benchmark(policies, cfg)

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv = out / f"{SCENARIO_ID}_eta{horizon_eta}_{ts}.csv"
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
    parser.add_argument("--out-dir",   type=Path, default=None,
                        help="Directory for CSV results (default: scratch on cluster, local otherwise).")
    parser.add_argument("--mlp-model", type=Path, default=None,
                        help="Path to the MLPValuationHead .pt file (default: auto-resolved).")
    parser.add_argument("--n-workers", type=int,  default=None,
                        help="Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count). Use 1 for sequential.")
    args = parser.parse_args()
    run_job(horizon_eta=args.horizon_eta, n_seeds=args.n_seeds, out_dir=args.out_dir, mlp_model=args.mlp_model, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
