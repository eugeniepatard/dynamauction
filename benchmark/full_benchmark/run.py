"""
Full benchmark: FiPla, GreedyMyopic, Shading×4, MPC×6, MPCOracle×6, MPCTimeBinned×24
for a given horizon_eta scenario parameter.

Parameters are loaded from benchmark/conf/eval.yaml (override with --config).
RL models (PPO, SAC, TD3) are optional: set their paths in eval.yaml under
rl_models, or pass --ppo-model / --sac-model / --td3-model on the command line.
When no RL paths are provided the script evaluates planning baselines only.

Usage (from repo root)
----------------------
    # planning baselines only (rl_models set to null in eval.yaml)
    python benchmark/full_benchmark/run.py --horizon-eta 50
    python benchmark/full_benchmark/run.py --horizon-eta 100

    # RL + planning (set rl_models paths in eval.yaml, or pass CLI flags)
    python benchmark/full_benchmark/run.py --horizon-eta 50 \\
        --ppo-model /scratch/epatard/models/ppo/best_model

    # custom results directory
    python benchmark/full_benchmark/run.py --horizon-eta 50 --out-dir /scratch/.../results

    # use a different MLP model
    python benchmark/full_benchmark/run.py --horizon-eta 50 --mlp-model /path/to/model.pt

    # use a different config file
    python benchmark/full_benchmark/run.py --horizon-eta 50 --config /path/to/eval.yaml
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dynamauction as dac

from benchmark.evaluate import (
    get_planning_policies,
    get_simulation_policies,
    get_rl_policies,
    load_eval_conf,
    cfg_from_yaml,
)
from benchmark.full_benchmark.parallel_evaluate import benchmark
from benchmark.train_rl import SCENARIO_ID

_DEFAULT_MLP_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "mlp_valuation_head"
    / "survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt"
)

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
    cfg_path: Optional[Path] = None,
    ppo_model: Optional[Path] = None,
    sac_model: Optional[Path] = None,
    td3_model: Optional[Path] = None,
) -> None:
    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    conf = load_eval_conf(cfg_path)
    cfg  = cfg_from_yaml(conf)

    cfg.seeds                    = list(range(n_seeds))
    cfg.scenario_overrides       = {"horizon_eta": horizon_eta}
    cfg.n_workers                = n_workers

    # CLI model path overrides (take priority over YAML)
    for algo, path in [("ppo", ppo_model), ("sac", sac_model), ("td3", td3_model)]:
        if path is not None:
            cfg.rl_model_paths[algo] = str(path)

    if mlp_model is not None:
        cfg.mpc_model_paths = [str(mlp_model)]
    elif cfg.mpc_model_paths == [None]:
        model_path = _DEFAULT_MLP_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"MLP model not found: {model_path}\n"
                "Transfer it from your local machine with scp (see module docstring)."
            )
        cfg.mpc_model_paths = [str(model_path)]

    scenario = dac.make(SCENARIO_ID, horizon_eta=horizon_eta)

    policies = {}
    policies.update(get_rl_policies(scenario, cfg.rl_model_paths))
    policies.update(get_planning_policies(cfg))
    policies.update(get_simulation_policies(cfg))

    print(
        f"\n=== horizon_eta={horizon_eta} | {len(policies)} policies: "
        f"{list(policies.keys())} ==="
    )
    df = benchmark(policies, cfg)

    ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "_rl" if cfg.rl_model_paths else ""
    csv    = out / f"{SCENARIO_ID}_eta{horizon_eta}{suffix}_{ts}.csv"
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
    parser.add_argument("--config",    type=Path, default=None,
                        help="Path to eval YAML config (default: benchmark/conf/eval.yaml).")
    parser.add_argument("--ppo-model", type=Path, default=None,
                        help="Path to PPO best_model (without .zip). Overrides eval.yaml.")
    parser.add_argument("--sac-model", type=Path, default=None,
                        help="Path to SAC best_model (without .zip). Overrides eval.yaml.")
    parser.add_argument("--td3-model", type=Path, default=None,
                        help="Path to TD3 best_model (without .zip). Overrides eval.yaml.")
    args = parser.parse_args()
    run_job(
        horizon_eta=args.horizon_eta,
        n_seeds=args.n_seeds,
        out_dir=args.out_dir,
        mlp_model=args.mlp_model,
        n_workers=args.n_workers,
        cfg_path=args.config,
        ppo_model=args.ppo_model,
        sac_model=args.sac_model,
        td3_model=args.td3_model,
    )


if __name__ == "__main__":
    main()
