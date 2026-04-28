"""
Parallel variant of benchmark.evaluate for multi-core machines and SLURM clusters.

Imports all shared code from benchmark.evaluate and overrides benchmark() with a
ProcessPoolExecutor implementation that dispatches (policy, seed) tasks across
workers.

Usage
-----
    from benchmark.full_benchmark.parallel_evaluate import benchmark, EvalConfig
    df = benchmark(policies, cfg)

Worker count resolution (in order):
  1. cfg.n_workers if set explicitly
  2. SLURM_CPUS_PER_TASK environment variable (set automatically on the cluster)
  3. os.cpu_count()
"""
from __future__ import annotations

import copy
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import gymnasium as gym
import pandas as pd
from tqdm import tqdm

import dynamicauc as dac
from dynamicauc.baseline.policies import BiddingPolicy

# Re-export everything from the sequential module so callers can import from
# either evaluate or parallel_evaluate interchangeably.
from benchmark.evaluate import (
    EvalConfig,
    RESULTS_DIR,
    AVAILABLE_BASELINES,
    _run_episode,
    _make_baseline,
    _mpc_label,
    evaluate_policy,
    get_planning_policies,
    get_simulation_policies,
    get_rl_policies,
    run_full_benchmark,
    episode_summary,
    score_matrix,
)

__all__ = [
    "EvalConfig",
    "RESULTS_DIR",
    "AVAILABLE_BASELINES",
    "evaluate_policy",
    "get_planning_policies",
    "get_simulation_policies",
    "get_rl_policies",
    "run_full_benchmark",
    "episode_summary",
    "score_matrix",
    "benchmark",
]

# --------------------------------------------------------------------------- #
# Worker state (module-level globals, populated by _init_worker in each process)
# --------------------------------------------------------------------------- #

_WORKER_POLICIES: Dict[str, BiddingPolicy] = {}
_WORKER_SCENARIO = None


def _default_n_workers() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm) if slurm else (os.cpu_count() or 1)


def _init_worker(policies_pkl: bytes, scenario_id: str, scenario_overrides: dict) -> None:
    global _WORKER_POLICIES, _WORKER_SCENARIO
    _WORKER_POLICIES = pickle.loads(policies_pkl)
    _WORKER_SCENARIO = dac.make(scenario_id, **scenario_overrides)


def _run_task(policy_name: str, seed: int) -> pd.DataFrame:
    policy = copy.deepcopy(_WORKER_POLICIES[policy_name])
    env    = gym.make("DynamicAuction-v0", scenario=_WORKER_SCENARIO)
    df     = _run_episode(policy, env, seed, policy_name)
    env.close()
    return df


# --------------------------------------------------------------------------- #
# Parallel benchmark
# --------------------------------------------------------------------------- #

def benchmark(
    policies: Dict[str, BiddingPolicy],
    cfg:      EvalConfig = EvalConfig(),
) -> pd.DataFrame:
    """
    Run all (policy, seed) combinations, dispatching to worker processes.

    Falls back to the sequential path when n_workers == 1 to avoid
    ProcessPoolExecutor overhead on single-core machines.
    """
    if not policies:
        raise ValueError("No policies to benchmark. Check your --algos, --baselines, and --sim-policies flags.")

    n_workers = cfg.n_workers if cfg.n_workers is not None else _default_n_workers()

    if n_workers <= 1:
        from benchmark.evaluate import benchmark as _sequential
        return _sequential(policies, cfg)

    n_tasks = len(policies) * len(cfg.seeds)
    print(
        f"\nBenchmarking {len(policies)} policies × {len(cfg.seeds)} seeds"
        f" = {n_tasks} tasks, {n_workers} workers"
    )

    policies_pkl = pickle.dumps(policies)
    tasks = [(name, seed) for name in policies for seed in cfg.seeds]

    frames: List[pd.DataFrame] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(policies_pkl, cfg.scenario_id, cfg.scenario_overrides),
    ) as executor:
        futures = {executor.submit(_run_task, name, seed): (name, seed) for name, seed in tasks}
        for fut in tqdm(as_completed(futures), total=n_tasks, desc="Tasks completed"):
            frames.append(fut.result())

    return pd.concat(frames, ignore_index=True)
