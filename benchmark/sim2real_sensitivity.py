"""
Sim-to-real sensitivity analysis for MPC.

Tests how much MPC performance degrades when its internal simulator uses
slightly-to-very-different parameters than the real world it bids on.

Architecture
------------
MPC is configured with a FIXED nominal world (its "model of the world").
The REAL evaluation environment is varied across a parameter grid.
MPC never adapts — it always plans with the same sim scenario.

MismatchedMPCTimeBinnedPolicy.set_env(real_env) does NOT give MPC access
to the real env's dynamics.  Instead, at each bid() call, the policy
transplants only the current real observable state (t, step_idx, ctx,
pending_conversions, …) into its own separate sim env, so rollouts start
from the right position but use the sim scenario's competition distribution,
horizon generator, and arrival process.

Two mismatch types (MPC's sim is always LogNormal(mu=0, σ=0.5) / Exp(η=50)):
  competition  — real world has bids ~ LogNormal(mu_real, 0.5),
                 mu_real ∈ {-2.0, -1.5, -1.0, -0.5, 0.0*, 0.5, 1.0, 1.5}
                 (mu=0 is the true scenario default; negative → weak competition,
                 positive → fierce competition)
  horizon      — real world has campaign length ~ Exp(eta_real),
                 eta_real  ∈ {10, 20, 35, 50*, 75, 100, 200}

  (* = no mismatch; sim and real are the same)

Connection to the n_rollouts finding
--------------------------------------
Since reward is flat across n_rollouts=1..50, V_0 (the MLP valuation head)
carries most of the signal.  This experiment asks whether the rollout
dynamics matter at all: does sim-to-real mismatch hurt when the rollouts
barely contribute anyway?

Usage (SLURM job array):
  sbatch benchmark/run_scripts/run_sim2real.sh

Local test:
  TRIAL_ID=4 python benchmark/sim2real_sensitivity.py
"""
from __future__ import annotations

import copy
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dynamauction as dac
import gymnasium as gym
from benchmark.evaluate import _DEFAULT_MLP
from benchmark.train_mlp import load_mlp
from dynamauction.baseline.baseline_algo import (
    MPCTimeBinnedPolicy,
    ShadingPolicy,
    SimulationEstimator,
)
from dynamauction.baseline.policies import FiPlaPolicy, ConstantPolicy, BiddingPolicy

# ── Experiment grid ──────────────────────────────────────────────────────────

# MPC's sim world is fixed at the nominal scenario defaults:
SIM_COMPETITION_MU    = 0.0    # MPC always assumes LogNormal(mu=0, sigma=0.5)
SIM_COMPETITION_SIGMA = 0.5
SIM_HORIZON_ETA       = 50     # MPC always assumes Exp(50)

SCENARIO_ID = "survival-second-price-exponential"

# Real world VARIES:
REAL_COMPETITION_MUS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]   # 8 values
REAL_HORIZON_ETAS    = [10,   20,   35,   50,   75,  100, 200]            # 7 values

CONFIGS: List[Dict[str, Any]] = (
    [{"type": "competition", "value": mu} for mu in REAL_COMPETITION_MUS]
    + [{"type": "horizon",   "value": e}  for e  in REAL_HORIZON_ETAS]
)   # 15 total; SLURM --array=0-14

N_SEEDS           = 100
N_ROLLOUTS        = 10
PLANNING_HORIZON  = 10.0
N_BINS            = 2
ALPHA_CANDIDATES  = [0.6, 0.8, 1.0, 1.2, 1.5]

OUT_DIR = Path(__file__).parent / "benchmark_results" / "sim2real"


# ── Mismatched MPC policy ─────────────────────────────────────────────────────

class MismatchedMPCTimeBinnedPolicy(MPCTimeBinnedPolicy):
    """
    MPCTimeBinnedPolicy that plans in its own separate sim env with (potentially)
    mismatched dynamics.

    The policy has NO access to the real env's scenario.  At each bid() call,
    it transplants the current real observable state into the sim env so rollouts
    start from the correct position — but competition, horizon, and arrival come
    entirely from the sim scenario (fixed at construction).

    This cleanly tests sim-to-real mismatch: a policy configured for world A
    deployed in world B.
    """

    def __init__(self, v0, sim_scenario, **kwargs) -> None:
        super().__init__(v0, **kwargs)
        self._sim_env = gym.make("DynamAuction-v0", scenario=sim_scenario)
        self._sim_env.reset(seed=0)   # initialise internal state; will be overwritten
        self._real_env: Optional[gym.Env] = None
        self._env = self._sim_env   # parent's bid() deepcopies self._env → sim dynamics

    def set_env(self, real_env: gym.Env) -> None:
        """Store the real env reference for state transplanting; never expose it to rollouts."""
        self._real_env = real_env
        # Do NOT call super().set_env() — self._env stays pointing to _sim_env.

    def bid(self, observations: Dict[str, Any]) -> float:
        if self._real_env is None:
            raise RuntimeError("Call set_env(real_env) before bid().")
        self._transplant_state()
        return super().bid(observations)   # uses self._env = _sim_env with real state

    def _transplant_state(self) -> None:
        """Copy current real observable state into the sim env.

        After transplant, a deepcopy of sim_env has:
          - real t, step_idx, ctx, display_features, y_func (current position)
          - real pending_conversions, t_past_won_auctions (episode history)
          - sim scenario's competition, horizon_generator, arrival (mismatched dynamics)

        _resample_remaining_horizon() will then resample the future horizon and
        future auction schedule from the sim scenario's generators.
        """
        real = self._real_env.unwrapped
        sim  = self._sim_env.unwrapped

        sim.state                     = copy.deepcopy(real.state)
        sim.auction_times             = real.auction_times.copy()
        sim.pending_conversions       = list(real.pending_conversions)
        sim.matured_conversions_reward = real.matured_conversions_reward
        sim.matured_conversions_times = list(real.matured_conversions_times)
        sim.t_past_won_auctions       = list(real.t_past_won_auctions)
        sim.horizon                   = real.horizon
        if real.user_features is not None:
            sim.user_features = real.user_features.copy()


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(
    env: gym.Env,
    policy: BiddingPolicy,
    seed: int,
) -> List[Dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)

    rows: List[Dict[str, Any]] = []
    cum_reward = 0.0

    while True:
        bid = policy.bid(obs)
        obs, reward, done, _trunc, info = env.step(bid)
        cum_reward += float(reward)

        rows.append({
            "seed":          seed,
            "step":          int(info["auction"]["step"]),
            "t":             float(info["auction"]["time"]),
            "reward":        float(reward),
            "cum_reward":    cum_reward,
            "bid":           float(bid),
            "won":           int(bool(info["auction"]["won"])),
            "competing_bid": float(info["auction"]["competing_bid"]),
        })

        if done:
            break

    return rows


# ── Per-trial runner ──────────────────────────────────────────────────────────

def run_config(trial_id: int, scratch_dir: Optional[Path] = None) -> Path:
    cfg = CONFIGS[trial_id]
    mtype, mvalue = cfg["type"], cfg["value"]

    # ── Real eval environment (VARIES across trials) ──────────────────────────
    if mtype == "competition":
        real_scenario = dac.make(
            SCENARIO_ID,
            horizon_eta=SIM_HORIZON_ETA,
            competition_mu=float(mvalue),
            competition_sigma=SIM_COMPETITION_SIGMA,
        )
    else:
        real_scenario = dac.make(
            SCENARIO_ID,
            horizon_eta=float(mvalue),
            competition_mu=SIM_COMPETITION_MU,
            competition_sigma=SIM_COMPETITION_SIGMA,
        )
    real_env = gym.make("DynamAuction-v0", scenario=real_scenario)

    # ── MPC sim world (FIXED at nominal parameters for all trials) ────────────
    sim_scenario = dac.make(
        SCENARIO_ID,
        horizon_eta=SIM_HORIZON_ETA,
        competition_mu=SIM_COMPETITION_MU,
        competition_sigma=SIM_COMPETITION_SIGMA,
    )

    # ── Load MLP and build policies ───────────────────────────────────────────
    mlp = load_mlp(SCENARIO_ID, str(_DEFAULT_MLP))

    mpc_binned_policy = MismatchedMPCTimeBinnedPolicy(
        v0=mlp,
        sim_scenario=sim_scenario,
        alpha_candidates=ALPHA_CANDIDATES,
        n_rollouts=N_ROLLOUTS,
        planning_horizon=PLANNING_HORIZON,
        n_bins=N_BINS,
    )
    # K=1 is equivalent to standard MPC (single alpha for the full horizon)
    mpc_policy = MismatchedMPCTimeBinnedPolicy(
        v0=mlp,
        sim_scenario=sim_scenario,
        alpha_candidates=ALPHA_CANDIDATES,
        n_rollouts=N_ROLLOUTS,
        planning_horizon=PLANNING_HORIZON,
        n_bins=1,
    )

    # Reference policies (oracle access to expected_reward / correct real env dynamics)
    fipla_policy   = FiPlaPolicy()  # uses observations["expected_reward"] directly
    shading_policy = ShadingPolicy(
        SimulationEstimator(
            pi0=ConstantPolicy(0.0), n_rollouts=N_ROLLOUTS, seed=0
        ),
        alpha=0.7,
    )

    policies = {
        f"MPCTimeBinned(H={int(PLANNING_HORIZON)},K={N_BINS})[mismatch]": mpc_binned_policy,
        f"MPC(H={int(PLANNING_HORIZON)})[mismatch]":                       mpc_policy,
        "FiPla":           fipla_policy,
        "Shading(α=0.7)":  shading_policy,
    }

    all_rows: List[Dict[str, Any]] = []
    nom_label = (
        f"LogNormal(μ={SIM_COMPETITION_MU},σ={SIM_COMPETITION_SIGMA})"
        if mtype == "competition"
        else f"Exp({SIM_HORIZON_ETA})"
    )
    real_label = (
        f"LogNormal(μ={mvalue},σ={SIM_COMPETITION_SIGMA})"
        if mtype == "competition"
        else f"Exp({mvalue})"
    )

    for pol_name, pol in policies.items():
        print(
            f"  [{trial_id:02d}] mtype={mtype}  sim={nom_label}  real={real_label}"
            f"  policy={pol_name}  seeds=0..{N_SEEDS - 1}",
            flush=True,
        )
        for seed in range(N_SEEDS):
            rows = _run_episode(real_env, pol, seed)
            for r in rows:
                r["policy"]            = pol_name
                r["mismatch_type"]     = mtype
                r["mismatch_value"]    = float(mvalue)
                # Signed absolute mismatch: 0 = no mismatch
                if mtype == "competition":
                    r["mismatch_intensity"] = float(mvalue) - SIM_COMPETITION_MU
                else:
                    r["mismatch_intensity"] = (
                        (float(mvalue) - SIM_HORIZON_ETA) / SIM_HORIZON_ETA
                    )
            all_rows.extend(rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(scratch_dir) / "sim2real" if scratch_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = out_dir / f"sim2real_{mtype}_{mvalue}_{ts}.csv"

    fieldnames = [
        "policy", "mismatch_type", "mismatch_value", "mismatch_intensity",
        "seed", "step", "t", "reward", "cum_reward", "bid", "won", "competing_bid",
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  → {out_file}  ({len(all_rows):,} rows)", flush=True)
    return out_file


def main() -> None:
    trial_id = int(os.environ.get("SLURM_ARRAY_TASK_ID",
                                  sys.argv[1] if len(sys.argv) > 1 else "0"))
    scratch_dir_env = os.environ.get("SCRATCH_DIR")
    scratch_dir = Path(scratch_dir_env) / "dynamicauc" if scratch_dir_env else None

    cfg = CONFIGS[trial_id]
    print(
        f"=== sim2real trial {trial_id}/{len(CONFIGS) - 1}: "
        f"type={cfg['type']}  real_value={cfg['value']} "
        f"sim=(mu={SIM_COMPETITION_MU},σ={SIM_COMPETITION_SIGMA},η={SIM_HORIZON_ETA}) ===",
        flush=True,
    )
    run_config(trial_id, scratch_dir=scratch_dir)


if __name__ == "__main__":
    main()
