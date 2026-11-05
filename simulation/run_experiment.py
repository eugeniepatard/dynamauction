#!/usr/bin/env python3
"""
Compare FiPla, Shading and MPC on one DynamAuction scenario, on a single machine (all its cores).

What it runs, in this order (every policy on the same campaigns, so all comparisons are paired):
  1. FiPla           bids the exact myopic value V0 of the display (the `expected_reward` observation).
  2. Shading sweep   bids alpha * V0 for every alpha of --alphas: "oracle" shading, FiPla scaled by alpha, fast. With
                     --shading simulated it is the paper's ShadingPolicy instead: alpha times a Monte-Carlo estimate of
                     the display value (M = --shading-rollouts rollouts per bid), much slower.
  3. MLP head        the valuation head V0_hat used by MPC (MLP 32-32 ReLU trained on FiPla episodes, as in the paper),
                     trained once and reused (skipped with --mpc-v0 oracle: MPC then uses the exact V0).
  4. MPC             MPCPolicy for every planning horizon of --mpc-horizons and every rollout continuation of
                     --alpha-cont: at each auction it evaluates every first-step multiplier of --mpc-alphas with
                     --mpc-rollouts simulated rollouts of H time units, and bids alpha* * V0_hat.
  5. Summary         mean reward +- 95% CI of every policy, paired differences with FiPla and with the best Shading,
                     win rate; printed and written to <out>/summary.csv.
Campaign s uses env seed s and policy seed s + 1000 (the paper's protocol); the MLP is trained on env seeds 10000+.

Commands (from the repository root, with the package installed: pip install -e ".[benchmark]")

  # quick test, about 2 minutes: few campaigns, few rollouts, small MLP training
  python simulation/run_experiment.py --n-campaigns 8 --mpc-horizons 5 --mpc-rollouts 4 --mlp-episodes 50 --mlp-epochs 20

  # main survival scenario, 200 campaigns, MPC with H = 15 and 25
  python simulation/run_experiment.py --scenario survival-second-price-exponential --n-campaigns 200 --workers 16

  # any parameter of the preset: --set NAME=VALUE, the overrides of dynamauction.make (repeat --set)
  python simulation/run_experiment.py --set gamma_beta=0.1 --set gamma_c=0.9 --set competition_mu=-1 --set horizon_eta=50

  # burst of auctions (rate 10 on [0, 10), then 1), beta = 0.3: MPC with a continuation per phase (0.3 during the burst,
  # 0.7 after) and with the best Shading alpha, H = 5, 15, 25
  python simulation/run_experiment.py --scenario survival-second-price-exponential-burst-arrivals-beta0.3 \
      --mpc-horizons 5 15 25 --alpha-cont 0:0.3,10:0.7 best --workers 32

  # step-decay arrivals lambda(t) = rate * factor^floor(t / period), with other parameters than the preset
  python simulation/run_experiment.py --scenario survival-second-price-exponential-decaying-arrivals-beta0.1 \
      --set arrival_rate=10 --set arrival_period=10 --set arrival_factor=0.25 --alpha-cont 0.3

  # effective-frequency kernel (wear-in, where MPC beats Shading): bids above V0 must be allowed
  python simulation/run_experiment.py --scenario effective-frequency-second-price-lognormal \
      --alphas 0.5 0.8 1 1.25 1.5 2 3 --mpc-alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.5 2 3 5 \
      --mpc-horizons 25 --mpc-rollouts 80

  # only the table, from the results of a finished or interrupted run
  python simulation/run_experiment.py --out simulation/results/<run> --summary-only

  On a SLURM cluster: run the same command in one job, with --workers equal to the cores of the job, e.g.
      sbatch --cpus-per-task=32 --mem=32G --time=24:00:00 --wrap="python simulation/run_experiment.py ... --workers 32"

Parameters to adapt
  --scenario            preset of dynamauction (dynamauction.list_scenarios()); the survival presets of the study:
                        survival-second-price-exponential (constant rate 1, c = 0.9, beta = 1),
                        survival-second-price-exponential-decaying-arrivals[-beta0.1] (rate 1 * 0.5^floor(t / 10)),
                        survival-second-price-exponential-burst-arrivals-beta0.3 (rate 10 on [0, 10), then 1)
  --set NAME=VALUE      override of a preset parameter (value read as JSON: numbers, lists, strings), e.g.
                        gamma_c, gamma_beta (fatigue strength / recovery rate), competition_mu, horizon_eta,
                        arrival_rate / arrival_period / arrival_factor (step-decay presets). To change the family of
                        the arrival process, prefer a preset (e.g. the burst one).
  --n-campaigns         number of campaigns (default 200; the paper's final numbers use 1000)
  --workers             parallel processes (default: all the cores)
  --alphas              Shading multipliers (default 0.1 ... 1.0)
  --shading             oracle (alpha * exact V0, fast; default) or simulated (the paper's ShadingPolicy)
  --mpc-horizons        planning horizons H of MPC, in time units (default 15 25); --no-mpc to skip MPC
  --mpc-rollouts        rollouts M per candidate alpha (default 60; 20 is about 3x faster but noisier)
  --mpc-alphas          first-step multipliers of MPC (default 0.1 ... 1.0)
  --alpha-cont          multiplier of the rollout continuation, one MPC run per value (default 1): a number, best
                        (the best alpha of the Shading sweep of this run), or a schedule T0:A0,T1:A1 (A0 from time T0,
                        A1 from time T1, e.g. 0:0.3,10:0.7). A continuation close to the best Shading alpha matters
                        when auctions are dense: with 1, the rollouts over-estimate how cheap a win is.
  --mpc-v0              mlp (the paper's valuation head, trained here; default) or oracle (exact V0, no training)
  --mlp-episodes / --mlp-epochs   MLP training size (default 1000 FiPla episodes, 500 epochs)
  --out                 output folder (default simulation/results/<scenario>[_<overrides>]_seeds<first>-<last>)

Outputs, in --out
  config.json                     the command line of the run
  fipla.csv, shading_alpha=<a>.csv (simulated: shading_sim_M<M>_alpha=<a>.csv), mpc_H<H>_M<M>[_ac...].csv
                                  one row per campaign: seed, reward, auctions, wins, payment (MPC: mean alpha chosen)
  mlp/                            the trained valuation head (.pt + .json)
  summary.csv                     the summary table
  A run whose CSV exists is skipped: rerunning the same command resumes an interrupted experiment (each finished
  campaign is kept in <run>.partial.csv until its run is complete).

Running time (one core of a recent machine, main survival preset, about 48 auctions per campaign; denser arrival
schedules cost more, e.g. about 10x for the burst preset)
  FiPla, oracle Shading     a few seconds per 100 campaigns      simulated Shading (M = 20)   about 3 s per campaign
  MLP training              about 5 minutes                      MPC(H = 25, M = 60)          about 1-2 min per campaign
  MPC cost grows with M, with H and with the number of auctions in the planning window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))   # the benchmark package (MLP training) lives in the repository

POLICY_SEED_OFFSET = 1000
MLP_SEED = 10_000
DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ── policies, run in the worker processes ────────────────────────────────────────────────────────────────────────────

class OracleV0:
    """Exact myopic value of the display (the expected_reward observation), as a valuation head for MPC."""

    def predict(self, obs) -> float:
        return float(obs["expected_reward"].ravel()[0])


_V0_CACHE: dict = {}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)   # one core per process: the parallelism is over campaigns


def _valuation_head(scenario_id: str, v0: str):
    if v0 not in _V0_CACHE:
        if v0 == "oracle":
            _V0_CACHE[v0] = OracleV0()
        else:
            from benchmark.train_mlp import load_mlp
            _V0_CACHE[v0] = load_mlp(scenario_id, v0)
    return _V0_CACHE[v0]


def _make_policy(spec: dict, scenario_id: str, seed: int):
    from dynamauction.baseline.baseline_algo import MPCPolicy, ShadingPolicy
    from dynamauction.baseline.estimators import SimulationEstimator
    from dynamauction.baseline.policies import ConstantPolicy

    if spec["kind"] == "shading" and spec["mode"] == "simulated":
        return ShadingPolicy(SimulationEstimator(ConstantPolicy(0.0), n_rollouts=spec["m"], seed=seed + POLICY_SEED_OFFSET),
                             alpha=spec["alpha"])
    if spec["kind"] == "mpc":
        return MPCPolicy(v0=_valuation_head(scenario_id, spec["v0"]), alpha_candidates=spec["alphas"],
                         alpha_cont=spec["alpha_cont"], n_rollouts=spec["m"], planning_horizon=spec["h"],
                         seed=seed + POLICY_SEED_OFFSET, tie_alpha=spec["tie_alpha"])
    return None   # FiPla and oracle Shading bid from the observation


def run_campaigns(task: tuple) -> list[dict]:
    """Play the campaigns `seeds` with the policy `spec`; one result row per campaign."""
    import gymnasium as gym
    import dynamauction as dac

    spec, scenario_id, overrides, seeds = task
    env = gym.make("DynamAuction-v0", scenario=dac.make(scenario_id, **overrides))
    rows = []
    for s in seeds:
        policy = _make_policy(spec, scenario_id, s)
        obs, _ = env.reset(seed=s)
        if policy is not None:
            policy.set_env(env)
        reward, payment, n_auctions, wins, done = 0.0, 0.0, 0, 0, False
        while not done:
            v0 = float(obs["expected_reward"].ravel()[0])
            if policy is None:
                bid = v0 if spec["kind"] == "fipla" else spec["alpha"] * v0
            else:
                bid = policy.bid(obs)
            obs, r, done, _, info = env.step(np.array([bid], dtype=np.float32))
            reward += float(r)
            payment += float(info["auction"]["payment"])
            wins += int(bool(info["auction"]["won"]))
            n_auctions += 1
        row = {"seed": s, "reward": reward, "auctions": n_auctions, "wins": wins, "payment": payment}
        if spec["kind"] == "mpc":
            row["mean_alpha"] = float(np.mean(policy.alpha_log)) if policy.alpha_log else math.nan
        rows.append(row)
    env.close()
    return rows


# ── runs, in the main process ────────────────────────────────────────────────────────────────────────────────────────

def _atomic_csv(df, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_policy(pool, out: Path, name: str, spec: dict, scenario_id: str, overrides: dict, seeds: list[int],
               chunk: int) -> None:
    """All campaigns of one policy, in parallel; resumable (finished campaigns are kept in <name>.partial.csv)."""
    import pandas as pd

    final, partial = out / f"{name}.csv", out / f"{name}.partial.csv"
    if final.exists():
        done = sorted(pd.read_csv(final)["seed"].astype(int))
        if done != sorted(seeds):
            raise RuntimeError(f"{final} holds other campaigns ({len(done)} seeds from {done[0]}) than requested: "
                               "use another --out")
        print(f"  {name}: done already ({final.name})")
        return
    done_rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    todo = [s for s in seeds if s not in {int(r["seed"]) for r in done_rows}]
    tasks = [(spec, scenario_id, overrides, todo[i:i + chunk]) for i in range(0, len(todo), chunk)]
    t0, n_done = time.time(), len(done_rows)
    for rows in pool.imap_unordered(run_campaigns, tasks):
        done_rows += rows
        n_done += len(rows)
        _atomic_csv(pd.DataFrame(done_rows), partial)
        print(f"\r  {name}: {n_done}/{len(seeds)} campaigns ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    _atomic_csv(pd.DataFrame(done_rows).sort_values("seed"), final)
    partial.unlink(missing_ok=True)


def train_head(out: Path, scenario_id: str, overrides: dict, args) -> str:
    """The MLP valuation head of MPC (the paper's procedure); trained once per output folder."""
    import torch
    from benchmark.train_mlp import train_mlp

    model_dir = out / "mlp"
    models = sorted(model_dir.glob("*.pt"))
    if models:
        print(f"  MLP head: reusing {models[0].relative_to(out)}")
        return str(models[0])
    print(f"  MLP head: training on {args.mlp_episodes} FiPla episodes, {args.mlp_epochs} epochs")
    torch.manual_seed(MLP_SEED)
    train_mlp(scenario_id=scenario_id, scenario_overrides=overrides, n_episodes=args.mlp_episodes,
              hidden_dims=args.mlp_hidden, activation="relu", n_epochs=args.mlp_epochs, lr=1e-3, batch_size=256,
              val_frac=0.15, seed=MLP_SEED, save_dir=model_dir, log_dir=model_dir / "tb")
    return str(sorted(model_dir.glob("*.pt"))[0])


def shading_name(alpha: float, args) -> str:
    """shading_alpha=<a> (oracle Shading) or shading_sim_M<M>_alpha=<a> (the paper's simulated Shading)."""
    return (f"shading_alpha={alpha:g}" if args.shading == "oracle"
            else f"shading_sim_M{args.shading_rollouts}_alpha={alpha:g}")


def best_shading_alpha(out: Path, args) -> float:
    import pandas as pd

    means = {a: pd.read_csv(out / f"{shading_name(a, args)}.csv")["reward"].mean() for a in args.alphas
             if (out / f"{shading_name(a, args)}.csv").exists()}
    if not means:
        raise RuntimeError("--alpha-cont best needs the Shading sweep of this run")
    return max(means, key=means.get)


def parse_alpha_cont(value: str):
    """'1' -> 1.0, 'best' -> 'best', 'T0:A0,T1:A1' -> ((T0, A0), (T1, A1))."""
    if value == "best":
        return "best"
    if ":" in value:
        return tuple((float(t), float(a)) for t, a in (pair.split(":") for pair in value.split(",")))
    return float(value)


def ac_suffix(value) -> str:
    if value == "best":
        return "_acbest"
    if isinstance(value, tuple):
        return "_acs" + "_".join(f"{t:g}={a:g}" for t, a in value)
    return "" if value == 1.0 else f"_ac{value:g}"


# ── summary ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def summarize(out: Path) -> None:
    import pandas as pd

    runs = {p.stem: pd.read_csv(p).set_index("seed") for p in sorted(out.glob("*.csv"))
            if p.name != "summary.csv" and not p.name.endswith(".partial.csv")}
    if "fipla" not in runs:
        print(f"No results in {out}")
        return
    ci = lambda x: 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else math.nan   # noqa: E731
    shading = {k: v for k, v in runs.items() if k.startswith("shading_")}
    best = max(shading, key=lambda k: shading[k]["reward"].mean()) if shading else None
    order = ["fipla"] + sorted(shading, key=lambda k: float(k.split("=")[1])) + sorted(k for k in runs if k.startswith("mpc_"))
    rows = []
    for name in order:
        r = runs[name]
        row = {"policy": name, "campaigns": len(r), "mean_reward": r["reward"].mean(), "ci95": ci(r["reward"]),
               "win_rate": r["wins"].sum() / max(r["auctions"].sum(), 1)}
        for ref, col in (("fipla", "vs_fipla"), (best, "vs_best_shading")):
            if ref is not None and ref != name:
                d = (r["reward"] - runs[ref]["reward"]).dropna()
                row[col], row[col + "_ci95"] = d.mean(), ci(d)
        if "mean_alpha" in r:
            row["mpc_mean_alpha"] = r["mean_alpha"].mean()
        rows.append(row)
    cols = ["policy", "campaigns", "mean_reward", "ci95", "vs_fipla", "vs_fipla_ci95", "vs_best_shading",
            "vs_best_shading_ci95", "win_rate", "mpc_mean_alpha"]
    table = pd.DataFrame(rows).reindex(columns=cols)
    _atomic_csv(table, out / "summary.csv")
    pd.set_option("display.width", 200)
    print(f"\n{out}  (best Shading: {best}; differences are paired, on the same campaigns, +- 95% CI)\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ── command line ─────────────────────────────────────────────────────────────────────────────────────────────────────

def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="survival-second-price-exponential", help="dynamauction preset")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="preset override (repeatable)")
    p.add_argument("--n-campaigns", type=int, default=200)
    p.add_argument("--first-seed", type=int, default=0, help="campaigns use env seeds first-seed ... first-seed + n - 1")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--chunk", type=int, default=10, help="campaigns per task for FiPla and Shading (MPC: 1)")
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS, help="Shading multipliers")
    p.add_argument("--shading", choices=["oracle", "simulated"], default="oracle")
    p.add_argument("--shading-rollouts", type=int, default=20, help="rollouts of the simulated Shading")
    p.add_argument("--no-mpc", action="store_true")
    p.add_argument("--mpc-horizons", type=float, nargs="+", default=[15.0, 25.0])
    p.add_argument("--mpc-rollouts", type=int, default=60)
    p.add_argument("--mpc-alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--alpha-cont", nargs="+", default=["1"], help="number, best, or schedule T0:A0,T1:A1 (one run each)")
    p.add_argument("--tie-alpha", type=float, default=None, help="tie-break of MPC towards this alpha (default: first candidate)")
    p.add_argument("--mpc-v0", choices=["mlp", "oracle"], default="mlp")
    p.add_argument("--mlp-episodes", type=int, default=1000)
    p.add_argument("--mlp-epochs", type=int, default=500)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    overrides = parse_overrides(args.set)
    tag = "".join(f"_{k}={v}" for k, v in sorted(overrides.items())).replace("/", "-").replace(" ", "")
    seeds_tag = f"_seeds{args.first_seed}-{args.first_seed + args.n_campaigns - 1}"
    out = args.out or REPO_ROOT / "simulation" / "results" / f"{args.scenario}{tag}{seeds_tag}"
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(out)
        return

    import dynamauction as dac
    dac.make(args.scenario, **overrides)   # fails here, not in the workers, on a wrong preset or override
    (out / "config.json").write_text(json.dumps({"argv": sys.argv[1:], "scenario": args.scenario, "overrides": overrides},
                                                indent=2))
    seeds = list(range(args.first_seed, args.first_seed + args.n_campaigns))
    print(f"[{args.scenario} {overrides or ''}] {len(seeds)} campaigns, {args.workers} workers, output {out}")
    with Pool(args.workers, initializer=_init_worker) as pool:
        run_policy(pool, out, "fipla", {"kind": "fipla"}, args.scenario, overrides, seeds, args.chunk)
        for a in args.alphas:
            spec = {"kind": "shading", "mode": args.shading, "alpha": a, "m": args.shading_rollouts}
            run_policy(pool, out, shading_name(a, args), spec, args.scenario, overrides, seeds, args.chunk)
        if not args.no_mpc:
            v0 = "oracle" if args.mpc_v0 == "oracle" else train_head(out, args.scenario, overrides, args)
            for value in (parse_alpha_cont(v) for v in args.alpha_cont):
                alpha_cont = best_shading_alpha(out, args) if value == "best" else value
                if value == "best":
                    print(f"  alpha_cont = best Shading alpha = {alpha_cont:g}")
                for h in args.mpc_horizons:
                    spec = {"kind": "mpc", "h": h, "m": args.mpc_rollouts, "alphas": args.mpc_alphas, "alpha_cont": alpha_cont,
                            "tie_alpha": args.tie_alpha, "v0": v0}
                    name = f"mpc_H{h:g}_M{args.mpc_rollouts}{ac_suffix(value)}" + ("_v0oracle" if v0 == "oracle" else "")
                    run_policy(pool, out, name, spec, args.scenario, overrides, seeds, chunk=1)
    summarize(out)


if __name__ == "__main__":
    main()
