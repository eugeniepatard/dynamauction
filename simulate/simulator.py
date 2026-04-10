"""
Scenario dataclass and Simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from primitives import (
    AuctionRule,
    ArrivalProcess,
    CompetitionDistribution,
    DisplaySampler,
    ExogenousProcess,
    RewardKernel,
)
from policy import BiddingPolicy


@dataclass
class Scenario:
    auction: AuctionRule
    competition: CompetitionDistribution
    arrival: ArrivalProcess
    kernel: RewardKernel
    display_sampler: DisplaySampler
    exogenous: ExogenousProcess
    policy: BiddingPolicy


class Simulator:
    def __init__(self, scenario: Scenario, campaign_cfg):
        self.scenario = scenario
        self.cfg = campaign_cfg

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run n_runs independent trajectories.

        Returns
        -------
        auctions_df    : one row per auction opportunity
        displays_df    : one row per won auction (display)
        conversions_df : one row per conversion event
        summary_df     : one row per run with aggregate statistics
        """
        u = np.array(self.cfg.user_features, dtype=float)
        horizon = float(self.cfg.horizon)
        n_runs = int(self.cfg.n_runs)
        rng = np.random.default_rng(int(self.cfg.seed))

        all_auctions: List[dict] = []
        all_displays: List[dict] = []
        all_conversions: List[dict] = []

        for run_id in range(n_runs):
            a, d, c = self._run_single(run_id, u, horizon, rng)
            all_auctions.extend(a)
            all_displays.extend(d)
            all_conversions.extend(c)

        auctions_df = pd.DataFrame(all_auctions)
        displays_df = pd.DataFrame(all_displays)
        conversions_df = pd.DataFrame(all_conversions)
        summary_df = self._summarise(auctions_df, conversions_df, n_runs)

        return auctions_df, displays_df, conversions_df, summary_df

    # ── private ────────────────────────────────────────────────────────────────

    def _run_single(
        self,
        run_id: int,
        u: np.ndarray,
        horizon: float,
        rng: np.random.Generator,
    ) -> Tuple[List[dict], List[dict], List[dict]]:

        sc = self.scenario
        sc.exogenous.generate(horizon, rng)
        y_func = sc.exogenous.y

        auction_times = sc.arrival.sample_times(horizon, rng)
        ctx = sc.kernel.initial_context()

        auctions: List[dict] = []
        displays: List[dict] = []
        conversions: List[dict] = []

        for auction_id, t in enumerate(auction_times):
            t = float(t)
            x = sc.display_sampler.sample(rng)
            bid = sc.policy.bid(u, x, ctx, y_func, t, horizon, sc)
            B = sc.competition.sample(rng)
            won = sc.auction.win(bid, B)
            payment = sc.auction.payment(bid, B) if won else 0.0
            y_t = y_func(t)

            auctions.append(dict(
                run_id=run_id,
                auction_id=auction_id,
                time=t,
                bid=bid,
                competing_bid=B,
                won=won,
                payment=payment,
                context=ctx,
                y=y_t,
            ))

            if won:
                displays.append(dict(
                    run_id=run_id,
                    auction_id=auction_id,
                    time=t,
                    context=ctx,
                    y=y_t,
                    **{f"x_{j}": float(v) for j, v in enumerate(x)},
                ))

                # Immediate conversions ~ Poisson(κ)
                kap = sc.kernel.kappa(u, x, ctx, y_t)
                for _ in range(int(rng.poisson(kap))):
                    conversions.append(dict(
                        run_id=run_id,
                        auction_id=auction_id,
                        display_time=t,
                        conversion_time=t,
                        delay=0.0,
                        is_immediate=True,
                    ))

                # Delayed conversions via Ogata thinning
                for s in _sample_delayed(sc.kernel, u, x, ctx, y_func, t, horizon, rng):
                    conversions.append(dict(
                        run_id=run_id,
                        auction_id=auction_id,
                        display_time=t,
                        conversion_time=t + s,
                        delay=s,
                        is_immediate=False,
                    ))

                ctx = sc.kernel.update_context(ctx, x, t)

        return auctions, displays, conversions

    @staticmethod
    def _summarise(
        auctions: pd.DataFrame,
        conversions: pd.DataFrame,
        n_runs: int,
    ) -> pd.DataFrame:
        records = []
        for run_id in range(n_runs):
            a = auctions[auctions.run_id == run_id] if len(auctions) else auctions
            c = conversions[conversions.run_id == run_id] if len(conversions) else conversions
            n_auc = len(a)
            n_wins = int(a.won.sum()) if n_auc else 0
            n_conv = len(c)
            n_imm = int(c.is_immediate.sum()) if n_conv else 0
            records.append(dict(
                run_id=run_id,
                n_auctions=n_auc,
                n_wins=n_wins,
                win_rate=round(n_wins / n_auc, 4) if n_auc else 0.0,
                total_spend=round(float(a.payment.sum()), 4) if n_auc else 0.0,
                n_conversions=n_conv,
                n_immediate=n_imm,
                n_delayed=n_conv - n_imm,
            ))
        return pd.DataFrame(records)


# ── Ogata thinning ─────────────────────────────────────────────────────────────

def _sample_delayed(
    kernel: RewardKernel,
    u: np.ndarray,
    x: np.ndarray,
    ctx: float,
    y_func: Callable[[float], float],
    t: float,
    horizon: float,
    rng: np.random.Generator,
) -> List[float]:
    """
    Sample delayed conversion delays in (0, horizon−t] via Ogata's thinning.

    Steps:
      1. Compute λ_max, an upper bound on λ(s) over s ∈ [0, horizon−t].
      2. Draw candidate times from a homogeneous Poisson(λ_max) process.
      3. Accept candidate at delay s with probability λ(s) / λ_max.
    """
    remaining = horizon - t
    if remaining <= 0.0:
        return []
    lam_max = kernel.lambda_upper_bound(u, x, ctx, t, remaining)
    if lam_max <= 0.0:
        return []

    delays: List[float] = []
    s = 0.0
    while s < remaining:
        s += rng.exponential(1.0 / lam_max)
        if s >= remaining:
            break
        lam = kernel.lambda_(u, x, ctx, y_func(t + s), t, s)
        if rng.uniform() < lam / lam_max:
            delays.append(s)
    return delays
