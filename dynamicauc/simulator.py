"""
Scenario dataclass and Simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Dict

import numpy as np
import pandas as pd


from dynamicauc.policy import BiddingPolicy
from dynamicauc.dataclasses import AuctionState, Scenario
from dynamicauc.utils import sample_delayed

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

    @staticmethod
    def _run_single_auction(
        self,
        run_id: int,
        auction_state: AuctionState,
        u: np.ndarray,
        y_func: Any,
        horizon: float,
        rng: np.random.Generator
        ) -> Tuple[Dict, Dict, List[Dict], Any]:
        """
        Handles the logic for a single auction event: bidding, winning, 
        and generating rewards (conversions).

        Parameters
        ----------
            run_id : int
                Identifier for the current simulation run.
            auction_id : int
                Identifier for the current auction within the run.
            t : float
                Time of the auction event.
            u : np.ndarray
                User features vector.
            ctx : Any
                Current kernel context.
            y_func : Callable
                Function to get exogenous information y(t).
            horizon : float
                Total simulation horizon (for delayed conversion sampling).
            rng : np.random.Generator
                Random number generator for stochasticity.
        Returns
        -------
            auction_data : dict
                Record of the auction event (bids, outcome, context, etc.).
            display_data : dict or None
                Record of the display event if the auction was won, else None.
            conversions_data : List[dict], Shape : (n_wins * avg_conversions_per_win,)
                List of conversion event records generated from this display.
            new_ctx : Any
                Updated kernel context after this auction/display.
        """
        sc = self.scenario
        x = sc.display_sampler.sample(rng)
        bid = sc.policy.bid(u, x, auction_state.ctx, y_func, auction_state.t, horizon, sc)
        B = sc.competition.sample(rng)
        
        won = sc.auction.win(bid, B)
        payment = sc.auction.payment(bid, B) if won else 0.0
        y_t = y_func(auction_state.t)

        # Prepare Auction Record
        auction_data = dict(
            run_id=run_id,
            auction_id=auction_state.step_idx,
            time=auction_state.t,
            bid=bid,
            competing_bid=B,
            won=won,
            payment=payment,
            context=auction_state.ctx,
            y=y_t,
        )

        display_data = None
        conversions_data = []
        new_ctx = auction_state.ctx

        if won:
            # Prepare Display Record
            display_data = dict(
                run_id=run_id,
                auction_id=auction_state.step_idx,
                time=auction_state.t,
                context=auction_state.ctx,
                y=y_t,
                **{f"x_{j}": float(v) for j, v in enumerate(x)},
            )

            # Handle Immediate conversions ~ Poisson(κ)
            kap = sc.kernel.kappa(u, x, auction_state.ctx, y_t)
            for _ in range(int(rng.poisson(kap))):
                conversions_data.append(dict(
                    run_id=run_id,
                    auction_id=auction_state.step_idx,
                    display_time=auction_state.t,
                    conversion_time=auction_state.t,
                    delay=0.0,
                    is_immediate=True,
                ))

            # Handle Delayed conversions via Ogata thinning
            for s in sample_delayed(sc.kernel, u, x, auction_state.ctx, y_func, auction_state.t, horizon, rng):
                conversions_data.append(dict(
                    run_id=run_id,
                    auction_id=auction_state.step_idx,
                    display_time=auction_state.t,
                    conversion_time=auction_state.t + s,
                    delay=s,
                    is_immediate=False,
                ))

            new_ctx = sc.kernel.update_context(auction_state.ctx, x, auction_state.t)

        return auction_data, display_data, conversions_data, new_ctx

# TODO : change that to use the env gym
    def _run_single(
        self,
        run_id: int,
        u: np.ndarray,
        horizon: float,
        rng: np.random.Generator,
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Runs a single full simulation episode (one user over the horizon).
        """
        sc = self.scenario
        
        # Generate external interference (competitor clicks, etc.)
        sc.exogenous.generate(horizon, rng)
        y_func = sc.exogenous.y

        # Sample arrival times (opportunity process)
        auction_times = sc.arrival.sample_times(horizon, rng)
        ctx = sc.kernel.initial_context()

        auctions: List[dict] = []
        displays: List[dict] = []
        conversions: List[dict] = []

        for auction_id, t in enumerate(auction_times):
            # Call the single auction logic
            auction_rec, display_rec, conversion_recs, ctx = self._run_single_auction(
                run_id=run_id,
                auction_id=auction_id,
                t=float(t),
                u=u,
                ctx=ctx,
                y_func=y_func,
                horizon=horizon,
                rng=rng
            )

            auctions.append(auction_rec)
            if display_rec:
                displays.append(display_rec)
            if conversion_recs:
                conversions.extend(conversion_recs)

        return auctions, displays, conversions

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
                for s in sample_delayed(sc.kernel, u, x, ctx, y_func, t, horizon, rng):
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
    

def sample_delayed(
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