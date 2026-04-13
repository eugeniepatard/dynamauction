"""
Bidding policy interface and baseline implementations.

All policies have oracle access to the full scenario (no learning).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from dynamicauc.simulator import Scenario

# TODO: coment rendre ce classes compatibles avec des training RL

class BiddingPolicy(ABC):
    """
    Interface for bidding policies.

    At each auction opportunity the policy maps the current campaign state
    to a non-negative bid.

    Parameters passed to bid()
    --------------------------
    u        : user feature vector (fixed for the whole campaign)
    x        : display feature vector for this auction opportunity
    ctx      : carry-over context (summarises past won displays)
    y_func   : exogenous process y(·)
    t        : current auction time
    horizon  : campaign end time T
    scenario : full scenario — gives access to all primitives
    """

    @abstractmethod
    def bid(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        scenario: "Scenario",
    ) -> float: ...


class FiPlaPolicy(BiddingPolicy):
    """
    First-Price-Like (FiPla) baseline.

    Bids the expected marginal reward generated if the auction is won:

        b = κ(u, x, ctx, y(t)) + ∫₀^{T−t} λ(u, x, ctx, y(t+s), t, s) ds

    This is the truthful bid equal to the expected value of the display.
    It is optimal under second-price rules and serves as a natural
    reference baseline under first-price rules.
    """

    def bid(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        scenario: "Scenario",
    ) -> float:
        return scenario.kernel.expected_reward(u, x, ctx, y_func, t, scenario.horizon)
