"""
Bidding policy interface and simple baseline implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Any

import numpy as np

if TYPE_CHECKING:
    from dynamicauc.simulator import Scenario

class BiddingPolicy(ABC):
    """
    Interface for bidding policies.

    At each auction opportunity the policy maps the current campaign state
    to a non-negative bid.

    Parameters passed to bid()
    --------------------------
    observations : Dict[str, Any] - Gymnasium-style observation dictionary.
    """

    @abstractmethod
    def bid(
        self,
        observations: Dict[str, Any],
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
        observations: Dict[str, Any],
    ) -> float:
        return observations["expected_reward"].item()

class ConstantPolicy(BiddingPolicy):
    """Always Bid the same value, ignoring saturation/recency."""
    def __init__(self, bid_value: float = 0.5):
        self.bid_value = bid_value

    def bid(self, observations: Dict[str, Any]) -> float:
        return self.bid_value

class RandomPolicy(BiddingPolicy):
    """Bid randomly between 0 and an upper bound."""
    def __init__(self, max_bid: float = 2.0):
        self.max_bid = max_bid

    def bid(self, observations: Dict[str, Any]) -> float:
        return np.random.uniform(0, self.max_bid)