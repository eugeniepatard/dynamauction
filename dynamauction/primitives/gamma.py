"""Carry-over context primitives x^{-i} = aggregate_{i'<i}[gamma(xi', ti', ti)]"""

import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Gamma(ABC):
    """Abstract carry-over context function gamma(x', t', t).

    The stored context only changes when an auction is won (``update_context``).
    ``current_context`` gives S(t⁻), the context used to value the auction at time t
    and shown in the observation; it is not stored.
    """

    @abstractmethod
    def initiate_context(self) -> float: ...

    @abstractmethod
    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float: ...

    def current_context(self, ctx: float, t: float, t_past_won_auctions: List[float]) -> float:
        return ctx

class AdditiveGamma(Gamma):
    """Carry-over context function of the form gamma(x', t', ti) = sum_{i'<i}[past won auctions]"""
    def __init__(self, initial_value: float = 0.0):
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return ctx + 1.0

class MultiplicativeGamma(Gamma):
    """S(t⁻) = Π_{i: t_i < t} (1 − c·exp(−beta·(t − t_i))) over the past wins t_i, 1 before the first win.

    Eq. (carry-over) with the monoid ([0, 1], ×, 1) and gamma(x_i, t_i, t) = 1 − c·exp(−beta·(t − t_i)):
    each win contributes its own factor, 1 − c right after the win, which recovers towards 1 as time
    passes. beta is the recovery rate: beta < 1 makes the fatigue of a win last longer, beta > 1 shorter.
    The context only depends on the win times: the stored context is not used and stays 1.
    """

    def __init__(self, c : float, beta: float = 1.0):
        self.c = c
        self.beta = beta

    def initiate_context(self) -> float:
        return 1.0

    def current_context(self, ctx: float, t: float, t_past_won_auctions: List[float]) -> float:
        s = 1.0   # empty product: no factor before the first win
        for t_i in t_past_won_auctions:
            s *= 1.0 - self.c * math.exp(-self.beta * (t - t_i))
        return s

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return ctx

class LastWonTimeGamma(Gamma):
    """Context is the time of the last won auction."""
    def __init__(self, initial_value: float = -1e6):
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return t_display

class TimeSinceLastWinGamma(Gamma):
    """S(t⁻) = t − t_last: the time since the last won auction (recency).

    Eq. (carry-over) with the monoid ([0, +∞], min, +∞) and gamma(x_i, t_i, t) = t − t_i: the minimum over the past
    wins is the time since the most recent one. The stored context is the time of the last win, ``initial_value``
    before the first win, so that t − initial_value stands for +∞ (−1000 by default, as the last-win time of the
    recency preset). With the survival kernel, f = clip(S·(w_u⊤u)(w_x⊤x), 0, 1): a display is worth 0 right after a
    win and recovers linearly with the time since it, faster for more relevant displays, up to its full value.
    """
    def __init__(self, initial_value: float = -1000.0):
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

    def current_context(self, ctx: float, t: float, t_past_won_auctions: List[float]) -> float:
        return t - ctx

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return t_display

class ConstantGamma(Gamma):
    """Context is always constant."""
    def __init__(self, value: float = 0.0):
        self.value = value

    def initiate_context(self) -> float:
        return self.value

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return self.value