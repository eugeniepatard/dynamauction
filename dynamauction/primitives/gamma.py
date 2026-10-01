"""Carry-over context primitives x^{-i} = aggregate_{i'<i}[gamma(xi', ti', ti)]"""

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
    """S(t_{i+1}⁻) = S(t_i⁻) · (1 − c·exp(−(t_{i+1} − t_i))), S(0⁻) = initial_value.

    At an auction at time t: S(t⁻) = stored · (1 − c·exp(−(t − t_last))) with t_last the
    last won auction, and no factor before the first win. A win stores S(t_display⁻).
    """

    def __init__(self, c : float, initial_value: float = 1.0):
        self.c = c
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

    def current_context(self, ctx: float, t: float, t_past_won_auctions: List[float]) -> float:
        if not t_past_won_auctions:
            return ctx
        return ctx * (1.0 - self.c * np.exp(-(t - max(t_past_won_auctions))))

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return self.current_context(ctx, t_display, t_past_won_auctions)

class LastWonTimeGamma(Gamma):
    """Context is the time of the last won auction."""
    def __init__(self, initial_value: float = -1e6):
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

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