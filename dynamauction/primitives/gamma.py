"""Carry-over context primitives x^{-i} = aggregate_{i'<i}[gamma(xi', ti', ti)]"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Gamma(ABC):
    """Abstract carry-over context function gamma(x', t', t)."""

    @abstractmethod
    def initiate_context(self) -> float: ...

    @abstractmethod
    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float: ...

class AdditiveGamma(Gamma):
    """Carry-over context function of the form gamma(x', t', ti) = sum_{i'<i}[past won auctions]"""
    def __init__(self, initial_value: float = 0.0):
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        return ctx + 1.0

class MultiplicativeGamma(Gamma):
    """Carry-over context function of the form gamma(x', t', ti) = produit_{i'<i}[(1 - c*exp(-(ti - ti')))]
    with ti' the last won auction."""

    def __init__(self, c : float, initial_value: float = 1.0):
        self.c = c
        self.initial_value = initial_value

    def initiate_context(self) -> float:
        return self.initial_value

    def update_context(self, ctx: float, x: np.ndarray, u: np.ndarray, t_display: float, t_past_won_auctions: List[float]) -> float:
        sorted_past_won_auctions = sorted(t_past_won_auctions)
        last_won_auction_time = sorted_past_won_auctions[-1] if sorted_past_won_auctions else self.initial_value
        return ctx * (1.0 - self.c * np.exp(-(t_display - last_won_auction_time)))

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