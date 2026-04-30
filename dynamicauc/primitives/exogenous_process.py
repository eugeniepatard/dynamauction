"""
Exogenous time-varying signal y(t) independent of the bidding policy.

``ExogenousProcess`` (abstract): evaluates the signal at time t.
``TrivialProcess``: always returns 1.0 (no exogenous variation).
The signal is passed to the ``RewardKernel`` and observed by the agent,
modelling external factors such as seasonality or user engagement trends.
"""
from abc import ABC, abstractmethod

import numpy as np


class ExogenousProcess(ABC):
    """
    External information process y(t).
    Must be re-generated at the start of each simulation run via generate().
    """

    @abstractmethod
    def generate(self, horizon: float, rng: np.random.Generator) -> None: ...

    @abstractmethod
    def y(self, t: float) -> float: ...


class TrivialProcess(ExogenousProcess):
    """No exogenous events; y(t) = 0 everywhere."""

    def generate(self, horizon: float, rng: np.random.Generator) -> None:
        pass

    def y(self, t: float) -> float:
        return 0.0


class CompetitorClickProcess(ExogenousProcess):
    """
    Homogeneous Poisson competitor click process.

    y(t) = time of the most recent competitor click strictly before t,
           or −inf if no click has occurred yet.

    Convention used by CompetitorExtinctionKernel:
        a click occurred in (t_display, t_display + s)  ⟺  y(t_display + s) > t_display
    """

    def __init__(self, rate: float = 1.0):
        self.rate = rate
        self._click_times: np.ndarray = np.array([])

    def generate(self, horizon: float, rng: np.random.Generator) -> None:
        n = rng.poisson(self.rate * horizon)
        self._click_times = np.sort(rng.uniform(0.0, horizon, n))

    def y(self, t: float) -> float:
        before = self._click_times[self._click_times < t]
        return float(before[-1]) if len(before) > 0 else float("-inf")
