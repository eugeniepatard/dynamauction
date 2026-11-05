"""
Auction opportunity arrival processes.

``ArrivalProcess`` (abstract): generates the sequence of auction times
``t⁽¹⁾ < … < t⁽ᴺ⁾`` within a campaign horizon T.
``PiecewiseConstantPoisson``: Poisson process with a piecewise-constant
rate λ(t), generalising the homogeneous Poisson case.
``ConstantPoisson``: homogeneous Poisson process.
``StepDecayPoisson``: Poisson process whose rate is divided by a constant
factor every ``period`` time units, λ(t) = rate · factor^⌊t / period⌋.
"""
import math
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ArrivalProcess(ABC):
    @abstractmethod
    def sample_times(self, horizon: float, rng: np.random.Generator) -> np.ndarray: ...

    def sample_times_from(self, t_start: float, duration: float, rng: np.random.Generator) -> np.ndarray:
        """Auction times in [t_start, t_start + duration] of the process started at time 0 (planning rollouts).

        The default shifts a draw on [0, duration]: exact for a time-homogeneous process only;
        processes whose rate depends on the absolute time override it.
        """
        return self.sample_times(duration, rng) + t_start


class PiecewiseConstantPoisson(ArrivalProcess):
    """
    Non-homogeneous Poisson process with piecewise-constant intensity.

    Parameters
    ----------
    breakpoints : interior breakpoints t^(1), ..., t^(m-1)  (not 0 or T)
    intensities : one intensity per interval (len = len(breakpoints) + 1)
    """

    def __init__(self, breakpoints: List[float], intensities: List[float]):
        self.breakpoints = breakpoints
        self.intensities = intensities

    def sample_times(self, horizon: float, rng: np.random.Generator) -> np.ndarray:
        edges = [0.0] + list(self.breakpoints) + [horizon]
        times: List[float] = []
        for k, mu in enumerate(self.intensities):
            t0, t1 = edges[k], min(edges[k + 1], horizon)
            if t0 >= horizon:
                break
            n = rng.poisson(mu * (t1 - t0))
            times.extend(rng.uniform(t0, t1, n).tolist())
        return np.sort(times)

    def sample_times_from(self, t_start: float, duration: float, rng: np.random.Generator) -> np.ndarray:
        """The intensities of the intervals that [t_start, t_start + duration] overlaps, in absolute time."""
        if not self.breakpoints:   # homogeneous: the shifted draw is exact
            return super().sample_times_from(t_start, duration, rng)
        t_end = t_start + duration
        edges = [0.0] + list(self.breakpoints) + [math.inf]
        times: List[float] = []
        for k, mu in enumerate(self.intensities):
            t0, t1 = max(edges[k], t_start), min(edges[k + 1], t_end)
            if t0 >= t1:
                continue
            n = rng.poisson(mu * (t1 - t0))
            times.extend(rng.uniform(t0, t1, n).tolist())
        return np.sort(times)


class ConstantPoisson(ArrivalProcess):
    """Homogeneous Poisson process with constant rate."""

    def __init__(self, rate: float):
        self.rate = rate

    def sample_times(self, horizon: float, rng: np.random.Generator) -> np.ndarray:
        n = rng.poisson(self.rate * horizon)
        return np.sort(rng.uniform(0.0, horizon, n))


class StepDecayPoisson(ArrivalProcess):
    """
    Poisson process whose rate drops by a constant factor every ``period`` time units:

        λ(t) = rate · factor^⌊t / period⌋,

    e.g. rate = 1, period = 10, factor = 0.5: λ = 1 on [0, 10), 0.5 on [10, 20), 0.25 on [20, 30), …
    The rate depends on the absolute time t since the start of the campaign, also in the planning rollouts
    (``sample_times_from``). Expected number of auctions before T: rate · period · (1 − factor^{T/period}) / (1 − factor)
    for T a multiple of period (at most rate · period / (1 − factor) when factor < 1).
    """

    def __init__(self, rate: float = 1.0, period: float = 10.0, factor: float = 0.5):
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        if rate < 0 or factor < 0:
            raise ValueError(f"rate and factor must be >= 0, got rate={rate}, factor={factor}")
        self.rate = rate
        self.period = period
        self.factor = factor

    def intensity(self, t: float) -> float:
        """λ(t) = rate · factor^⌊t / period⌋."""
        return self.rate * self.factor ** math.floor(t / self.period)

    def sample_times(self, horizon: float, rng: np.random.Generator) -> np.ndarray:
        return self.sample_times_from(0.0, horizon, rng)

    def sample_times_from(self, t_start: float, duration: float, rng: np.random.Generator) -> np.ndarray:
        t_end = t_start + duration
        times: List[float] = []
        k = math.floor(t_start / self.period)
        while k * self.period < t_end:
            t0, t1 = max(k * self.period, t_start), min((k + 1) * self.period, t_end)
            if t1 > t0:
                n = rng.poisson(self.rate * self.factor ** k * (t1 - t0))
                times.extend(rng.uniform(t0, t1, n).tolist())
            k += 1
        return np.sort(times)
