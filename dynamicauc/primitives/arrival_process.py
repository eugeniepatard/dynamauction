# ── Opportunity arrival ────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from typing import List
import numpy as np

class ArrivalProcess(ABC):
    @abstractmethod
    def sample_times(self, horizon: float, rng: np.random.Generator) -> np.ndarray: ...


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
