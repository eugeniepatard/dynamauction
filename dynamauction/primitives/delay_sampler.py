"""
Delay samplers for conversion timing after a won auction.

``DelaySampler`` (abstract): given a ``ConversionProbe``, determines how
many conversions a display generates and samples a delay for each.
Delayed rewards are accumulated by the environment when their deadlines
are reached in subsequent steps.
"""
from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np
from dynamauction.primitives.conversion_probe import ConversionProbe


class DelaySampler(ABC):
    """Decides how many conversions a display can produce."""

    @abstractmethod
    def sample(
        self,
        probe: ConversionProbe,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
        rng: np.random.Generator,
    ) -> List[float]:
        ...


class SingleConversionSampler(DelaySampler):
    """At most one conversion per display. Uses probe.sample_one()."""

    def sample(self, probe, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        return probe.sample_one(u, x, ctx, y_func, t, horizon, rng)


class MultiConversionSampler(DelaySampler):
    """Zero or more conversions per display. Uses probe.sample_all()."""

    def sample(self, probe, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        return probe.sample_all(u, x, ctx, y_func, t, horizon, rng)