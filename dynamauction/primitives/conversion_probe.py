"""
Conversion probe: bridges a RewardKernel and a DelaySampler.

``ConversionProbe`` wraps a ``RewardKernel`` and exposes the sampling
primitives needed by ``DelaySampler`` to decide how many conversions a
display produces and to draw their individual delay times.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from dynamauction.primitives.kernel import RewardKernel


class ConversionProbe(ABC):
    """
    Encapsulates how a kernel converts a display into delays.
    Wraps a RewardKernel and exposes sampling primitives.
    """
    def __init__(self, kernel: RewardKernel):
        self.kernel = kernel

    @abstractmethod
    def sample_one(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func,
        t: float,
        horizon: float,
        rng: np.random.Generator,
    ) -> List[float]:
        """Return a list of 0 or 1 delay (survival semantics)."""
        ...

    @abstractmethod
    def sample_all(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func,
        t: float,
        horizon: float,
        rng: np.random.Generator,
    ) -> List[float]:
        """Return a list of 0..N delays (Poisson/point-process semantics)."""
        ...


class PoissonProbe(ConversionProbe):
    """
    Ogata thinning for general intensity kernels.
    sample_all() returns any number of delays.
    sample_one() returns at most 1 (first accepted event).
    """

    def sample_all(self, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        remaining = horizon - t
        if remaining <= 0.0:
            return []
        lam_max = self.kernel.lambda_upper_bound(u, x, ctx, t, remaining)
        if lam_max <= 0.0:
            return []
        delays: List[float] = []
        s = 0.0
        while s < remaining:
            s += rng.exponential(1.0 / lam_max)
            if s >= remaining:
                break
            lam = self.kernel.lambda_(u, x, ctx, y_func(t + s), t, s)
            if rng.uniform() < lam / lam_max:
                delays.append(s)
        return delays

    def sample_one(self, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        remaining = horizon - t
        if remaining <= 0.0:
            return []
        lam_max = self.kernel.lambda_upper_bound(u, x, ctx, t, remaining)
        if lam_max <= 0.0:
            return []
        s = 0.0
        while s < remaining:
            s += rng.exponential(1.0 / lam_max)
            if s >= remaining:
                break
            lam = self.kernel.lambda_(u, x, ctx, y_func(t + s), t, s)
            if rng.uniform() < lam / lam_max:
                return [s]   # stop at first accepted event
        return []


class BernoulliProbe(ConversionProbe):
    """
    For SurvivalKernel: Bernoulli(p) gate then Exp(alpha) delay.
    sample_one() and sample_all() are identical: returns at most 1 conversion by design.
    alpha is read from the kernel.
    """

    def _draw(self, u, x, ctx, t, horizon, rng) -> List[float]:
        from dynamauction.primitives.kernel import SurvivalKernel
        assert isinstance(self.kernel, SurvivalKernel), (
            "BernoulliProbe requires a SurvivalKernel"
        )
        prob = self.kernel.conversion_probability(u, x, ctx)
        if rng.uniform() >= prob:
            return []
        delay = rng.exponential(1.0 / self.kernel.alpha)
        return [delay] if t + delay <= horizon else []

    def sample_one(self, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        return self._draw(u, x, ctx, t, horizon, rng)

    def sample_all(self, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        return self._draw(u, x, ctx, t, horizon, rng)  # same — survival is always 0/1


class NullProbe(ConversionProbe):
    """Probe that produces no delayed conversions. Use with immediate-reward kernels."""

    def sample_one(self, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        return []

    def sample_all(self, u, x, ctx, y_func, t, horizon, rng) -> List[float]:
        return []