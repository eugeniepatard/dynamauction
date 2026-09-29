"""
Conversion probe: bridges a RewardKernel and a DelaySampler.

``ConversionProbe`` wraps a ``RewardKernel`` and exposes the sampling
primitives needed by ``DelaySampler`` to decide how many conversions a
display produces and to draw their individual delay times.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from dynamauction.primitives.kernel import (
    RewardKernel,
    SurvivalKernel,
    SaturationKernel,
    RecencyDisplayKernel,
)


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
    Bernoulli(p) gate then Exp(delay_rate) delay.

    Valid for any kernel with kappa=0 and lambda(s) = p * delay_rate * exp(-delay_rate * s),
    where p is constant in s (i.e. lambda_upper_bound is the tight max, attained at s=0).
    Compatible kernels: SurvivalKernel, SaturationKernel, RecencyDisplayKernel.
    sample_one() and sample_all() are identical: returns at most 1 conversion by design.
    """

    _COMPATIBLE_KERNELS = (SurvivalKernel, SaturationKernel, RecencyDisplayKernel)

    def _draw(self, u, x, ctx, t, horizon, rng) -> List[float]:
        assert isinstance(self.kernel, self._COMPATIBLE_KERNELS), (
            f"BernoulliProbe requires a kernel with a constant-in-s conversion "
            f"probability ({', '.join(k.__name__ for k in self._COMPATIBLE_KERNELS)}), "
            f"got {type(self.kernel).__name__}"
        )
        remaining = horizon - t
        if remaining <= 0.0:
            return []
        prob = self.kernel.lambda_upper_bound(u, x, ctx, t, remaining) / self.kernel.delay_rate
        if rng.uniform() >= prob:
            return []
        delay = rng.exponential(1.0 / self.kernel.delay_rate)
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