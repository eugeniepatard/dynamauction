from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List  

import numpy as np


# ── Reward kernels ─────────────────────────────────────────────────────────────

class RewardKernel(ABC):
    """
    Abstract reward kernel K(u, x, ctx, y(·), t, ·).

    The kernel decomposes (Lebesgue) as:
        K(·, ds) = κ(u, x, ctx, y(t)) δ_0(ds)
                 + λ(u, x, ctx, y(t+s), t, s) ds

    Subclasses must implement:
      kappa              – immediate conversion weight
      lambda_            – delayed conversion intensity
      lambda_upper_bound – upper bound on lambda_ (for Ogata thinning)
      update_context     – how a won display updates the carry-over context
      initial_context    – context value before any display
    """

    @abstractmethod
    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        """Immediate conversion weight κ(u, x, ctx, y(t_i))."""
        ...

    @abstractmethod
    def lambda_(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y: float,
        t: float,
        s: float,
    ) -> float:
        """Delayed conversion intensity λ(u, x, ctx, y(t+s), t, s) at delay s."""
        ...

    @abstractmethod
    def lambda_upper_bound(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        t: float,
        remaining: float,
    ) -> float:
        """Upper bound on λ over s ∈ [0, remaining]. Used by Ogata thinning."""
        ...

    @abstractmethod
    def update_context(self, ctx: float, x: np.ndarray, t_display: float) -> float:
        """Return updated carry-over context after a won display at t_display."""
        ...

    @abstractmethod
    def initial_context(self) -> float:
        """Initial carry-over context (before any display)."""
        ...

    def expected_reward(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
    ) -> float:
        """
        Expected total reward from display (x, t) up to horizon.
        Computed as κ + ∫₀^{T−t} λ ds via scipy numerical integration.
        """
        from scipy.integrate import quad

        kap = self.kappa(u, x, ctx, y_func(t))
        remaining = horizon - t
        if remaining <= 0.0:
            return kap
        lam_int, _ = quad(
            lambda s: self.lambda_(u, x, ctx, y_func(t + s), t, s),
            0.0,
            remaining,
            limit=100,
        )
        return kap + lam_int


class SaturationKernel(RewardKernel):
    """
    (S) Saturation by cumulative exposure.

    ctx  = number of past won displays.
    κ    = 0
    λ    = base_rate · exp(−ctx / saturation_scale) · delay_rate · exp(−delay_rate · s)
    """

    def __init__(
        self,
        base_rate: float = 1.0,
        saturation_scale: float = 10.0,
        delay_rate: float = 0.5,
    ):
        self.base_rate = base_rate
        self.saturation_scale = saturation_scale
        self.delay_rate = delay_rate

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return (
            self.base_rate
            * np.exp(-ctx / self.saturation_scale)
            * self.delay_rate
            * np.exp(-self.delay_rate * s)
        )

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self.base_rate * np.exp(-ctx / self.saturation_scale) * self.delay_rate

    def update_context(self, ctx: float, x: np.ndarray, t_display: float) -> float:
        return ctx + 1.0

    def initial_context(self) -> float:
        return 0.0


class RecencyDisplayKernel(RewardKernel):
    """
    (R) Recency of last display.

    ctx  = time of the last won display (−inf before any display).
    τ    = t_i − ctx  (gap since previous display; treated as ∞ at campaign start)
    κ    = 0
    λ    = (1 − exp(−α·τ)) · delay_rate · exp(−delay_rate · s)
    """

    def __init__(self, alpha: float = 0.5, delay_rate: float = 0.5):
        self.alpha = alpha
        self.delay_rate = delay_rate

    def _recency_factor(self, ctx: float, t: float) -> float:
        if ctx == float("-inf"):
            return 1.0
        return 1.0 - np.exp(-self.alpha * (t - ctx))

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return self._recency_factor(ctx, t) * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self._recency_factor(ctx, t) * self.delay_rate

    def update_context(self, ctx: float, x: np.ndarray, t_display: float) -> float:
        return t_display

    def initial_context(self) -> float:
        return float("-inf")


class CompetitorExtinctionKernel(RewardKernel):
    """
    (E) Competitor-click extinction.

    ctx    = unused (always 0).
    κ      = 0
    C(t+s) = 1  iff  y(t+s) > t  (a competitor click occurred after display time t)
    λ      = base_rate · delay_rate · exp(−delay_rate · s) · (1 − C(t+s))

    Requires CompetitorClickProcess as the exogenous process.
    """

    def __init__(self, base_rate: float = 1.0, delay_rate: float = 0.5):
        self.base_rate = base_rate
        self.delay_rate = delay_rate

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        if y > t:   # competitor click occurred after display time t and before t + s
            return 0.0
        return self.base_rate * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        # Conservative: assume no competitor click (C = 0 throughout)
        return self.base_rate * self.delay_rate

    def update_context(self, ctx: float, x: np.ndarray, t_display: float) -> float:
        return ctx

    def initial_context(self) -> float:
        return 0.0


class MixedKernel(RewardKernel):
    """
    (D) Mixed: instantaneous + delayed conversions.

    Wraps any base kernel and adds a constant immediate conversion weight κ > 0.
    All delayed behaviour and context management delegate to base_kernel.
    """

    def __init__(self, base_kernel: RewardKernel, kappa_base_rate: float = 0.1):
        self.base_kernel = base_kernel
        self.kappa_base_rate = kappa_base_rate

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return self.kappa_base_rate

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return self.base_kernel.lambda_(u, x, ctx, y, t, s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self.base_kernel.lambda_upper_bound(u, x, ctx, t, remaining)

    def update_context(self, ctx: float, x: np.ndarray, t_display: float) -> float:
        return self.base_kernel.update_context(ctx, x, t_display)

    def initial_context(self) -> float:
        return self.base_kernel.initial_context()
