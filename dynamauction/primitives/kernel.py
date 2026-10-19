"""
Reward kernels modelling how repeated exposures accumulate value.

``RewardKernel`` (abstract): maps (user features, display features, context,
exogenous signal, time) to a reward distribution.
Concrete implementations:
- ``SurvivalKernel``: value decays with the number of prior exposures
  (Cox-style hazard; models ad fatigue / survival of attention).
- ``SaturationKernel``: value saturates as cumulative context grows.
- ``RecencyDisplayKernel``: value depends on elapsed time since last win.
- ``MixedKernel``: weighted combination of multiple kernels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List  

import numpy as np

from dynamauction.primitives.gamma import Gamma, AdditiveGamma, LastWonTimeGamma, ConstantGamma

# ── Reward kernels ─────────────────────────────────────────────────────────────

class RewardKernel(ABC):
    """
    Abstract reward kernel K(u, x, ctx, y(·), t, ·).
    
    The kernel's methods depend on the complete user feature vector `u`, which may
    contain components that are not observable by bidding policies. This allows for
    modeling information asymmetry.
    
    The kernel decomposes (Lebesgue) as:
        K(·, ds) = κ(u, x, ctx, y(t)) δ_0(ds)
                 + λ(u, x, ctx, y(t+s), t, s) ds

    Subclasses must implement:
      kappa              – immediate conversion weight
      lambda_            – delayed conversion intensity
      lambda_upper_bound – upper bound on lambda_ (for Ogata thinning)
    """

    @abstractmethod
    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        """
        Immediate conversion weight κ(u, x, ctx, y(t_i)).
        `u` is the complete user feature vector.
        """
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
        """
        Delayed conversion intensity λ(u, x, ctx, y(t+s), t, s) at delay s.
        `u` is the complete user feature vector.
        """
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
        """
        Upper bound on λ over s ∈ [0, remaining]. Used by Ogata thinning.
        `u` is the complete user feature vector.
        """
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
        `u` is the complete user feature vector.
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

    def _conversion_prob(self, ctx: float) -> float:
        return self.base_rate * np.exp(-ctx / self.saturation_scale)

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return self._conversion_prob(ctx) * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self._conversion_prob(ctx) * self.delay_rate

    def expected_reward(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
    ) -> float:
        """Closed form: p_i is constant in s, so ∫₀^R λ ds = p_i·(1 − e^{−delay_rate·R})."""
        remaining = horizon - t
        if remaining <= 0.0:
            return 0.0
        return self._conversion_prob(ctx) * (1.0 - np.exp(-self.delay_rate * remaining))


class RecencyDisplayKernel(RewardKernel):
    """
    (R) Recency of last display.

    τ    = t_i − ctx  (gap since previous display; treated as ∞ at campaign start)
    κ    = 0
    λ    = (1 − exp(−α·τ)) · delay_rate · exp(−delay_rate · s)
    """

    def __init__(self, recency_coef: float = 0.6, delay_rate: float = 0.5):
        self.delay_rate = delay_rate
        self.recency_coef = recency_coef

    def _conversion_prob(self, ctx: float, t: float) -> float:
        return 1.0 - np.exp(-self.recency_coef * (t - ctx))

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return self._conversion_prob(ctx, t) * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self._conversion_prob(ctx, t) * self.delay_rate

    def expected_reward(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
    ) -> float:
        """Closed form: p_i is constant in s, so ∫₀^R λ ds = p_i·(1 − e^{−delay_rate·R})."""
        remaining = horizon - t
        if remaining <= 0.0:
            return 0.0
        return self._conversion_prob(ctx, t) * (1.0 - np.exp(-self.delay_rate * remaining))


class EffectiveFrequencyKernel(RewardKernel):
    """
    (F) Effective frequency: wear-in, then wear-out, of repeated exposures.

    n    = ctx  (number of past won displays, AdditiveGamma)
    p    = rel(u, x) · base_rate · σ((n − wear_in_mid) / wear_in_slope) · exp(−n / saturation_scale),  σ = logistic
    rel  = clip((w_u⊤u)(w_x⊤x), 0, 1) if w_u and w_x are given (as in SurvivalKernel), else 1
    κ    = 0
    λ    = p · delay_rate · exp(−delay_rate · s)

    The first exposures rarely convert (awareness builds up: wear-in), the next ones convert most, and many exposures
    saturate the user (wear-out). Every won display counts as an exposure, whatever its relevance. A win therefore first
    raises the value of the next displays, so the optimal bid can exceed the myopic value of the display (investment),
    and later lowers it.
    """

    def __init__(
        self,
        base_rate: float = 1.0,
        wear_in_mid: float = 2.0,
        wear_in_slope: float = 0.7,
        saturation_scale: float = 10.0,
        delay_rate: float = 0.5,
        w_u=None,
        w_x=None,
    ):
        self.base_rate = base_rate
        self.wear_in_mid = wear_in_mid
        self.wear_in_slope = wear_in_slope
        self.saturation_scale = saturation_scale
        self.delay_rate = delay_rate
        self.w_u = None if w_u is None else tuple(float(v) for v in w_u)
        self.w_x = None if w_x is None else tuple(float(v) for v in w_x)

    def relevance(self, u: np.ndarray, x: np.ndarray) -> float:
        """clip((w_u⊤u)(w_x⊤x), 0, 1), or 1 without weights."""
        if self.w_u is None or self.w_x is None:
            return 1.0
        return float(np.clip(np.dot(u, self.w_u) * np.dot(x, self.w_x), 0.0, 1.0))

    def _conversion_prob(self, ctx: float) -> float:
        """Exposure effect of the n = ctx past wins (wear-in, then wear-out), without the relevance."""
        wear_in = 1.0 / (1.0 + np.exp(-(ctx - self.wear_in_mid) / self.wear_in_slope))
        return float(np.clip(self.base_rate * wear_in * np.exp(-ctx / self.saturation_scale), 0.0, 1.0))

    def conversion_probability(self, u: np.ndarray, x: np.ndarray, ctx: float) -> float:
        return self.relevance(u, x) * self._conversion_prob(ctx)

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return self.conversion_probability(u, x, ctx) * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self.conversion_probability(u, x, ctx) * self.delay_rate

    def expected_reward(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
    ) -> float:
        """Closed form: p_i is constant in s, so ∫₀^R λ ds = p_i·(1 − e^{−delay_rate·R})."""
        remaining = horizon - t
        if remaining <= 0.0:
            return 0.0
        return self.conversion_probability(u, x, ctx) * (1.0 - np.exp(-self.delay_rate * remaining))


class ReminderKernel(RewardKernel):
    """
    (M) Reminder window: the value of a display depends on the time since the previous won display.

    τ    = t − ctx  (time since the last won display, LastWonTimeGamma; very large before the first win)
    p    = rel(u, x) · [p_cold + (p_peak − p_cold) · (τ / peak_time) · exp(1 − τ / peak_time)]
    rel  = clip((w_u⊤u)(w_x⊤x), 0, 1) if w_u and w_x are given (as in SurvivalKernel), else 1
    κ    = 0
    λ    = p · delay_rate · exp(−delay_rate · s)

    A cold contact converts with p_cold. A reminder shortly after a contact is wasted, one peak_time after it converts
    with p_peak, and the effect fades back to p_cold (the user forgets). Winning a cold display opens a high-value window,
    and winning too early wastes it: the value of a win lies in its timing.
    """

    def __init__(
        self,
        p_cold: float = 0.1,
        p_peak: float = 0.8,
        peak_time: float = 2.0,
        delay_rate: float = 0.5,
        w_u=None,
        w_x=None,
    ):
        self.p_cold = p_cold
        self.p_peak = p_peak
        self.peak_time = peak_time
        self.delay_rate = delay_rate
        self.w_u = None if w_u is None else tuple(float(v) for v in w_u)
        self.w_x = None if w_x is None else tuple(float(v) for v in w_x)

    def relevance(self, u: np.ndarray, x: np.ndarray) -> float:
        """clip((w_u⊤u)(w_x⊤x), 0, 1), or 1 without weights."""
        if self.w_u is None or self.w_x is None:
            return 1.0
        return float(np.clip(np.dot(u, self.w_u) * np.dot(x, self.w_x), 0.0, 1.0))

    def _conversion_prob(self, ctx: float, t: float) -> float:
        """Reminder effect of the time since the last win, without the relevance."""
        r = max(t - ctx, 0.0) / self.peak_time
        return float(np.clip(self.p_cold + (self.p_peak - self.p_cold) * r * np.exp(1.0 - r), 0.0, 1.0))

    def conversion_probability(self, u: np.ndarray, x: np.ndarray, ctx: float, t: float) -> float:
        return self.relevance(u, x) * self._conversion_prob(ctx, t)

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(
        self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float
    ) -> float:
        return self.conversion_probability(u, x, ctx, t) * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float
    ) -> float:
        return self.conversion_probability(u, x, ctx, t) * self.delay_rate

    def expected_reward(
        self,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
    ) -> float:
        """Closed form: p_i is constant in s, so ∫₀^R λ ds = p_i·(1 − e^{−delay_rate·R})."""
        remaining = horizon - t
        if remaining <= 0.0:
            return 0.0
        return self.conversion_probability(u, x, ctx, t) * (1.0 - np.exp(-self.delay_rate * remaining))


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


class SurvivalKernel(RewardKernel):
    """
    Hazard-based survival kernel (Cox-style).
    Each display generates at most one conversion.

    f(x, u, history) = gamma(history) * (U . W_u^T) * (X . W_x^T)
    h(s) = alpha (Exponential baseline hazard) : s ~ Exp(1 / alpha)
    """

    def __init__(
        self,
        w_u: np.ndarray,
        w_x: np.ndarray,
        delay_rate: float = 1.0,
    ):
        self.w_u = np.asarray(w_u, dtype=float)
        self.w_x = np.asarray(w_x, dtype=float)
        self.delay_rate = delay_rate

    def conversion_probability(self, u: np.ndarray, x: np.ndarray, ctx: float) -> float:
        """p(z) = clip(ctx · (w_u⊤u) · (w_x⊤x), 0, 1)."""
        g_u = np.dot(u, self.w_u)
        g_x = np.dot(x, self.w_x)
        return float(np.clip(ctx * g_u * g_x, 0.0, 1.0))

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float) -> float:
        f = self.conversion_probability(u, x, ctx)
        return f * self.delay_rate * np.exp(-self.delay_rate * s)

    def lambda_upper_bound(self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float) -> float:
        f = self.conversion_probability(u, x, ctx)
        return f * self.delay_rate

    def expected_reward(self, u: np.ndarray, x: np.ndarray, ctx: float, y_func: Callable[[float], float], t: float, horizon: float) -> float:
        f = self.conversion_probability(u, x, ctx)
        remaining = horizon - t
        if remaining <= 0.0:
            return 0.0
        return f * (1.0 - np.exp(-self.delay_rate * remaining))


class ImmediateSurvivalKernel(SurvivalKernel):
    """
    Same conversion probability as SurvivalKernel but reward is realized
    immediately at bid time (kappa = f, lambda_ = 0).  Used to diagnose
    whether delayed credit assignment is the bottleneck for RL.
    """

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return self.conversion_probability(u, x, ctx)

    def lambda_(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float) -> float:
        return 0.0

    def expected_reward(self, u: np.ndarray, x: np.ndarray, ctx: float, y_func: Callable[[float], float], t: float, horizon: float) -> float:
        return self.conversion_probability(u, x, ctx)