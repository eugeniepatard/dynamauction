from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List  

import numpy as np

from dynamicauc.primitives.gamma import Gamma, AdditiveGamma, LastWonTimeGamma, ConstantGamma

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


class RecencyDisplayKernel(RewardKernel):
    """
    (R) Recency of last display.

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
    h(s) = alpha (Exponential baseline hazard)
    """

    def __init__(self, w_u: np.ndarray, w_x: np.ndarray, alpha: float = 1.0):
        self.w_u = np.asarray(w_u, dtype=float)
        self.w_x = np.asarray(w_x, dtype=float)
        self.alpha = alpha

    def conversion_probability(self, u: np.ndarray, x: np.ndarray, ctx: float) -> float:
        """Compute conversion scaling factor f(x, u, history) and clip to [0, 1]."""
        g_u = np.dot(u, self.w_u)
        g_x = np.dot(x, self.w_x)
        # ctx corresponds to gamma(history) factor which inherently models recency/saturation
        prob = ctx * g_u * g_x
        return float(np.clip(prob, 0.0, 1.0))

    def sample_survival(
        self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, horizon: float, rng: np.random.Generator
    ) -> List[float]:
        """Sample at most 1 conversion delay based on the survival formulation."""
        prob = self.conversion_probability(u, x, ctx)
        if rng.uniform() < prob:
            delay = rng.exponential(1.0 / self.alpha) # TODO : préciser si alpha est E(X) ou 1/E(X)
            if t + delay <= horizon:
                return [delay]
        return []

    def kappa(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float) -> float:
        return 0.0

    def lambda_(self, u: np.ndarray, x: np.ndarray, ctx: float, y: float, t: float, s: float) -> float:
        f = self.conversion_probability(u, x, ctx)
        return f * self.alpha * np.exp(-self.alpha * s)

    def lambda_upper_bound(self, u: np.ndarray, x: np.ndarray, ctx: float, t: float, remaining: float) -> float:
        f = self.conversion_probability(u, x, ctx)
        return f * self.alpha

    def expected_reward(self, u: np.ndarray, x: np.ndarray, ctx: float, y_func: Callable[[float], float], t: float, horizon: float) -> float:
        f = self.conversion_probability(u, x, ctx)
        remaining = horizon - t
        if remaining <= 0.0:
            return 0.0
        return f * (1.0 - np.exp(-self.alpha * remaining))