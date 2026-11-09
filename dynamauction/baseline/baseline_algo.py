"""Planning-based baseline bidding policies. See docs/API_REFERENCE.md § 8."""
from __future__ import annotations

import bisect
import copy
import itertools
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from dynamauction.baseline.estimators import MLPValuationHead, SimulationEstimator
from dynamauction.baseline.policies import BiddingPolicy
from dynamauction.baseline.utils import _credited_after_window, _per_component_rngs, _resample_remaining_horizon
from dynamauction.env import AuctionEnv

# ─────────────────────────────────────────────────────────────────────────────
# Planning policies
# ─────────────────────────────────────────────────────────────────────────────

class GreedyMyopicPolicy(BiddingPolicy):
    """
    Greedy myopic baseline: bids the incremental value of winning.

        b_t = Δ(s_t, x) = V¹(s_t, x) − V⁰(s_t)

    Parameters
    ----------
    estimator : SimulationEstimator
        Provides the Monte Carlo Δ estimates.
    """

    def __init__(self, estimator: SimulationEstimator,) -> None:
        self.estimator = estimator

    def set_env(self, env: gym.Env) -> None:
        """Bind the live env.  Must be called once after each ``env.reset()``."""
        self.estimator.set_env(env)

    def bid(self, observations: Dict[str, Any]) -> float:
        delta = self.estimator.estimate(observations)
        return float(max(0.0, delta))


class ShadingPolicy(BiddingPolicy):
    """
    Bid-shading variant of the greedy baseline:

        b_t = α · Δ(s_t, x)

    α < 1 reduces overpayment (first-price auctions) and acts as a
    regulariser against noise in the Δ estimate.  α = 1 recovers the
    greedy truthful bid.

    Parameters
    ----------
    estimator : SimulationEstimator
        Provides the Monte Carlo Δ estimates.
    alpha : float
        Shading factor in (0, ∞).
    """

    def __init__(self, estimator: SimulationEstimator, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.estimator = estimator
        self.alpha = alpha

    def set_env(self, env: gym.Env) -> None:
        """Bind the live env.  Must be called once after each ``env.reset()``."""
        self.estimator.set_env(env)

    def bid(self, observations: Dict[str, Any]) -> float:
        delta = self.estimator.estimate(observations)
        return float(max(0.0, self.alpha * delta))


class MPCOraclePolicy(BiddingPolicy):
    """
    Oracle variant of MPCPolicy that uses the true pre-sampled auction times.

    The deep-copied env retains the episode's real future auction schedule and
    horizon T, so rollouts see the actual future — this is privileged information
    the policy would not have in practice.  Kept for ablation / upper-bound
    comparison.  For the realistic version see :class:`MPCPolicy`.

    Parameters
    ----------
    v0 : MLPValuationHead
        Pre-trained valuation head; bids are always of the form α · V₀(s, x).
    alpha_candidates : sequence of float
        Discrete set A₀ of first-step multipliers to evaluate.
    alpha_cont : float
        Multiplier used for all continuation steps beyond t.
    n_rollouts : int
        Number of CRN rollout pairs M per decision step.
    planning_horizon : float
        Maximum time H to simulate into the future (in scenario time units).
        Rollouts that reach the natural episode end before H are truncated early.
    seed : int
        Base seed for the CRN stream generator.
    """

    def __init__(
        self,
        v0: MLPValuationHead,
        alpha_candidates: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.5),
        alpha_cont: float = 1.0,
        n_rollouts: int = 10,
        planning_horizon: float = 7.0,
        seed: int = 0,
    ) -> None:
        self.v0 = v0
        self.alpha_candidates = list(alpha_candidates)
        self.alpha_cont = alpha_cont
        self.n_rollouts = n_rollouts
        self.planning_horizon = planning_horizon
        self._env: Optional[AuctionEnv] = None
        self._rng = np.random.default_rng(seed)
        self.alpha_log: List[float] = []

    def set_env(self, env: AuctionEnv) -> None:
        """Bind the live env.  Must be called once after each ``env.reset()``."""
        self._env = env

    def bid(self, observations: Dict[str, Any]) -> float:
        if self._env is None:
            raise RuntimeError(
                "Call set_env(env) before bid(). "
                "MPCOraclePolicy requires a bound env for planning rollouts."
            )

        v0_now = self.v0.predict(observations)
        t_start = float(observations["t"].ravel()[0])

        crn_seeds = [int(s) for s in self._rng.integers(0, 2**31, size=self.n_rollouts)]

        q_hat: Dict[float, float] = {}
        for alpha in self.alpha_candidates:
            cumulative = 0.0
            for seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                env_copy.unwrapped.rng = np.random.default_rng(seed)
                cumulative += self._mpc_rollout(
                    env_copy, alpha, v0_now, t_start
                )
            q_hat[alpha] = cumulative / self.n_rollouts

        alpha_star = max(q_hat, key=q_hat.get)
        self.alpha_log.append(alpha_star)
        return float(max(0.0, alpha_star * v0_now))

    def _mpc_rollout(
        self,
        env: AuctionEnv,
        alpha_first: float,
        v0_now: float,
        t_start: float,
    ) -> float:
        bid_0 = alpha_first * v0_now
        obs, reward, done, _trunc, _ = env.step(np.array([bid_0], dtype=np.float32))
        total = float(reward)

        while not done:
            t_now = float(obs["t"].ravel()[0])
            if t_now - t_start > self.planning_horizon:
                break
            bid_k = self.alpha_cont * self.v0.predict(obs)
            obs, reward, done, _trunc, _ = env.step(
                np.array([bid_k], dtype=np.float32)
            )
            total += float(reward)

        return total


class MPCPolicy(BiddingPolicy):
    """
    Model-Predictive Control (MPC) bidding policy.

    At each auction the policy selects the first-step shading multiplier α*
    by simulating H time-units of future trajectories and picking the α that
    maximises empirical expected reward:

        α*_t = argmax_{α ∈ A₀} Q̂(α)

        Q̂(α) = (1/M) Σ_m R^(m)(α)

    where R^(m)(α) is the cumulative reward of rollout m when bid b_k = α_k · V₀(s_k, x_k),
    with α_first = α at step t and α_k = alpha_cont for all k > t.

    **Common Random Numbers (CRN)**: each rollout m uses two independent seeds —
    ``scenario_seed_m`` for resampling T′ / arrival schedule / exogenous, and
    ``noise_seed_m`` for env step stochasticity.  Both seeds are shared across
    all α candidates, so every α faces the same resampled future and the same
    noise stream.  Using separate seeds keeps the two randomization axes
    orthogonal, matching the design of :class:`SimulationEstimator`.

    Parameters
    ----------
    v0 : MLPValuationHead
        Pre-trained valuation head; bids are always of the form α · V₀(s, x).
    alpha_candidates : sequence of float
        Discrete set A₀ of first-step multipliers to evaluate.
    alpha_cont : float
        Multiplier used for all continuation steps beyond t.
    n_rollouts : int
        Number of CRN rollout pairs M per decision step.
    planning_horizon : float
        Maximum time H to simulate into the future (in scenario time units).
        Rollouts that reach the natural episode end before H are truncated early.
    seed : int
        Base seed for the CRN stream generator.
    tie_alpha : float, optional
        When several candidates reach the maximal Q̂ (frequent: with CRN in a
        second-price auction, candidates that produce the same first-step outcome
        in every rollout get identical Q̂), pick the tied candidate closest to
        ``tie_alpha``.  ``None`` keeps the first candidate in ``alpha_candidates``.
    crn : {"shared_stream", "per_component"}
        How the noise seed drives the rollout.  ``"shared_stream"`` (default): one
        generator for the competing bids, the displays and the conversions, drawn in
        the order in which they occur; a won display draws its conversion and a lost
        one does not, so after the first auction at which two candidates' outcomes
        differ their later draws are shifted.  ``"per_component"``: one stream for the
        competing bids, one for the displays and one generator per auction index for
        the conversions, set on each rollout copy (``_per_component_rngs``), so that all
        candidates share every draw at every auction.  Both are unbiased;
        ``"per_component"`` reduces the noise of the comparison between candidates.
    attribution : {"window", "campaign"}
        Which conversions a rollout counts.  ``"window"`` (default): those credited
        within the planning window (the rollout stops before the first auction after
        t + H).  ``"campaign"``: also the conversions of the window's displays that
        arrive after the window but are still credited in the simulated campaign
        (before its last auction), so that a win is valued for all its conversions.
    reward : {"simulated", "v0"}
        What a simulated auction is worth.  ``"simulated"`` (default): the environment's
        step reward, i.e. the simulated conversions (counted as set by ``attribution``)
        minus the payments.  ``"v0"``: V₀(s_k, x_k) − payment if the auction is won and 0
        otherwise, V₀ being the prediction used for the bid; the simulated conversions
        are not counted (their expected value, V₀, replaces them).  ``attribution`` then
        does not apply and must stay ``"window"``.
    """

    def __init__(
        self,
        v0: MLPValuationHead,
        alpha_candidates: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.5),
        alpha_cont: float = 1.0,
        n_rollouts: int = 10,
        planning_horizon: float = 7.0,
        seed: int = 0,
        tie_alpha: Optional[float] = None,
        crn: str = "shared_stream",
        attribution: str = "window",
        reward: str = "simulated",
    ) -> None:
        if crn not in ("shared_stream", "per_component"):
            raise ValueError(f"crn must be 'shared_stream' or 'per_component', got {crn!r}")
        if attribution not in ("window", "campaign"):
            raise ValueError(f"attribution must be 'window' or 'campaign', got {attribution!r}")
        if reward not in ("simulated", "v0"):
            raise ValueError(f"reward must be 'simulated' or 'v0', got {reward!r}")
        if reward == "v0" and attribution != "window":
            raise ValueError("attribution only applies to reward='simulated' (with reward='v0' a win is worth V0)")
        self.v0 = v0
        self.alpha_candidates = list(alpha_candidates)
        self.alpha_cont = alpha_cont
        self.n_rollouts = n_rollouts
        self.planning_horizon = planning_horizon
        self.tie_alpha = tie_alpha
        self.crn = crn
        self.attribution = attribution
        self.reward = reward
        self._env: Optional[AuctionEnv] = None
        self._rng = np.random.default_rng(seed)
        self.alpha_log: List[float] = []

    def set_env(self, env: AuctionEnv) -> None:
        """Bind the live env.  Must be called once after each ``env.reset()``."""
        self._env = env

    def bid(self, observations: Dict[str, Any]) -> float:
        """
        Select α* via CRN rollouts and return b_t = α* · V₀(s_t, x_t).
        """
        if self._env is None:
            raise RuntimeError(
                "Call set_env(env) before bid(). "
                "MPCPolicy requires a bound env for planning rollouts."
            )

        # predict() uses _obs_to_features() to remove "expected_reward" from the input features, so it's safe to pass the raw obs.
        v0_now = self.v0.predict(observations)
        t_start = float(observations["t"].ravel()[0])

        # Two independent seeds per rollout — keeps scenario (T′, arrivals, exogenous)
        # and env step noise on orthogonal RNG axes, matching SimulationEstimator design.
        # Both seeds are shared across all α candidates (CRN holds).
        crn_seeds = [
            (int(self._rng.integers(0, 2**31 - 1)), int(self._rng.integers(0, 2**31 - 1)))
            for _ in range(self.n_rollouts)
        ]

        q_hat: Dict[float, float] = {}
        for alpha in self.alpha_candidates:
            cumulative = 0.0
            for scenario_seed, noise_seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                _resample_remaining_horizon(env_copy, np.random.default_rng(scenario_seed))
                env_copy.unwrapped.rng = np.random.default_rng(noise_seed)
                if self.crn == "per_component":
                    _per_component_rngs(env_copy, noise_seed)
                cumulative += self._mpc_rollout(
                    env_copy, alpha, v0_now, t_start
                )
            q_hat[alpha] = cumulative / self.n_rollouts

        alpha_star = self._select_alpha(q_hat)
        self.alpha_log.append(alpha_star)
        return float(max(0.0, alpha_star * v0_now))

    def _select_alpha(self, q_hat: Dict[float, float]) -> float:
        """argmax of Q̂; ties go to the first candidate, or to the one closest to ``tie_alpha``."""
        alpha_star = max(q_hat, key=q_hat.get)
        if self.tie_alpha is not None:
            tied = [a for a, q in q_hat.items() if q == q_hat[alpha_star]]
            alpha_star = min(tied, key=lambda a: abs(a - self.tie_alpha))
        return alpha_star

    def _mpc_rollout(
        self,
        env: AuctionEnv,
        alpha_first: float,
        v0_now: float,
        t_start: float,
    ) -> float:
        """
        Simulate one trajectory from the current state.

        Step t  : bid = alpha_first · V₀(s_t, x_t)  (the candidate being evaluated)
        Step t+k: bid = alpha_cont  · V₀(s_k,  x_k)  (default continuation)

        Stops at the natural episode end or when the elapsed time exceeds
        ``planning_horizon``; with ``attribution="campaign"``, a stop at the end of the
        window adds the conversions still to be credited (``_credited_after_window``).
        Each simulated auction is valued by ``_step_value`` (``reward``).
        """
        bid_0 = alpha_first * v0_now
        obs, reward, done, _trunc, info = env.step(np.array([bid_0], dtype=np.float32))
        total = self._step_value(reward, info, v0_now)

        while not done:
            t_now = float(obs["t"].ravel()[0])
            if t_now - t_start > self.planning_horizon:
                if self.attribution == "campaign":
                    total += _credited_after_window(env)
                break
            v0_k = self.v0.predict(obs)
            obs, reward, done, _trunc, info = env.step(
                np.array([self.alpha_cont * v0_k], dtype=np.float32)
            )
            total += self._step_value(reward, info, v0_k)

        return total

    def _step_value(self, reward: float, info: Dict[str, Any], v0: float) -> float:
        """Value of one simulated auction: the environment's step reward (``reward="simulated"``), or V₀ minus the
        payment if the auction is won and 0 otherwise (``reward="v0"``)."""
        if self.reward == "simulated":
            return float(reward)
        auction = info["auction"]
        return float(v0) - float(auction["payment"]) if auction["won"] else 0.0

class MPCTimeBinnedPolicy(BiddingPolicy):
    """
    Time-binned MPC (V3): jointly optimises K shading multipliers over the
    planning window [t, t+H), one per equal-width time bin.

    The window is split into K bins:
        I_j = [t + j·H/K,  t + (j+1)·H/K)   for j = 0, …, K-1

    Every auction k in a rollout is assigned to bin j(k) = floor((t_k − t) / (H/K))
    and bids:
        b_k = α_{j(k)} · V₀(s_k, x_k)

    The policy grid-searches all α ∈ A₀^K (|A₀|^K candidates), picks
        α* = argmax_{α} Q̂(α)
    executes b_t = α*₀ · V₀(s_t, x_t), and re-plans at the next auction.

    CRN holds: all candidate vectors share the same M rollouts, each defined by
    an independent scenario seed (T′, arrivals, exogenous) and noise seed (env
    step stochasticity) — two orthogonal RNG axes per rollout.

    Special case: K = 1 is equivalent to MPCPolicy where α_cont = α₀.

    **Non-uniform bins** (``bin_edges``): the bin boundaries can be given explicitly, e.g. ``bin_edges=(1.5,)`` gives
    a short first bin [t, t+1.5) and one bin for the rest [t+1.5, t+H), so the policy can express "wait now (low
    α₀ for the next 1.5 time units), bid later (α₁)" at the cost of K = 2. ``bin_edges=(0.0,)`` puts only the
    current auction in the first bin: α₀ is the current bid multiplier and α₁ the continuation multiplier, i.e.
    :class:`MPCPolicy` with α_cont optimised jointly at every decision.

    Parameters
    ----------
    v0 : MLPValuationHead
    alpha_candidates : sequence of float
        Grid A₀ searched independently per bin. Total grid size: |A₀|^K.
        Keep K ≤ 3 with a coarse grid to control cost.
    n_bins : int
        Number of time bins K.
    n_rollouts : int
        Number of CRN rollouts M per candidate vector.
    planning_horizon : float
        H in scenario time units.
    seed : int
    bin_edges : sequence of float, optional
        Interior bin boundaries, in time elapsed since the decision (increasing, ≥ 0); bin j covers
        [bin_edges[j-1], bin_edges[j]). Requires ``n_bins == len(bin_edges) + 1``. None (default): K equal bins
        of width H/K.
    """

    def __init__(
        self,
        v0: MLPValuationHead,
        alpha_candidates: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.5),
        n_bins: int = 2,
        n_rollouts: int = 10,
        planning_horizon: float = 7.0,
        seed: int = 0,
        bin_edges: Optional[Sequence[float]] = None,
    ) -> None:
        if bin_edges is not None:
            bin_edges = [float(e) for e in bin_edges]
            if len(bin_edges) + 1 != n_bins:
                raise ValueError(f"n_bins={n_bins} but bin_edges={bin_edges} defines {len(bin_edges) + 1} bins")
            if bin_edges[0] < 0 or any(b <= a for a, b in zip(bin_edges, bin_edges[1:])):
                raise ValueError(f"bin_edges must be increasing and non-negative, got {bin_edges}")
        self.bin_edges = bin_edges
        self.v0 = v0
        self.alpha_candidates = list(alpha_candidates)
        self.n_bins = n_bins
        self.n_rollouts = n_rollouts
        self.planning_horizon = planning_horizon
        self._env: Optional[AuctionEnv] = None
        self._rng = np.random.default_rng(seed)
        self.alpha_log: List[Tuple[float, ...]] = []

    def set_env(self, env: AuctionEnv) -> None:
        """Bind the live env. Must be called once after each env.reset()."""
        self._env = env

    def bid(self, observations: Dict[str, Any]) -> float:
        """
        Grid-search A₀^K, return b_t = α*₀ · V₀(s_t, x_t).
        """
        if self._env is None:
            raise RuntimeError("Call set_env(env) before bid().")

        v0_now    = self.v0.predict(observations)
        t_start   = float(observations["t"].ravel()[0])
        bin_width = self.planning_horizon / self.n_bins

        crn_seeds = [
            (int(self._rng.integers(0, 2**31 - 1)), int(self._rng.integers(0, 2**31 - 1)))
            for _ in range(self.n_rollouts)
        ]

        q_hat: Dict[Tuple[float, ...], float] = {}
        for alpha_vec in itertools.product(self.alpha_candidates, repeat=self.n_bins):
            cumulative = 0.0
            for scenario_seed, noise_seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                _resample_remaining_horizon(env_copy, np.random.default_rng(scenario_seed))
                env_copy.unwrapped.rng = np.random.default_rng(noise_seed)
                cumulative += self._rollout(
                    env_copy, observations, alpha_vec, v0_now, t_start, bin_width
                )
            q_hat[alpha_vec] = cumulative / self.n_rollouts

        alpha_star = max(q_hat, key=q_hat.get)
        self.alpha_log.append(alpha_star)
        return float(max(0.0, alpha_star[0] * v0_now))

    def _bin_index(self, elapsed: float, bin_width: float) -> int:
        """Bin of an auction ``elapsed`` time units after the decision (equal bins unless ``bin_edges``)."""
        if self.bin_edges is None:
            return min(int(elapsed / bin_width), self.n_bins - 1)
        return bisect.bisect_right(self.bin_edges, elapsed)

    def _rollout(
        self,
        env: AuctionEnv,
        obs: Dict[str, Any],
        alpha_vec: Tuple[float, ...],
        v0_now: float,
        t_start: float,
        bin_width: float,
    ) -> float:
        # First step is always in bin 0.
        obs, reward, done, _trunc, _ = env.step(
            np.array([alpha_vec[0] * v0_now], dtype=np.float32)
        )
        total = float(reward)

        while not done:
            t_now   = float(obs["t"].ravel()[0])
            elapsed = t_now - t_start
            if elapsed >= self.planning_horizon:
                break
            j     = self._bin_index(elapsed, bin_width)
            bid_k = alpha_vec[j] * self.v0.predict(obs)
            obs, reward, done, _trunc, _ = env.step(
                np.array([bid_k], dtype=np.float32)
            )
            total += float(reward)

        return total
