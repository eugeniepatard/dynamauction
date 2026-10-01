"""Planning-based baseline bidding policies. See docs/API_REFERENCE.md § 8."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from dynamauction.baseline.estimators import MLPValuationHead, SimulationEstimator
from dynamauction.baseline.policies import BiddingPolicy
from dynamauction.baseline.utils import _resample_remaining_horizon
from dynamauction.env import AuctionEnv
from dynamauction.primitives.auctions import SecondPriceAuction

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
    ) -> None:
        self.v0 = v0
        self.alpha_candidates = list(alpha_candidates)
        self.alpha_cont = alpha_cont
        self.n_rollouts = n_rollouts
        self.planning_horizon = planning_horizon
        self.tie_alpha = tie_alpha
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
        ``planning_horizon``.
        """
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


class MPCWinLossPolicy(MPCPolicy):
    """
    MPC variant for second-price auctions that integrates the first competing bid with many draws.

    In a second-price auction a rollout depends on α only through whether the first bid
    b = α·V₀ beats the competing bid B (the payment is B, the next state only depends on
    win / loss).  With W the rollout reward when the first auction is won (payment
    excluded) and L the reward when it is lost:

        Q(α) = F(b)·W − E[B·1{B < b}] + (1 − F(b))·L,      F = CDF of B.

    W and L are estimated with ``n_rollouts`` CRN rollouts each (first bid forced to win,
    resp. lose; continuation bids α_cont·V₀ as in :class:`MPCPolicy`).  F and
    E[B·1{B < b}] are estimated with ``n_first_draws`` samples of the scenario's competition
    distribution, so candidates are separated even when no rollout's B falls between their
    bids.  Cost per decision: 2·n_rollouts rollouts instead of len(alpha_candidates)·n_rollouts.

    Parameters
    ----------
    n_first_draws : int
        Number of competing-bid samples used for the first auction.
    Other parameters: see :class:`MPCPolicy`.
    """

    _FORCE_WIN_BID = 1e9

    def __init__(self, *args, n_first_draws: int = 1000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_first_draws = n_first_draws
        self.win_loss_log: List[Tuple[float, float]] = []

    def bid(self, observations: Dict[str, Any]) -> float:
        if self._env is None:
            raise RuntimeError("Call set_env(env) before bid().")
        scenario = self._env.unwrapped.scenario
        if not isinstance(scenario.auction, SecondPriceAuction):
            raise ValueError("MPCWinLossPolicy assumes a second-price auction (payment = competing bid).")

        v0_now = self.v0.predict(observations)
        t_start = float(observations["t"].ravel()[0])
        crn_seeds = [
            (int(self._rng.integers(0, 2**31 - 1)), int(self._rng.integers(0, 2**31 - 1)))
            for _ in range(self.n_rollouts)
        ]
        draw_rng = np.random.default_rng(int(self._rng.integers(0, 2**31 - 1)))

        win_values, lose_values = [], []
        for scenario_seed, noise_seed in crn_seeds:
            for first_bid, values in ((self._FORCE_WIN_BID, win_values), (0.0, lose_values)):
                env_copy = copy.deepcopy(self._env)
                _resample_remaining_horizon(env_copy, np.random.default_rng(scenario_seed))
                env_copy.unwrapped.rng = np.random.default_rng(noise_seed)
                values.append(self._rollout_from_first_bid(env_copy, first_bid, t_start))
        w_hat, l_hat = float(np.mean(win_values)), float(np.mean(lose_values))

        b_draws = np.array([scenario.competition.sample(draw_rng) for _ in range(self.n_first_draws)])
        q_hat: Dict[float, float] = {}
        for alpha in self.alpha_candidates:
            won = b_draws < alpha * v0_now
            p_win = float(won.mean())
            q_hat[alpha] = p_win * w_hat - float((b_draws * won).mean()) + (1.0 - p_win) * l_hat

        alpha_star = self._select_alpha(q_hat)
        self.alpha_log.append(alpha_star)
        self.win_loss_log.append((w_hat, l_hat))
        return float(max(0.0, alpha_star * v0_now))

    def _rollout_from_first_bid(self, env: AuctionEnv, first_bid: float, t_start: float) -> float:
        """Rollout reward with the first payment added back (W or L above)."""
        obs, reward, done, _trunc, info = env.step(np.array([first_bid], dtype=np.float32))
        total = float(reward) + float(info["auction"]["payment"])

        while not done:
            t_now = float(obs["t"].ravel()[0])
            if t_now - t_start > self.planning_horizon:
                break
            bid_k = self.alpha_cont * self.v0.predict(obs)
            obs, reward, done, _trunc, _ = env.step(np.array([bid_k], dtype=np.float32))
            total += float(reward)

        return total


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
    """

    def __init__(
        self,
        v0: MLPValuationHead,
        alpha_candidates: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.5),
        n_bins: int = 2,
        n_rollouts: int = 10,
        planning_horizon: float = 7.0,
        seed: int = 0,
    ) -> None:
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
            j     = min(int(elapsed / bin_width), self.n_bins - 1)
            bid_k = alpha_vec[j] * self.v0.predict(obs)
            obs, reward, done, _trunc, _ = env.step(
                np.array([bid_k], dtype=np.float32)
            )
            total += float(reward)

        return total
