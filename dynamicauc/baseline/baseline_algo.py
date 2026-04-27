"""Planning-based baseline bidding policies. See docs/API_REFERENCE.md § 8."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from dynamicauc.baseline.estimators import MLPValuationHead, SimulationEstimator
from dynamicauc.baseline.policies import BiddingPolicy
from dynamicauc.env import AuctionEnv

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

    **Common Random Numbers (CRN)**: M noise streams (arriving competing bids,
    conversions, delays) are pre-fixed via seeds shared across all α candidates.
    This eliminates inter-candidate variance from common stochastic factors and
    makes the comparison of candidates meaningful with few rollouts.

    Parameters
    ----------
    v0 : MLPValuationHead
        Pre-trained valuation head; bids are always of the form α · V₀(s, x).
    scenario : Scenario
        Needed to create fresh envs for CRN rollouts (not stored after init;
        the live env reference is stored via set_env).
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
        """
        Select α* via CRN rollouts and return b_t = α* · V₀(s_t, x_t).
        """
        if self._env is None:
            raise RuntimeError(
                "Call set_env(env) before bid(). "
                "MPCPolicy requires a bound env for planning rollouts."
            )

        v0_now = self.v0.predict(observations)
        t_start = float(observations["t"].ravel()[0])

        crn_seeds = [int(s) for s in self._rng.integers(0, 2**31, size=self.n_rollouts)]

        q_hat: Dict[float, float] = {}
        for alpha in self.alpha_candidates:
            cumulative = 0.0
            for seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                # Enforce CRN: same noise stream for every α at this seed.
                env_copy.rng = np.random.default_rng(seed)
                cumulative += self._mpc_rollout(
                    env_copy, observations, alpha, v0_now, t_start
                )
            q_hat[alpha] = cumulative / self.n_rollouts

        alpha_star = max(q_hat, key=q_hat.get)
        self.alpha_log.append(alpha_star)
        return float(max(0.0, alpha_star * v0_now))

    def _mpc_rollout(
        self,
        env: AuctionEnv,
        obs: Dict[str, Any],
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


class MPCDeltaPolicy(BiddingPolicy):
    """
    MPC variant that uses SimulationEstimator's delta as the base value
    instead of a pre-trained MLPValuationHead.

    At each auction the policy:
    1. Estimates delta = V¹(s, x) − V⁰(s) via SimulationEstimator.
    2. For each alpha in alpha_candidates, runs M CRN rollouts bidding
       alpha · delta at the current step.
    3. Picks alpha* that maximises empirical cumulative reward.
    4. Returns alpha* · delta.

    Continuation steps inside rollouts use alpha_cont · expected_reward
    (shaded FiPla) to avoid nested deepcopies of the SimulationEstimator.

    Parameters
    ----------
    estimator : SimulationEstimator
        Used to compute delta at each bid decision.
    alpha_candidates : sequence of float
        Shading multipliers to evaluate at the current step.
    alpha_cont : float
        Shading factor applied to expected_reward in continuation steps.
    n_rollouts : int
        Number of CRN rollout pairs M per decision step.
    planning_horizon : float
        Maximum time H to simulate into the future.
    seed : int
        Base seed for the CRN stream generator.
    """

    def __init__(
        self,
        estimator: SimulationEstimator,
        alpha_candidates: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.5),
        alpha_cont: float = 1.0,
        n_rollouts: int = 10,
        planning_horizon: float = 7.0,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.alpha_candidates = list(alpha_candidates)
        self.alpha_cont = alpha_cont
        self.n_rollouts = n_rollouts
        self.planning_horizon = planning_horizon
        self._env: Optional[AuctionEnv] = None
        self._rng = np.random.default_rng(seed)
        self.alpha_log: List[float] = []

    def set_env(self, env: AuctionEnv) -> None:
        """Bind the live env and the estimator. Call once after each env.reset()."""
        self._env = env
        self.estimator.set_env(env)

    def bid(self, observations: Dict[str, Any]) -> float:
        if self._env is None:
            raise RuntimeError("Call set_env(env) before bid().")

        delta_now = self.estimator.estimate(observations)
        if delta_now <= 0.0:
            self.alpha_log.append(0.0)
            return 0.0

        t_start   = float(observations["t"].ravel()[0])
        crn_seeds = [int(s) for s in self._rng.integers(0, 2**31, size=self.n_rollouts)]

        q_hat: Dict[float, float] = {}
        for alpha in self.alpha_candidates:
            cumulative = 0.0
            for seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                env_copy.rng = np.random.default_rng(seed)
                cumulative += self._rollout(env_copy, observations, alpha, delta_now, t_start)
            q_hat[alpha] = cumulative / self.n_rollouts

        alpha_star = max(q_hat, key=q_hat.get)
        self.alpha_log.append(alpha_star)
        return float(max(0.0, alpha_star * delta_now))

    def _rollout(
        self,
        env: AuctionEnv,
        obs: Dict[str, Any],
        alpha_first: float,
        delta_now: float,
        t_start: float,
    ) -> float:
        obs, reward, done, _trunc, _ = env.step(
            np.array([alpha_first * delta_now], dtype=np.float32)
        )
        total = float(reward)

        # Cost: n_rollouts_outer × planning_steps × n_rollouts_estimator deepcopies.
        cont_estimator = copy.deepcopy(self.estimator)
        cont_estimator.set_env(env)

        while not done:
            t_now = float(obs["t"].ravel()[0])
            if t_now - t_start > self.planning_horizon:
                break
            delta_k = cont_estimator.estimate(obs)
            bid_k = self.alpha_cont * float(max(0.0, delta_k))
            obs, reward, done, _trunc, _ = env.step(
                np.array([bid_k], dtype=np.float32)
            )
            total += float(reward)

        return total
