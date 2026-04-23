"""
Planning-based baseline bidding policies.

Three algorithms + supporting estimators:

  SimulationEstimator  — Monte Carlo Δ(s, x) = V¹(s, x) − V⁰(s)
  MLPValuationHead     — offline-trained MLP approximating per-display value V₀(s, x)
  GreedyMyopicPolicy   — bids Δ(s, x)  (truthful under second-price)
  ShadingPolicy        — bids α · Δ(s, x)
  MPCPolicy            — optimises the first-step α via H-step CRN rollouts

All three policies expose a ``set_env(env)`` hook that **must** be called once
after each ``env.reset()``::

    obs, _ = env.reset(seed=seed)
    policy.set_env(env)          # store reference to live env for rollouts
    while not done:
        bid   = policy.bid(obs)
        obs, reward, done, _, info = env.step(np.array([bid]))

The env reference is stored (not copied), so subsequent ``env.step()`` calls
advance the live episode while the policy deepcopies the env on demand inside
each ``bid()`` call.

Performance note
----------------
Each ``bid()`` call performs O(n_rollouts) ``copy.deepcopy(env)`` operations.
For long episodes this can be slow; restrict ``n_rollouts`` accordingly or
use the precomputed ``obs["expected_reward"]`` as a fast approximation.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dynamicauc.baseline.utils import _obs_to_features, _rollout_to_end
from dynamicauc.baseline.policies import BiddingPolicy, FiPlaPolicy

_BIG_BID: float = 1e12

class SimulationEstimator:
    """
    Monte Carlo estimator of the incremental display value:

        Δ(s_t, x) = V¹(s_t, x) − V⁰(s_t)

    where

    * **V¹(s_t, x)** is the gross expected reward from step t onward if the
      agent wins the current auction (forced via a very high bid) and then
      follows the continuation policy π₀.
    * **V⁰(s_t)** is the expected reward if the agent loses (bid = 0) and
      then follows π₀.

    The payment at the forced first step is added back when computing V¹ so
    that Δ reflects the value of winning, not the profit net of payment.

    Parameters
    ----------
    pi0 : BiddingPolicy
        Continuation policy used after the forced first step.
        Defaults to :class:`FiPlaPolicy` (truthful / oracle bidder).
    n_rollouts : int
        Number of Monte Carlo trajectories for each V¹ and V⁰ estimate.
    """

    def __init__(
        self,
        pi0: BiddingPolicy = None,
        n_rollouts: int = 20,
    ) -> None:
        self.pi0 = pi0 if pi0 is not None else FiPlaPolicy()
        self.n_rollouts = n_rollouts
        self._env: Optional[gym.Env] = None

    # ── env binding ───────────────────────────────────────────────

    def set_env(self, env: gym.Env) -> None:
        """
        Bind to the live env.  Call once after ``env.reset()``.

        The estimator stores a reference (not a copy), so it always sees the
        env's current mid-episode state when ``estimate()`` is called.
        """
        self._env = env

    # ── estimation ────────────────────────────────────────────────

    def estimate(self, obs: Dict[str, Any]) -> float:
        """
        Return Δ(s_t, x) ≈ mean(V¹) − mean(V⁰) over ``n_rollouts`` trajectories.

        The live env is deepcopied at each call so the running episode is
        never disturbed.
        """
        if self._env is None:
            raise RuntimeError(
                "Call set_env(env) before estimate(). "
                "Planning policies require a bound env."
            )
        v1_sum = v0_sum = 0.0
        for _ in range(self.n_rollouts):
            # V¹: force a win, recover gross value by adding back the payment
            v1_sum += _rollout_to_end(
                copy.deepcopy(self._env), obs, self.pi0,
                first_bid=_BIG_BID, gross_first_step=True,
            )
            # V⁰: force a loss (bid = 0)
            v0_sum += _rollout_to_end(
                copy.deepcopy(self._env), obs, self.pi0,
                first_bid=0.0,
            )
        return (v1_sum - v0_sum) / self.n_rollouts


# ─────────────────────────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    """Two-layer MLP backbone (Tanh activations, scalar output)."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: list = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPValuationHead:
    """
    MLP trained offline to approximate the per-display value:

        V₀(s, x) = E[reward | win display x in state s]

    **Training targets** are the FiPla attribution labels
    ``y_t = obs["expected_reward"]``, which equal the kernel-integrated
    expected conversion value and require no learning by the advertiser.

    The MLP is used by :class:`MPCPolicy` to provide fast value estimates
    inside the CRN planning rollouts without re-running the full kernel
    integration.

    Parameters
    ----------
    input_dim : int
        Length of the feature vector produced by ``_obs_to_features``.
        Automatically inferred by :meth:`from_scenario`.
    hidden_dims : sequence of int
        Hidden layer widths.  Default: ``(64, 64)``.

    Examples
    --------
    Train from a scenario::

        mlp = MLPValuationHead.from_scenario(scenario, n_episodes=200)
        mlp.save("v0.pt")

    Load and use::

        mlp = MLPValuationHead.load("v0.pt")
        mpc = MPCPolicy(v0=mlp, scenario=scenario)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        self.input_dim = input_dim
        self._net = _MLPNet(input_dim, list(hidden_dims))

    # ── training ──────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> "MLPValuationHead":
        """
        Train on pre-collected (feature, label) arrays.

        Parameters
        ----------
        X : ndarray of shape (N, input_dim)
            Feature vectors from ``_obs_to_features``.
        y : ndarray of shape (N,)
            Regression targets — ``obs["expected_reward"]`` from FiPla episodes.
        """
        self._net = self._net.to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                loss = loss_fn(self._net(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._net.eval()
        return self

    @classmethod
    def from_scenario(
        cls,
        scenario,
        n_episodes: int = 100,
        *,
        hidden_dims: Sequence[int] = (64, 64),
        seed: int = 0,
        fit_kwargs: Optional[dict] = None,
    ) -> "MLPValuationHead":
        """
        Collect FiPla rollouts on *scenario* and return a trained MLP.

        FiPla bids the exact kernel-integrated expected reward, which is the
        natural regression target.  Each ``(obs, obs["expected_reward"])``
        pair from a FiPla trajectory becomes one training sample.

        Parameters
        ----------
        scenario : Scenario
            The scenario on which to collect data.
        n_episodes : int
            Number of episodes to roll out.
        seed : int
            Base random seed; episode *i* uses ``seed + i``.
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to :meth:`fit`.
        """
        from dynamicauc.env import AuctionEnv
        env = AuctionEnv(scenario=scenario)
        pi0 = FiPlaPolicy()
        X_list: list = []
        y_list: list = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                X_list.append(_obs_to_features(obs))
                y_list.append(float(obs["expected_reward"].ravel()[0]))
                obs, _, done, _, _ = env.step(
                    np.array([pi0.bid(obs)], dtype=np.float32)
                )
        env.close()

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.float32)
        mlp = cls(input_dim=X.shape[1], hidden_dims=hidden_dims)
        mlp.fit(X, y, **(fit_kwargs or {}))
        return mlp

    # ── inference ─────────────────────────────────────────────────

    def predict(self, obs: Dict[str, Any]) -> float:
        """Return V₀(s, x) as a scalar from an obs dict."""
        feat = torch.tensor(_obs_to_features(obs)).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            return float(self._net(feat).item())

    # ── persistence ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save weights + metadata to *path* (uses torch.save, no extension forced)."""
        torch.save(
            {"input_dim": self.input_dim, "state_dict": self._net.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPValuationHead":
        """Load from a file created by :meth:`save`."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(input_dim=data["input_dim"])
        obj._net.load_state_dict(data["state_dict"])
        obj._net.eval()
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Planning policies
# ─────────────────────────────────────────────────────────────────────────────

class GreedyMyopicPolicy(BiddingPolicy):
    """
    Greedy myopic baseline: bids the incremental value of winning.

        b_t = Δ(s_t, x) = V¹(s_t, x) − V⁰(s_t)

    Under second-price rules this is the truthful bid.  Under first-price
    rules it overbids; use :class:`ShadingPolicy` with α < 1 instead.

    Parameters
    ----------
    estimator : SimulationEstimator
        Provides the Monte Carlo Δ estimates.
    """

    def __init__(self, estimator: SimulationEstimator) -> None:
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
        Shading factor in (0, ∞).  Values in [0.6, 1.0] are typical.
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
        scenario,
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
        self._env: Optional[gym.Env] = None
        self._rng = np.random.default_rng(seed)

    # ── env binding ───────────────────────────────────────────────

    def set_env(self, env: gym.Env) -> None:
        """Bind the live env.  Must be called once after each ``env.reset()``."""
        self._env = env

    # ── bid ───────────────────────────────────────────────────────

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

        # Pre-generate CRN seeds — shared across all α candidates.
        crn_seeds = [int(s) for s in self._rng.integers(0, 2**31, size=self.n_rollouts)]

        q_hat: Dict[float, float] = {}
        for alpha in self.alpha_candidates:
            cumulative = 0.0
            for seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                # Enforce CRN: same noise stream for every α at this seed.
                env_copy.np_random = np.random.default_rng(seed)
                cumulative += self._mpc_rollout(
                    env_copy, observations, alpha, v0_now, t_start
                )
            q_hat[alpha] = cumulative / self.n_rollouts

        alpha_star = max(q_hat, key=q_hat.get)
        return float(max(0.0, alpha_star * v0_now))

    # ── rollout ───────────────────────────────────────────────────

    def _mpc_rollout(
        self,
        env: gym.Env,
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
        # ── first step: apply candidate multiplier ────────────────
        bid_0 = alpha_first * v0_now
        obs, reward, done, _trunc, _ = env.step(np.array([bid_0], dtype=np.float32))
        total = float(reward)

        # ── continuation: apply alpha_cont until horizon or end ───
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
