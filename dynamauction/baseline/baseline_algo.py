"""Planning-based baseline bidding policies. See docs/API_REFERENCE.md § 8."""
from __future__ import annotations

import bisect
import copy
import itertools
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym
from dynamauction.baseline.estimators import MLPValuationHead, SimulationEstimator
from dynamauction.baseline.policies import BiddingPolicy
from dynamauction.baseline.utils import _resample_remaining_horizon
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
    alpha : float or sequence of (float, float)
        Shading factor in (0, ∞). A sequence of ``(t_from, alpha)`` pairs makes it depend on the time of the auction,
        as ``MPCPolicy.alpha_cont``: the ``alpha`` of the last pair with ``t_from <= t`` (the first pair's before its
        ``t_from``), e.g. ``[(0, 0.3), (10, 0.7)]`` for an arrival rate that changes at t = 10.
    """

    def __init__(
        self,
        estimator: SimulationEstimator,
        alpha: Union[float, Sequence[Tuple[float, float]]] = 1.0,
    ) -> None:
        if isinstance(alpha, (int, float)):
            if alpha <= 0:
                raise ValueError(f"alpha must be positive, got {alpha}")
        else:
            alpha = tuple(sorted((float(t), float(a)) for t, a in alpha))
            if not alpha:
                raise ValueError("alpha schedule must have at least one (t_from, alpha) pair")
            if any(a <= 0 for _, a in alpha):
                raise ValueError(f"alpha must be positive, got {alpha}")
        self.estimator = estimator
        self.alpha = alpha

    def set_env(self, env: gym.Env) -> None:
        """Bind the live env.  Must be called once after each ``env.reset()``."""
        self.estimator.set_env(env)

    def _alpha_at(self, t: float) -> float:
        """Shading factor of an auction at time t (``alpha`` or its schedule)."""
        if isinstance(self.alpha, (int, float)):
            return self.alpha
        alpha = self.alpha[0][1]
        for t_from, a in self.alpha:
            if t >= t_from:
                alpha = a
        return alpha

    def bid(self, observations: Dict[str, Any]) -> float:
        delta = self.estimator.estimate(observations)
        if isinstance(self.alpha, (int, float)):
            return float(max(0.0, self.alpha * delta))
        alpha = self._alpha_at(float(np.asarray(observations["t"]).ravel()[0]))
        return float(max(0.0, alpha * delta))


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
    alpha_cont : float, sequence of (float, float), or "candidate"
        Multiplier used for all continuation steps beyond t. A sequence of ``(t_from, alpha)`` pairs makes it depend
        on the (absolute) time of the continuation auction: the ``alpha`` of the last pair with ``t_from <= t``
        (the first pair's before its ``t_from``), e.g. ``[(0, 0.2), (10, 0.7)]`` for an arrival rate that changes at
        t = 10. ``"candidate"`` uses the candidate being evaluated for every bid of its rollouts (one α shared over
        the planning horizon): a receding-horizon constant shading that re-picks its α at every auction.
        ``"near_far"`` optimises a pair (α_near, α_far): in each rollout the auctions less than ``near_horizon`` after
        the decision time bid α_near · V₀ (the first one included), the later ones up to the planning horizon bid
        α_far · V₀, with α_near ∈ ``alpha_candidates`` and α_far ∈ ``far_candidates``; the policy bids α_near · V₀ now.
        The rollouts of one α_near share their near segment (same CRN seeds and bids), so it is simulated once and
        branched for every α_far: exactly the Q̂ of full rollouts, at a fraction of the cost.
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
        With ``alpha_cont="near_far"`` ties go to the first pair (α_near, then α_far, in candidate order).
    near_horizon : float
        ``alpha_cont="near_far"`` only: length of the near segment of the rollouts.
    far_candidates : sequence of float, optional
        ``alpha_cont="near_far"`` only: candidates for α_far (default: ``alpha_candidates``).
    """

    def __init__(
        self,
        v0: MLPValuationHead,
        alpha_candidates: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.5),
        alpha_cont: Union[float, Sequence[Tuple[float, float]], str] = 1.0,
        n_rollouts: int = 10,
        planning_horizon: float = 7.0,
        seed: int = 0,
        tie_alpha: Optional[float] = None,
        near_horizon: float = 5.0,
        far_candidates: Optional[Sequence[float]] = None,
    ) -> None:
        self.v0 = v0
        self.alpha_candidates = list(alpha_candidates)
        if isinstance(alpha_cont, str):
            if alpha_cont not in ("candidate", "near_far"):
                raise ValueError(f"alpha_cont must be a float, a (t_from, alpha) schedule, 'candidate' or 'near_far', "
                                 f"got {alpha_cont!r}")
            if alpha_cont == "near_far" and tie_alpha is not None:
                raise ValueError("tie_alpha is not supported with alpha_cont='near_far'")
        elif not isinstance(alpha_cont, (int, float)):
            alpha_cont = tuple(sorted((float(t), float(a)) for t, a in alpha_cont))
            if not alpha_cont:
                raise ValueError("alpha_cont schedule must have at least one (t_from, alpha) pair")
        self.alpha_cont = alpha_cont
        self.n_rollouts = n_rollouts
        self.planning_horizon = planning_horizon
        self.tie_alpha = tie_alpha
        self.near_horizon = near_horizon
        self.far_candidates = list(alpha_candidates if far_candidates is None else far_candidates)
        self._env: Optional[AuctionEnv] = None
        self._rng = np.random.default_rng(seed)
        self.alpha_log: List[float] = []
        self.alpha_far_log: List[float] = []   # alpha_cont="near_far": α_far of the selected pair

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
        if self.alpha_cont == "near_far":
            return self._bid_near_far(crn_seeds, v0_now, t_start)

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

    def _bid_near_far(self, crn_seeds: List[Tuple[int, int]], v0_now: float, t_start: float) -> float:
        """alpha_cont="near_far": the pair (α_near, α_far) maximising Q̂ over the same CRN rollouts; bid α_near · V₀."""
        q_hat: Dict[Tuple[float, float], float] = {}
        for alpha_near in self.alpha_candidates:
            sums = [0.0] * len(self.far_candidates)
            for scenario_seed, noise_seed in crn_seeds:
                env_copy = copy.deepcopy(self._env)
                _resample_remaining_horizon(env_copy, np.random.default_rng(scenario_seed))
                env_copy.unwrapped.rng = np.random.default_rng(noise_seed)
                for k, total in enumerate(self._mpc_rollout_near_far(env_copy, alpha_near, v0_now, t_start)):
                    sums[k] += total
            for k, alpha_far in enumerate(self.far_candidates):
                q_hat[(alpha_near, alpha_far)] = sums[k] / self.n_rollouts
        alpha_near, alpha_far = max(q_hat, key=q_hat.get)
        self.alpha_log.append(alpha_near)
        self.alpha_far_log.append(alpha_far)
        return float(max(0.0, alpha_near * v0_now))

    def _mpc_rollout_near_far(self, env: AuctionEnv, alpha_near: float, v0_now: float, t_start: float) -> List[float]:
        """
        One rollout per α_far, sharing the near segment: bids α_near · V₀ while t − t_start < ``near_horizon``,
        then the env is copied once per α_far and each copy bids α_far · V₀ up to the planning horizon.
        Returns the cumulative reward of every branch, in the order of ``far_candidates``.
        """
        obs, reward, done, _trunc, _ = env.step(np.array([alpha_near * v0_now], dtype=np.float32))
        near_total = float(reward)
        while not done:
            t_now = float(obs["t"].ravel()[0])
            if t_now - t_start > self.planning_horizon:
                return [near_total] * len(self.far_candidates)
            if t_now - t_start >= self.near_horizon:
                break
            obs, reward, done, _trunc, _ = env.step(np.array([alpha_near * self.v0.predict(obs)], dtype=np.float32))
            near_total += float(reward)
        if done:
            return [near_total] * len(self.far_candidates)

        v0_far = self.v0.predict(obs)   # first auction of the far segment, the same in every branch
        totals = []
        for alpha_far in self.far_candidates:
            branch = copy.deepcopy(env)
            obs_k, reward, done_k, _trunc, _ = branch.step(np.array([alpha_far * v0_far], dtype=np.float32))
            total = near_total + float(reward)
            while not done_k:
                if float(obs_k["t"].ravel()[0]) - t_start > self.planning_horizon:
                    break
                obs_k, reward, done_k, _trunc, _ = branch.step(
                    np.array([alpha_far * self.v0.predict(obs_k)], dtype=np.float32))
                total += float(reward)
            totals.append(total)
        return totals

    def _alpha_cont_at(self, t: float) -> float:
        """Continuation multiplier of an auction at time t (``alpha_cont`` or its schedule)."""
        if isinstance(self.alpha_cont, (int, float)):
            return self.alpha_cont
        alpha = self.alpha_cont[0][1]
        for t_from, a in self.alpha_cont:
            if t >= t_from:
                alpha = a
        return alpha

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
        Step t+k: bid = alpha_cont  · V₀(s_k,  x_k)  (default continuation; alpha_cont at the time of step k,
                                                     alpha_first for alpha_cont = "candidate")

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
            alpha_k = alpha_first if self.alpha_cont == "candidate" else self._alpha_cont_at(t_now)
            bid_k = alpha_k * self.v0.predict(obs)
            obs, reward, done, _trunc, _ = env.step(
                np.array([bid_k], dtype=np.float32)
            )
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
