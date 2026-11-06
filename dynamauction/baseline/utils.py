"""
Internal helpers for simulation-based planning rollouts.

``_resample_remaining_horizon``: replaces the pre-baked horizon and future
auction schedule of a copied environment with a fresh independent draw,
removing oracle knowledge of the true remaining campaign length (with a fixed
horizon the remaining time stays T − t: only the future auctions are redrawn).
``_per_component_rngs``: gives a rollout copy one random stream per component
(competing bids, displays, conversions), derived from one seed.
``_credited_after_window``: value of the conversions of a rollout still to be
credited when it stops at the end of its planning window.
``_rollout_to_end``: runs a policy to episode termination and returns the
cumulative reward.
``_obs_to_features``: converts a Dict observation to a flat numpy feature
vector for the MLPValuationHead.
"""
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from dynamauction.baseline.policies import BiddingPolicy
from dynamauction.primitives.horizon import FixedHorizon


def _resample_remaining_horizon(env: gym.Env, rng: np.random.Generator) -> None:
    """
    Replace the pre-baked horizon and future auction schedule with a fresh
    independent draw from the scenario's horizon distribution.

    This removes the rollout's implicit knowledge of T: the remaining episode
    length becomes an independent sample from the same distribution as T,
    which is the correct prior when T is memoryless (e.g. Exponential).
    With a fixed horizon the campaign end is known: the remaining time stays T − t
    and only the future auction schedule (and exogenous process) is resampled.

    Must be called on a deepcopy of the live env before a planning rollout.
    """
    unwrapped = env.unwrapped
    t_current = float(unwrapped.state.t)
    step_idx = int(unwrapped.state.step_idx)

    horizon_generator = unwrapped.scenario.horizon_generator
    if isinstance(horizon_generator, FixedHorizon):
        T_remaining = max(float(unwrapped.horizon) - t_current, 0.0)
    else:
        T_remaining = horizon_generator.sample(rng)
    new_horizon = t_current + T_remaining

    unwrapped.horizon = new_horizon

    # Keep already-processed times + current auction; resample future ones (with the rate at the absolute time).
    future_times = unwrapped.scenario.arrival.sample_times_from(t_current, T_remaining, rng)
    unwrapped.auction_times = np.concatenate([
        unwrapped.auction_times[:step_idx + 1],
        future_times,
    ])

    unwrapped.scenario.exogenous.generate(new_horizon, rng)


class _OwnStream:
    """A scenario component (competition or display sampler) that draws from its own generator, whatever generator
    the environment passes to it."""

    def __init__(self, component: Any, rng: np.random.Generator) -> None:
        self._component, self._rng = component, rng

    def sample(self, _rng: Any) -> Any:
        return self._component.sample(self._rng)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._component, name)


class _PerAuctionRNG:
    """Stands for the environment's generator in a rollout copy: the draws of the auction with index k (the
    conversions of its display) come from a generator seeded (seed, k)."""

    def __init__(self, env: Any, seed: int) -> None:
        self._env, self._seed, self._k, self._gen = env, seed, None, None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        k = int(self._env.state.step_idx)
        if k != self._k:
            self._k, self._gen = k, np.random.default_rng([self._seed, k])
        return getattr(self._gen, name)


def _per_component_rngs(env: gym.Env, seed: int) -> None:
    """
    Give a rollout copy one random stream per component, derived from one seed (``MPCPolicy(crn="per_component")``).

    The competing bids and the displays come from their own streams (one draw per auction), and the conversions of
    the auction with index k from a generator seeded (seed, k): rollouts given the same seed share every draw at every
    auction, whatever their earlier wins and losses. Only the copy is changed (its scenario components and its
    generator), never the environment code. Must be called on a deepcopy of the live env.
    """
    unwrapped = env.unwrapped
    competition, display, conversion = np.random.SeedSequence(seed).spawn(3)
    unwrapped.scenario.competition = _OwnStream(unwrapped.scenario.competition, np.random.default_rng(competition))
    unwrapped.scenario.display_sampler = _OwnStream(unwrapped.scenario.display_sampler, np.random.default_rng(display))
    unwrapped.rng = _PerAuctionRNG(unwrapped, int(conversion.generate_state(1)[0]))


def _credited_after_window(env: gym.Env) -> float:
    """
    Value of the conversions already drawn in a rollout that the environment will credit after the current step
    (``MPCPolicy(attribution="campaign")``, when the rollout stops at the end of its planning window).

    The conversions matured since the last auction are credited at the next one, and the pending ones at the first
    auction after them: those up to the last auction of the (simulated) campaign. A conversion after the last auction
    is never credited.
    """
    unwrapped = env.unwrapped
    last_auction = float(unwrapped.auction_times[-1]) if len(unwrapped.auction_times) else float(unwrapped.horizon)
    n_pending = sum(1 for c in unwrapped.pending_conversions if c <= last_auction)
    return float(unwrapped.matured_conversions_reward) + unwrapped.conversion_value * n_pending


def _obs_to_features(obs: Dict[str, Any]) -> np.ndarray:
    """
    Flatten the obs dict into a 1-D float32 vector for MLP input.

    Uses all available observation keys **except** ``expected_reward``,
    which is the regression target and must not appear as an input.
    """
    return np.concatenate([
        obs["t"].ravel(),
        obs["ctx"].ravel(),
        np.atleast_1d(obs["step"]).ravel(),
        obs["user_features"].ravel(),
        obs["display_features"].ravel(),
        obs["y_now"].ravel(),
        obs["cumulative_reward_since_last_step"].ravel(),
        obs["conversion_times_since_last_step"].ravel(),
    ]).astype(np.float32)


def _rollout_to_end(
    env: gym.Env,
    obs: Dict[str, Any],
    policy: BiddingPolicy,
    *,
    first_bid: Optional[float] = None,
    gross_first_step: bool = False,
) -> float:
    """
    Roll *env* (already in its current mid-episode state) to the end,
    using *policy* for every bid.

    Parameters
    ----------
    first_bid : float, optional
        Override the very first bid (used to force a win or loss).
        If None, ``policy.bid(obs)`` is used for the first step too.
    gross_first_step : bool
        If True, add back the first-step payment so the return reflects
        gross conversion revenue rather than net profit.  Required when
        estimating V¹ (we want the value of winning, not the profit).
    """
    total = 0.0
    first = True
    done = False
    while not done:
        bid = first_bid if (first and first_bid is not None) else policy.bid(obs)
        obs, reward, done, _trunc, info = env.step(np.array([bid], dtype=np.float32))
        reward = float(reward)
        if first and gross_first_step:
            reward += float(info["auction"]["payment"])
        total += reward
        first = False
    return total