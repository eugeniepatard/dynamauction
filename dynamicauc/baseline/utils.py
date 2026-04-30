"""
Internal helpers for simulation-based planning rollouts.

``_resample_remaining_horizon``: replaces the pre-baked horizon and future
auction schedule of a copied environment with a fresh independent draw,
removing oracle knowledge of the true remaining campaign length.
``_rollout_to_end``: runs a policy to episode termination and returns the
cumulative reward.
``_obs_to_features``: converts a Dict observation to a flat numpy feature
vector for the MLPValuationHead.
"""
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from dynamicauc.baseline.policies import BiddingPolicy


def _resample_remaining_horizon(env: gym.Env, rng: np.random.Generator) -> None:
    """
    Replace the pre-baked horizon and future auction schedule with a fresh
    independent draw from the scenario's horizon distribution.

    This removes the rollout's implicit knowledge of T: the remaining episode
    length becomes an independent sample from the same distribution as T,
    which is the correct prior when T is memoryless (e.g. Exponential).

    Must be called on a deepcopy of the live env before a planning rollout.
    """
    unwrapped = env.unwrapped
    t_current = float(unwrapped.state.t)
    step_idx = int(unwrapped.state.step_idx)

    T_remaining = unwrapped.scenario.horizon_generator.sample(rng)
    new_horizon = t_current + T_remaining

    unwrapped.horizon = new_horizon

    # Keep already-processed times + current auction; resample future ones.
    future_times = unwrapped.scenario.arrival.sample_times(T_remaining, rng) + t_current
    unwrapped.auction_times = np.concatenate([
        unwrapped.auction_times[:step_idx + 1],
        future_times,
    ])

    unwrapped.scenario.exogenous.generate(new_horizon, rng)


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