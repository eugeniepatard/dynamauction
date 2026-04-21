import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List

from dynamicauc.baseline.policies import BiddingPolicy

"""
Wrappers for dynamicauc gym environments.
"""

# TODO : Ajouter une partie sur les wrappers dans la doc et les tests unitaires.

# ── Observation wrapper ────────────────────────────────────────────────────────

class FlatAuctionObsWrapper(gym.ObservationWrapper):
    """
    Flatten the Dict observation space into a 1-D float32 vector.
    All Box fields are concatenated in insertion order.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        orig = env.observation_space
        self._keys: List[str] = list(orig.spaces.keys())

        dim = sum(
            1 if isinstance(sp, spaces.Discrete) else int(np.prod(sp.shape))
            for sp in orig.spaces.values()
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        parts = []
        for key in self._keys:
            v  = obs[key]
            sp = self.env.observation_space.spaces[key]
            if isinstance(sp, spaces.Discrete):
                parts.append(np.array([float(v)], dtype=np.float32))
            else:
                parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts)


# ── Action wrapper ─────────────────────────────────────────────────────────────

class BidScaleWrapper(gym.ActionWrapper):
    """
    Restrict the bid action space to [0, max_bid].

    The underlying env has action_space = Box([0, 1e12]), which is valid but
    too wide for stable RL training.  
    This wrapper clips the bid to a reasonable range: [0, max_bid].
    """

    def __init__(self, env: gym.Env, max_bid: float = 5.0) -> None:
        super().__init__(env)
        self._max_bid = max_bid
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(max_bid),
            shape=(1,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, 0.0, self._max_bid).astype(np.float32)

# ── Policy wrapper ─────────────────────────────────────────────────────────────

class SB3PolicyWrapper(BiddingPolicy):
    """
    Adapt a trained Stable-Baselines3 model to the ``BiddingPolicy`` interface.

    The model was trained with ``FlatAuctionObsWrapper`` applied; this class
    replicates that flattening exactly so that we can evaluate in a raw-Dict env.

    Parameters
    ----------
    model : SB3 model (PPO / SAC / TD3)
    raw_obs_space : gymnasium.spaces.Dict
        The *original* observation space of ``AuctionEnv`` (before any wrapper).
    """

    def __init__(self, model, raw_obs_space: spaces.Dict, obs_rms=None) -> None:
        self._model   = model
        self._keys    = list(raw_obs_space.spaces.keys())
        self._sp_dict = raw_obs_space.spaces
        self._obs_rms = obs_rms  # RunningMeanStd from VecNormalize, or None

    # ------------------------------------------------------------------
    def _flatten(self, obs: Dict[str, Any]) -> np.ndarray:
        """Mirror FlatAuctionObsWrapper.observation() exactly."""
        parts = []
        for key in self._keys:
            v  = obs[key]
            sp = self._sp_dict[key]
            if isinstance(sp, spaces.Discrete):
                parts.append(np.array([float(v)], dtype=np.float32))
            else:
                parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts)

    def bid(self, observations: Dict[str, Any]) -> float:
        flat = self._flatten(observations)
        if self._obs_rms is not None:
            flat = np.clip(
                (flat - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8),
                -10.0, 10.0,
            ).astype(np.float32)
        action, _ = self._model.predict(flat, deterministic=True)
        return float(np.asarray(action).ravel()[0])