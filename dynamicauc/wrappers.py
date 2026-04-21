import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List

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