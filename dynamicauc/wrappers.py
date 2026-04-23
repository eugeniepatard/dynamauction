import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List

from dynamicauc.baseline.policies import BiddingPolicy

"""
Wrappers for the auction environment to connect it to RL algorithms more easily :

- FlatAuctionObsWrapper: flatten the Dict obs space into a 1-D vector.
- MaskExpectedRewardWrapper: remove the "expected_reward" field from the obs space, so that RL agents don't have privileged oracle information.
- BidScaleWrapper: restrict the bid action space to [0, max_bid] for more stable RL training.
- SB3PolicyWrapper: adapt a trained Stable-Baselines3 model to the BiddingPolicy interface
"""

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


class MaskExpectedRewardWrapper(gym.ObservationWrapper):
    """
    Remove ``expected_reward`` from the observation space.

    ``expected_reward`` is the kernel-integrated conversion value
    κ + ∫λ ds — i.e. the exact oracle signal that ``FiPlaPolicy`` bids
    directly.  Exposing it to RL agents gives them privileged information
    a real advertiser would not have, making the RL vs FiPla comparison
    unfair.

    Apply this wrapper **before** ``FlatAuctionObsWrapper`` so that the
    flattened vector seen by the RL policy never contains the oracle value:

        AuctionEnv
        → MaskExpectedRewardWrapper   # ← removes expected_reward
        → FlatAuctionObsWrapper
        → BidScaleWrapper
        → Monitor
        → VecNormalize

    ``FiPlaPolicy`` reads ``obs["expected_reward"]`` from the **raw** env
    (no wrappers) and is unaffected by this wrapper.

    When evaluating a model trained with this wrapper via ``SB3PolicyWrapper``,
    pass ``exclude_keys=["expected_reward"]`` so the evaluation flattening
    matches the training flattening exactly.
    """

    _MASKED_KEY = "expected_reward"

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Dict({
            k: v
            for k, v in env.observation_space.spaces.items()
            if k != self._MASKED_KEY
        })

    def observation(self, obs: dict) -> dict:
        return {k: v for k, v in obs.items() if k != self._MASKED_KEY}


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
    obs_rms : RunningMeanStd, optional
        VecNormalize statistics saved at train time.
    exclude_keys : sequence of str, optional
        Keys to drop before flattening.  Must match the wrappers applied
        during training.  Pass ``["expected_reward"]`` when the model was
        trained with ``MaskExpectedRewardWrapper``.
    clip_obs : float, optional
        Clipping bound applied after VecNormalize standardisation.
        Must match the ``clip_obs`` value used in ``VecNormalize`` during
        training (SB3 default: 10.0).
    """

    def __init__(
        self,
        model,
        raw_obs_space: spaces.Dict,
        obs_rms=None,
        exclude_keys: List[str] = (),
        clip_obs: float = 10.0,
    ) -> None:
        self._model    = model
        self._keys     = [k for k in raw_obs_space.spaces.keys() if k not in exclude_keys]
        self._sp_dict  = raw_obs_space.spaces
        self._obs_rms  = obs_rms
        self._clip_obs = clip_obs

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
                -self._clip_obs, self._clip_obs,
            ).astype(np.float32)
        action, _ = self._model.predict(flat, deterministic=True)
        return float(np.asarray(action).ravel()[0])