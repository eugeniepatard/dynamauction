from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence
from dynamicauc.baseline.policies import BiddingPolicy, FiPlaPolicy
from dynamicauc.baseline.utils import _obs_to_features, _rollout_to_end
from dynamicauc.env import AuctionEnv
import copy
import gymnasium as gym
from typing import Dict, Any

_BIG_BID: float = 1e12


class SimulationEstimator:
    """
    Monte Carlo estimator of the incremental display value:

        Δ(s_t, x) = V¹(s_t, x) − V⁰(s_t)

    where

    * V¹(s_t, x) is the gross expected reward from step t onward if the
      agent wins the current auction (forced via a very high bid) and then
      follows the continuation policy π₀.
    * V⁰(s_t) is the expected reward if the agent loses (bid = 0) and
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

    def set_env(self, env: gym.Env) -> None:
        """
        Bind to the live env.  Call once after ``env.reset()``.
        """
        self._env = env

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

class _MLPNet(nn.Module):
    """MLP backbone."""

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
            Regression targets — ``obs["expected_reward"]``.
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
            Extra keyword arguments forwarded to ``fit()``.
        """
        env = AuctionEnv(scenario=scenario)
        X_list: list = []
        y_list: list = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                er = float(obs["expected_reward"].ravel()[0])
                X_list.append(_obs_to_features(obs))
                y_list.append(er)
                obs, _, done, _, _ = env.step(
                    np.array([er], dtype=np.float32)
                )
        env.close()

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.float32)
        mlp = cls(input_dim=X.shape[1], hidden_dims=hidden_dims)
        mlp.fit(X, y, **(fit_kwargs or {}))
        return mlp

    def predict(self, obs: Dict[str, Any]) -> float:
        """Return V₀(s, x) as a scalar from an obs dict."""
        feat = torch.tensor(_obs_to_features(obs)).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            return float(self._net(feat).item())

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
