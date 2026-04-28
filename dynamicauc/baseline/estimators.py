from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence
from dynamicauc.baseline.policies import BiddingPolicy, FiPlaPolicy
from dynamicauc.baseline.utils import _obs_to_features, _rollout_to_end, _resample_remaining_horizon
from dynamicauc.env import AuctionEnv
import copy
import gymnasium as gym
from typing import Dict, Any

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
        Defaults to FiPlaPolicy (truthful / oracle bidder).
    n_rollouts : int
        Number of Monte Carlo trajectories for each V¹ and V⁰ estimate.
    big_bid : float
        Bid used to force a win when estimating V¹.  Must exceed the
        highest possible competing bid in the scenario.  Default: 1e12.
    """

    def __init__(
        self,
        pi0: BiddingPolicy = None,
        n_rollouts: int = 20,
        big_bid: float = 1e12,
        seed: int = 0,
    ) -> None:
        self.pi0 = pi0 if pi0 is not None else FiPlaPolicy()
        self.n_rollouts = n_rollouts
        self.big_bid = big_bid
        self._rng = np.random.default_rng(seed)
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
            pair_seed = self._rng.integers(0, 2**31 - 1)
            # Shared horizon seed: V¹ and V⁰ face the same unknown T' (CRN).
            horizon_seed = self._rng.integers(0, 2**31 - 1)

            # V¹: force a win, recover gross value by adding back the payment
            env_v1 = copy.deepcopy(self._env)
            _resample_remaining_horizon(env_v1, np.random.default_rng(horizon_seed))
            env_v1.unwrapped.rng = np.random.default_rng(pair_seed)
            v1_sum += _rollout_to_end(
                env_v1, obs, self.pi0,
                first_bid=self.big_bid, gross_first_step=True,
            )

            # V⁰: force a loss (bid = 0) — CRN: same T' and same noise as V¹
            env_v0 = copy.deepcopy(self._env)
            _resample_remaining_horizon(env_v0, np.random.default_rng(horizon_seed))
            env_v0.unwrapped.rng = np.random.default_rng(pair_seed)
            v0_sum += _rollout_to_end(
                env_v0, obs, self.pi0,
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
        self.hidden_dims = list(hidden_dims)
        self._net = _MLPNet(input_dim, self.hidden_dims)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
        log_dir: Optional[str] = None,
    ) -> "MLPValuationHead":
        self._net = self._net.to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        has_val = X_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

        writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)

        train_losses: list = []
        val_losses: list = []

        from tqdm import tqdm
        self._net.train()
        for epoch in tqdm(range(n_epochs), desc="MLPValuationHead training", unit="epoch", leave=False):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                loss = loss_fn(self._net(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / n_batches
            train_losses.append(train_loss)

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self._net(X_val_t), y_val_t).item()
                val_losses.append(val_loss)
                self._net.train()

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                if has_val:
                    writer.add_scalar("Loss/val", val_loss, epoch)

        if writer is not None:
            writer.close()

        self._net.eval()
        self.history_ = {"train": train_losses, "val": val_losses}
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
        from tqdm import tqdm
        env = AuctionEnv(scenario=scenario)
        X_list: list = []
        y_list: list = []

        for ep in tqdm(range(n_episodes), desc="Collecting rollouts", unit="ep"):
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
            {
                "input_dim":   self.input_dim,
                "hidden_dims": self.hidden_dims,
                "state_dict":  self._net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPValuationHead":
        """Load from a file created by `save`.

        Architecture resolution order:
        1. ``hidden_dims`` key inside the ``.pt`` file (new saves)
        2. Companion ``<stem>.json`` produced by ``train_mlp.py``
        3. Infer from weight tensor shapes (legacy fallback)
        """
        import json as _json

        p = Path(path)
        data = torch.load(p, map_location="cpu", weights_only=True)

        if "hidden_dims" in data:
            hidden_dims = data["hidden_dims"]
        else:
            json_path = p.with_suffix(".json")
            if json_path.exists():
                meta = _json.loads(json_path.read_text())
                hidden_dims = meta["hidden_dims"]
            else:
                # infer from weight tensor shapes
                weights = [v for k, v in data["state_dict"].items() if k.endswith(".weight")]
                hidden_dims = [w.shape[0] for w in weights[:-1]]

        obj = cls(input_dim=data["input_dim"], hidden_dims=hidden_dims)
        obj._net.load_state_dict(data["state_dict"])
        obj._net.eval()
        return obj
