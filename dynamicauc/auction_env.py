from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from gymnasium import Env

import numpy as np

from dynamicauc.simulator import Scenario
from dynamicauc.primitives import RewardKernel


ObsType = Dict[str, Any]
ActType = float


@dataclass
class AuctionState:
    t: float
    ctx: float
    step_idx: int


class AuctionEnv(Env):
    """
    Gym-style environment for repeated auctions with delayed rewards.
    """

    def __init__(self, scenario: Scenario, campaign_cfg: Any, seed: Optional[int] = None):
        self.scenario = scenario
        self.campaign_cfg = campaign_cfg
        self.rng = np.random.default_rng(seed)

        self.kernel: RewardKernel = scenario.kernel

        self.state: Optional[AuctionState] = None

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------
    def reset(self) -> ObsType:
        self.state = AuctionState(
            t=0.0,
            ctx=self.kernel.initial_context(),
            step_idx=0,
        )

        return self._get_obs()

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict]:
        assert self.state is not None, "Call reset() first"

        t = self.state.t
        ctx = self.state.ctx

        # 1. sample competition / outcome (simplified placeholder)
        # (you can plug your real competition model here)
        win = self.rng.random() < 0.5

        reward = 0.0
        info: Dict[str, Any] = {}

        # 2. immediate reward
        if win:
            y = self.scenario.arrival.sample_y(t)
            reward += self.kernel.kappa(
                u=self._sample_user(),
                x=np.array([action]),
                ctx=ctx,
                y=y,
            )

            # update context
            ctx = self.kernel.update_context(ctx, np.array([action]), t)

        # 3. delayed events (simplified: sample delays)
        if win:
            delays = self._sample_delayed_events(action, ctx, t)
            info["delays"] = delays

        # 4. advance time
        dt = self.scenario.arrival.sample_interarrival()
        t_next = t + dt

        self.state = AuctionState(
            t=t_next,
            ctx=ctx,
            step_idx=self.state.step_idx + 1,
        )

        done = self.state.step_idx >= self.campaign_cfg.n_steps

        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------
    def _get_obs(self) -> ObsType:
        return {
            "t": self.state.t,
            "ctx": self.state.ctx,
            "step": self.state.step_idx,
        }

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _sample_user(self):
        # placeholder: replace with your arrival model
        return self.rng.normal(size=5)

    def _sample_delayed_events(self, action, ctx, t):
        kernel = self.kernel

        u = self._sample_user()
        x = np.array([action])
        y_func = self.scenario.arrival.y_function

        remaining = self.campaign_cfg.horizon - t
        if remaining <= 0:
            return []

        lam_max = kernel.lambda_upper_bound(u, x, ctx, t, remaining)

        delays = []
        s = 0.0

        while s < remaining:
            s += self.rng.exponential(1.0 / lam_max)
            if s >= remaining:
                break

            lam = kernel.lambda_(
                u, x, ctx, y_func(t + s), t, s
            )

            if self.rng.random() < lam / lam_max:
                delays.append(s)

        return delays