from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from gymnasium import Env

import numpy as np

from dynamicauc.dataclasses import AuctionState
from dynamicauc.simulator import Scenario, Simulator
from dynamicauc.primitives import RewardKernel


ObsType = Dict[str, Any]
ActType = float

#TODO: Function to initialize the user features => same user features for all runs it seems

class AuctionEnv(Env):
    """
    Gym-style environment for repeated auctions with delayed rewards studying one user features.
    """

    def __init__(self, scenario: Scenario, seed: Optional[int] = None):
        self.scenario = scenario

        self.rng = np.random.default_rng(seed)
        
        self.kernel: RewardKernel = scenario.kernel

        self.state: Optional[AuctionState] = None
        self.auction_times: Optional[np.ndarray] = None

    def reset(self) -> ObsType:
        self.auction_times = self.scenario.arrival.sample_times(self.horizon, self.rng)
        self.scenario.exogenous.generate(self.scenario.horizon, self.rng)
        y_func = self.scenario.exogenous.y

        self.state = AuctionState(
            t=self.auction_times[0] if len(self.auction_times) > 0 else 0.0,
            ctx=self.kernel.initial_context(),
            step_idx=0,
            y_func=y_func,
            user_features=self.scenario.user_features,
            display_features=self.scenario.display_sampler.sample(self.rng),
            horizon=self.scenario.horizon,
        )

        return self._get_obs()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict]:
        """Incremental step-by-step environment interaction loop where an action is taken at each auction opportunity.
        Action = bid for the current auction opportunity.
        
            Returns
            -------
            obs : dict
                Observation of the current state, including: 
                    - current time t
                    - current context ctx
                    - current step index
                    - user features
                    - display features
                    - exogenous process function y(·)
                    - campaign horizon T
            reward : float
                Reward obtained from the last action.
            done : bool
                Whether the episode has ended (campaign is over).
            info : dict
                Additional information, including details of delayed events.
        """
        assert self.state is not None, "Call reset() first"

        t = self.state.t
        ctx = self.state.ctx
        y_func = self.state.y_func



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

    def _get_obs(self) -> ObsType:
        """Construct the observation dictionary from the current state.
        This includes the current time, context, step index, user features, 
        display features, exogenous process function, and campaign horizon."""
        return {
            "t": self.state.t,
            "ctx": self.state.ctx,
            "step": self.state.step_idx,
            "user_features": self.scenario.user_features,
            "display_features": self.scenario.display_sampler.sample(self.rng),
            "y_func": self.state.y_func,
            "horizon": self.scenario.horizon,
        }

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