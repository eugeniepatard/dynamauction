"""
Gymnasium environment for repeated auctions with dynamic values.

Defines ``AuctionEnv``, registered as ``DynamAuction-v0``, which implements
the full episode loop: auction arrival, bid submission, competing bid draw,
payment computation, delayed reward accumulation, and context update.
The observation space is a Dict of named arrays; the action space is a
scalar bid clipped to ``[0, 1e12]``.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from gymnasium import Env
from gymnasium import spaces

import numpy as np
from numpy.random import seed, Generator

from dynamauction.types import AuctionState, Scenario
from dynamauction.primitives import RewardKernel


ObsType = Dict[str, Any]
ActType = float
OBSERVATION_KEYS= ["t", "ctx", "step", "user_features", 
                   "display_features", "y_now", "expected_reward", 
                   "cumulative_reward_since_last_step", 
                   "conversion_times_since_last_step"]

class AuctionEnv(Env):
    """
    Gym-style environment for repeated auctions with delayed rewards studying one user features.
    """

    def __init__(self, scenario: Scenario,):
        
        super().__init__()

        self.scenario = scenario
        self.rng : Generator = np.random.default_rng()
        
        self.kernel: RewardKernel = scenario.kernel
        self.conversion_value = scenario.conversion_value
        self.probe = scenario.probe
        self.delay_sampler = scenario.delay_sampler

        # --------------Gym initialization params -----------------
        self.action_space = spaces.Box(low=0.0, high=1e12, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "t": spaces.Box(low=0.0, high=1e12, shape=(1,), dtype=np.float32),
            "ctx": spaces.Box(low=-1e12, high=1e12, shape=(1,), dtype=np.float32),
            "step": spaces.Discrete(100000),
            "user_features": spaces.Box(low=-1e12, high=1e12, shape=(scenario.user_feature_generator.observable_dim,), dtype=np.float32),
            "display_features": spaces.Box(low=-1e12, high=1e12, shape=(self.scenario.display_sampler.dim,), dtype=np.float32),
            "y_now": spaces.Box(low=-1e12, high=1e12, shape=(1,), dtype=np.float32),
            "expected_reward": spaces.Box(low=-1e12, high=1e12, shape=(1,), dtype=np.float32),
            # observation fields for delayed reward feedback
            "cumulative_reward_since_last_step": spaces.Box(low=0.0, high=1e12, shape=(1,), dtype=np.float32),
            # We allow up to `max_history` conversions to be returned in the observation at each step, padded with zeros if fewer conversions occurred.
            "conversion_times_since_last_step": spaces.Box(low=0.0, high=1e12, shape=(scenario.max_history,), dtype=np.float32), # up to 10 conversions per step, padded with zeros
        })
        # --------------------------------------------------------

        self.state: Optional[AuctionState] = None
        self.user_features: Optional[np.ndarray] = None # Full user features for the episode
        self.horizon: Optional[float] = None # Horizon for the episode
        self.auction_times: Optional[np.ndarray] = None
        self.pending_conversions: List[float] = []
        
        # State for matured delayed rewards, to be included in the next observation
        self.matured_conversions_reward: float = 0.0
        self.matured_conversions_times: List[float] = []
        self.t_past_won_auctions: List[float] = []

    def reset(self, seed: Optional[int] = None, options: dict[str, Any] | None = None) -> Tuple[Any, dict[str, Any]]:
        # ---- Gym reset() logic -----
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Generate a new user for this episode
        self.user_features = self.scenario.user_feature_generator.generate(self.rng)
        self.pending_conversions = []
        self.matured_conversions_reward = 0.0
        self.matured_conversions_times = []
        self.t_past_won_auctions = []

        self.horizon = self.scenario.horizon_generator.sample(self.rng)
        self.auction_times = self.scenario.arrival.sample_times(self.horizon, self.rng)
        self.scenario.exogenous.generate(self.horizon, self.rng)
        y_func = self.scenario.exogenous.y

        self.state = AuctionState(
            t=self.auction_times[0] if len(self.auction_times) > 0 else 0.0,
            ctx=self.scenario.gamma.initiate_context(),
            step_idx=0,
            y_func=y_func,
            display_features=self.scenario.display_sampler.sample(self.rng),)
        
        infos = {
            "horizon": self.horizon,
            "auction_times": self.auction_times,
        }

        return self._get_obs(), infos
    
    def _run_single_auction(
        self,
        bid: float,
        auction_state: AuctionState,
        u: np.ndarray,
        x: np.ndarray,
        y_func: Any,
        horizon: float,
        rng: np.random.Generator
        ) -> Tuple[Dict, Optional[Dict], List[Dict], Any]:
        """
        Handles the logic for a single auction event: bidding, winning, 
        and generating rewards (conversions).

        Returns
        -------
            auction_data : dict
                Record of the auction event including :
                    - step : time of the auction
                    - auction_id : unique identifier for the auction
                    - time : time of the auction
                    - bid : the bid placed by the agent
                    - competing_bid : the sampled competing bid
                    - won : whether the auction was won
                    - payment : the payment made if the auction was won, else 0
                    - context : the kernel context before the auction
                    - y : the exogenous process value at the time of the auction
            display_data : dict or None
                Record of the display event if the auction was won, else None.
            conversions_data : List[dict], Shape : (n_wins * avg_conversions_per_win,)
                List of conversion event records generated from this display.
            new_ctx : Any
                Updated kernel context after this auction/display.
        """
        sc = self.scenario
        B = sc.competition.sample(rng)
        
        won = sc.auction.win(bid, B)
        payment = sc.auction.payment(bid, B) if won else 0.0
        y_t = y_func(auction_state.t)

        # Prepare Auction Record
        auction_data = dict(
            step=auction_state.t,
            auction_id=auction_state.step_idx,
            time=auction_state.t,
            bid=bid,
            competing_bid=B,
            won=won,
            payment=payment,
            context=auction_state.ctx,
            y=y_t,
        )

        display_data = None
        conversions_data = []
        new_ctx = auction_state.ctx

        if won:
            # Prepare Display Record
            display_data = dict(
                step=auction_state.t,
                auction_id=auction_state.step_idx,
                time=auction_state.t,
                context=auction_state.ctx,
                y=y_t,
                **{f"x_{j}": float(v) for j, v in enumerate(x)},
            )

            # Handle Immediate conversions ~ Poisson(κ)
            kap = sc.kernel.kappa(u, x, auction_state.ctx, y_t)
            for _ in range(int(rng.poisson(kap))):
                conversions_data.append(dict(
                    step=auction_state.t,
                    auction_id=auction_state.step_idx,
                    display_time=auction_state.t,
                    conversion_time=auction_state.t,
                    delay=0.0,
                    is_immediate=True,
                ))

            # Handle Delayed conversions via the configured sampler 
            for s in self.delay_sampler.sample(self.probe, u, x, auction_state.ctx, y_func, auction_state.t, horizon, rng):
                conversions_data.append(dict(
                    step=auction_state.t,
                    auction_id=auction_state.step_idx,
                    display_time=auction_state.t,
                    conversion_time=auction_state.t + s,
                    delay=s,
                    is_immediate=False,
                ))

            new_ctx = sc.gamma.update_context(auction_state.ctx, x, u, auction_state.t, self.t_past_won_auctions)
            self.t_past_won_auctions.append(auction_state.t)

        return auction_data, display_data, conversions_data, new_ctx

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        """Incremental step-by-step environment interaction loop where an action is taken at each auction opportunity (t=t^{(i)}).
        Action = bid for the current auction opportunity.
        
            Returns
            -------
            obs : ObsType
                Observation of the current state, including: 
                    - current time t
                    - current context ctx
                    - current step index
                    - user features
                    - display features
                    - exogenous process function y(·)
                    - campaign horizon T
            net_reward : float
                Reward obtained from the last action.
            done : bool
                Whether the episode has ended (campaign is over).
            truncated : bool
                Whether the episode was truncated.
            info : dict
                Additional information, including details of delayed events.
        """
        assert self.state is not None, "Call reset() first"

        t_previous = self.state.t
        ctx = self.state.ctx
        y_func = self.state.y_func
        u = self.user_features
        x = self.state.display_features

        auction_data, display_data, conversions_data, new_ctx = self._run_single_auction(
            bid=float(np.asarray(action).ravel()[0]),
            auction_state=self.state,
            u=u,
            x=x,
            y_func=y_func,
            horizon=self.horizon,
            rng=self.rng,
        )

        if auction_data["won"]:
            # Immediate conversions are those with delay=0.
            immediate_conversions = [c for c in conversions_data if c["is_immediate"]]
            immediate_gross_reward = float(len(immediate_conversions)) * self.conversion_value
            immediate_net_reward = immediate_gross_reward - auction_data['payment']

            # Delayed conversions are added to a pending list to be observed later.
            delayed_conversions = [c for c in conversions_data if not c["is_immediate"]]
            self.pending_conversions.extend([c["conversion_time"] for c in delayed_conversions])
        else:
            immediate_net_reward = 0.0

        # The total reward for the agent is the net immediate utility + the value of matured delayed conversions
        net_reward = immediate_net_reward + self.matured_conversions_reward

        self.state.step_idx += 1
        
        # Check if there are more auctions scheduled
        if self.state.step_idx < len(self.auction_times):
            t_next = self.auction_times[self.state.step_idx]
            done = False
        else:
            t_next = self.horizon
            done = True
        
        # Find delayed conversions that matured between the previous and next auction time.
        self.matured_conversions_times = sorted([c for c in self.pending_conversions if c > t_previous and c <= t_next])
        self.matured_conversions_reward = float(len(self.matured_conversions_times)) * self.conversion_value
        self.pending_conversions = [c for c in self.pending_conversions if c > t_next]

        truncated = False

        self.state.t = t_next
        self.state.ctx = new_ctx # Update context if auction was won
        self.state.display_features = self.scenario.display_sampler.sample(self.rng)

        # 4. Prepare info for debugging/benchmarking
        info = {
            "auction": auction_data,
            "display": display_data,
            "conversions_count": len(conversions_data),
            "actual_conversions": conversions_data,
        }

        obs = self._get_obs()

        return obs, net_reward, done, truncated, info

    def _get_obs(self) -> ObsType:
        """Construct the observation dictionary from the current state.
        This includes the current time, context, step index, user features, 
        display features, exogenous process function, and matured delayed rewards
        from the previous interval.
        """

        expected_reward = self.conversion_value * self.scenario.kernel.expected_reward(
            self.user_features,
            self.state.display_features,
            self.state.ctx,
            self.state.y_func,
            self.state.t,
            self.horizon,
        )
        u_obs = self.scenario.user_feature_generator.get_observable_features(self.user_features)

        # Keep only the most recent `max_history` matured conversion times, and pad with zeros if fewer.
        times = list(self.matured_conversions_times) # Create local copy
        if len(times) > self.scenario.max_history:
            times = times[-self.scenario.max_history:]
        else:
            times += [0.0] * (self.scenario.max_history - len(times))

        obs : Dict[str, Any] =  {
            "t": np.array([self.state.t], dtype=np.float32),
            "ctx": np.array([self.state.ctx], dtype=np.float32),
            "step": np.int64(self.state.step_idx),
            "user_features": u_obs.astype(np.float32),
            "display_features": self.state.display_features.astype(np.float32),
            "y_now": np.array([self.state.y_func(self.state.t)], dtype=np.float32),
            "expected_reward": np.array([float(expected_reward)], dtype=np.float32),
            "cumulative_reward_since_last_step": np.array([self.matured_conversions_reward], dtype=np.float32),
            "conversion_times_since_last_step": np.array(times, dtype=np.float32),
        }
        return obs
