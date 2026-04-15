from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from gymnasium import Env
from gymnasium import spaces

import numpy as np
from numpy.random import seed, Generator

from dynamicauc.dataclasses import AuctionState, Scenario
from dynamicauc.primitives import RewardKernel


ObsType = Dict[str, Any]
ActType = float

class AuctionEnv(Env):
    """
    Gym-style environment for repeated auctions with delayed rewards studying one user features.
    """

    def __init__(self, scenario: Scenario,):
        
        super().__init__()

        self.scenario = scenario
        self.rng : Optional[Generator] = None
        
        self.kernel: RewardKernel = scenario.kernel

        # --------------Gym initialization params -----------------
        # TODO : paramétrer la lower and upper bound de l'action space
        self.action_space = spaces.Box(low=0.0, high=1e10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "t": spaces.Box(low=0.0, high=scenario.horizon, shape=(1,), dtype=np.float32),
            "ctx": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "step": spaces.Discrete(100000),
            "user_features": spaces.Box(low=-np.inf, high=np.inf, shape=(scenario.user_feature_generator.observable_dim,), dtype=np.float32),
            "display_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.scenario.display_sampler.dim,), dtype=np.float32),
            "y_now": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "expected_reward": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            # observation fields for delayed reward feedback
            "cumulative_reward_since_last_step": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            # We allow up to `max_history` conversions to be returned in the observation at each step, padded with zeros if fewer conversions occurred.
            "conversion_times_since_last_step": spaces.Box(low=0.0, high=scenario.horizon, shape=(scenario.max_history,), dtype=np.float32), # up to 10 conversions per step, padded with zeros
        })
        # --------------------------------------------------------

        self.state: Optional[AuctionState] = None
        self.user_features: Optional[np.ndarray] = None # Full user features for the episode
        self.auction_times: Optional[np.ndarray] = None
        self.pending_conversions: List[float] = []
        
        # State for matured delayed rewards, to be included in the next observation
        self.matured_conversions_reward: float = 0.0
        self.matured_conversions_times: List[float] = []

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

        self.auction_times = self.scenario.arrival.sample_times(self.scenario.horizon, self.rng)
        self.scenario.exogenous.generate(self.scenario.horizon, self.rng)
        y_func = self.scenario.exogenous.y

        self.state = AuctionState(
            t=self.auction_times[0] if len(self.auction_times) > 0 else 0.0,
            ctx=self.kernel.initial_context(),
            step_idx=0,
            y_func=y_func,
            display_features=self.scenario.display_sampler.sample(self.rng),)
        
        infos = {
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

            # Handle Delayed conversions via Ogata thinning
            for s in self._sample_delayed(sc.kernel, u, x, auction_state.ctx, y_func, auction_state.t, horizon, rng):
                conversions_data.append(dict(
                    step=auction_state.t,
                    auction_id=auction_state.step_idx,
                    display_time=auction_state.t,
                    conversion_time=auction_state.t + s,
                    delay=s,
                    is_immediate=False,
                ))

            new_ctx = sc.kernel.update_context(auction_state.ctx, x, auction_state.t)

        return auction_data, display_data, conversions_data, new_ctx

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict]:
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
            bid=action,
            auction_state=self.state,
            u=u,
            x=x,
            y_func=y_func,
            horizon=self.scenario.horizon,
            rng=self.rng,
        )

        if auction_data["won"]:
            # Immediate conversions are those with delay=0.
            immediate_conversions = [c for c in conversions_data if c["is_immediate"]]
            immediate_reward = float(len(immediate_conversions))
            net_reward = immediate_reward - auction_data['payment']

            # Delayed conversions are added to a pending list to be observed later.
            delayed_conversions = [c for c in conversions_data if not c["is_immediate"]]
            self.pending_conversions.extend([c["conversion_time"] for c in delayed_conversions])
        else:
            net_reward = 0.0

        self.state.step_idx += 1
        
        # Check if there are more auctions scheduled
        if self.state.step_idx < len(self.auction_times):
            t_next = self.auction_times[self.state.step_idx]
            done = False
        else:
            t_next = self.scenario.horizon
            done = True
        
        # Find delayed conversions that matured between the previous and next auction time.
        self.matured_conversions_times = sorted([c for c in self.pending_conversions if c > t_previous and c <= t_next])
        self.matured_conversions_reward = float(len(self.matured_conversions_times))
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

        expected_reward = self.scenario.kernel.expected_reward(
            self.user_features,
            self.state.display_features,
            self.state.ctx,
            self.state.y_func,
            self.state.t,
            self.scenario.horizon,
        )
        u_obs = self.scenario.user_feature_generator.get_observable_features(self.user_features)

        # Keep only the most recent `max_history` matured conversion times, and pad with zeros if fewer.
        times = list(self.matured_conversions_times) # Create local copy
        if len(times) > self.scenario.max_history:
            times = times[-self.scenario.max_history:]
        else:
            times += [0.0] * (self.scenario.max_history - len(times))

        return {
            "t": np.array([self.state.t], dtype=np.float32),
            "ctx": np.array([self.state.ctx], dtype=np.float32),
            "step": int(self.state.step_idx),
            "user_features": u_obs.astype(np.float32),
            "display_features": self.state.display_features.astype(np.float32),
            "y_now": np.array([self.state.y_func(self.state.t)], dtype=np.float32),
            "expected_reward": np.array([expected_reward], dtype=np.float32),
            "cumulative_reward_since_last_step": np.array([self.matured_conversions_reward], dtype=np.float32),
            "conversion_times_since_last_step": np.array(times, dtype=np.float32),
        }

    @staticmethod
    def _sample_delayed(
        kernel: RewardKernel,
        u: np.ndarray,
        x: np.ndarray,
        ctx: float,
        y_func: Callable[[float], float],
        t: float,
        horizon: float,
        rng: np.random.Generator,
    ) -> List[float]:
        """
        Sample delayed conversion delays in (0, horizon−t] via Ogata's thinning.

        Steps:
        1. Compute λ_max, an upper bound on λ(s) over s ∈ [0, horizon−t].
        2. Draw candidate times from a homogeneous Poisson(λ_max) process.
        3. Accept candidate at delay s with probability λ(s) / λ_max.

        Returns
        -------
            delays : List[float]
                List of accepted conversion delays (in time units) for this display.
        Notes
        -----
        - If the kernel's λ is zero or negative over the entire interval, this will return an empty list.
        - The returned delays are relative to the display time t (i.e., actual conversion times would be t + s for s in delays).
        """
        remaining = horizon - t
        if remaining <= 0.0:
            return []
        lam_max = kernel.lambda_upper_bound(u, x, ctx, t, remaining)
        if lam_max <= 0.0:
            return []

        delays: List[float] = []
        s = 0.0
        while s < remaining:
            s += rng.exponential(1.0 / lam_max)
            if s >= remaining:
                break
            lam = kernel.lambda_(u, x, ctx, y_func(t + s), t, s)
            if rng.uniform() < lam / lam_max:
                delays.append(s)
        return delays