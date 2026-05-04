"""
DynamAuction API Reference & Best Practices
==========================================

This document describes the professional API for creating and running
repeated auction simulations with dynamic value.

Table of Contents
────────────────────
1. Quick Start
2. Creating Scenarios (3 Methods)
3. Registry System
4. Running Simulations
5. Benchmarking
6. Gym Integration


═══════════════════════════════════════════════════════════════════════════════
1. QUICK START
═══════════════════════════════════════════════════════════════════════════════

The fastest way to get started:

    import dynamauction as dac
    from dynamauction.simulator import Simulator
    
    # Create a scenario from preset
    scenario = dac.make("saturation-second-price-uniform")
    
    # Run simulations
    sim = Simulator(scenario, campaign_cfg={
        "horizon": 10.0,
        "n_runs": 20,
        "seed": 42,
    })
    auctions_df, displays_df, conversions_df, summary_df = sim.run()


═══════════════════════════════════════════════════════════════════════════════
2. CREATING SCENARIOS (3 Methods)
═══════════════════════════════════════════════════════════════════════════════

Method 1: PRESET (Fastest, reproducible, good for benchmarks)
──────────────────────────────────────────────────────────────

    scenario = dac.make("saturation-second-price-uniform")
    
Available presets:
    - "paper-experiment-scenario"
    - "saturation-second-price-uniform"
    - "recency-second-price-lognormal"
    - "mixed-second-price-exponential"
    - "survival-second-price-lognormal"

List all:
    print(dac.list_scenarios())
    print(dac.describe_scenarios())

Get preset info:
    info = dac.get_scenario_info("saturation-second-price-uniform")
    print(info["description"])


Method 2: BUILDER (Flexible, readable, recommended for custom setups)
──────────────────────────────────────────────────────────────────────

    scenario = (
        dac.ScenarioBuilder()
        .with_kernel("saturation", base_rate=0.5, saturation_scale=10.0)
        .with_auction("first-price")
        .with_competition("uniform", low=0.0, high=20.0)
        .with_arrival("pwpoisson", breakpoints=[5.0], intensities=[0.1, 1.2])
        .with_display("iid", dim=3)
        .with_exogenous("trivial")
        .with_horizon("fixed", value=15.0)
        .with_user_features("uniform", full_dim=3, observable_dim=3)
        .with_probe("poisson")
        .with_delay_sampler("single")
        .with_max_history(10)
        .build()
    )

Available components:
    
    Kernels: "saturation", "recency", "competitor-extinction", "mixed", "survival"
    Auctions: "first-price", "second-price"
    Competition: "uniform", "lognormal", "exponential"
    Arrival: "pwpoisson", "cstpoisson"
    Display: "iid", "catalogue", "fixed", "normal"
    Exogenous: "trivial"
    Probe: "poisson", "survival"
    DelaySampler: "single", "multi"


Method 3: HYBRID (Start from preset, customize parameters)
────────────────────────────────────────────────────────────

    scenario = dac.make(
        "saturation-first-price-uniform",
        horizon_params_value=20.0,  # Nested override: horizon.params.value
        user_params_observable_dim=2, # Nested override: user.params.observable_dim
        kernel_base_rate=0.8,       # Nested override: kernel.params.base_rate
        competition_high=50.0       # Nested override: competition.params.high
    )


═══════════════════════════════════════════════════════════════════════════════
3. REGISTRY SYSTEM
═══════════════════════════════════════════════════════════════════════════════

The registry provides factory functions for each primitive type:

    from dynamauction.scenarios.builder import (
        create_kernel,
        create_auction,
        create_competition,
        create_arrival,
        create_display_sampler,
        create_exogenous,
        create_probe,
        create_delay_sampler,
    )
    
    # Create individual primitives
    kernel = create_kernel("saturation", base_rate=0.5)
    auction = create_auction("first-price")
    
These are mostly used internally, but useful for custom scenarios.


═══════════════════════════════════════════════════════════════════════════════
4. RUNNING SIMULATIONS
═══════════════════════════════════════════════════════════════════════════════

Basic simulation:

    from dynamauction.simulator import Simulator
    
    sim = Simulator(scenario, campaign_cfg={
        "horizon": 10.0,
        "n_runs": 20,
        "seed": 42,
    })
    auctions_df, displays_df, conversions_df, summary_df = sim.run()

Campaign config parameters:
    - horizon: Campaign duration (time units)
    - max_history: Max number of conversion times in step observations
    - n_runs: Number of independent campaign runs
    - seed: Random seed for reproducibility

Output DataFrames:

    auctions_df: One row per auction opportunity
        Columns: run_id, auction_id, time, bid, competing_bid, won, payment, context, y
        
    displays_df: One row per won auction (display received)
        Columns: run_id, auction_id, time, context, y, x_0, x_1, ...
        
    conversions_df: One row per conversion event
        Columns: run_id, auction_id, display_time, conversion_time, delay, is_immediate
        
    summary_df: One row per run with aggregate statistics
        Columns: run_id, n_auctions, n_wins, win_rate, total_spend, n_conversions, 
                 n_immediate, n_delayed

═══════════════════════════════════════════════════════════════════════════════
5. BENCHMARKING
═══════════════════════════════════════════════════════════════════════════════

Compare multiple policies on the same scenario:

    from dynamauction.simulator import Simulator
    
    policies = {
        "FiPla": dac.FiPlaPolicy(),
        # Add more policies here
    }
    
    sim = Simulator(scenario, policies, campaign_cfg={"horizon": 10.0, "n_runs": 50, "seed": 42})
    _, _, _, summary_df = sim.run()
    
    results = summary_df.groupby("policy").agg({
        "n_conversions": ["mean", "std"],
        "total_spend": ["mean"]
    })
    print(results)

For more sophisticated benchmarking workflows, see dynamauction/benchmark.py (when implemented).


═══════════════════════════════════════════════════════════════════════════════
6. GYM INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

Use the Gym environment for step-by-step simulation:

    import gymnasium as gym
    import dynamauction as dac
    
    scenario = dac.make("saturation-second-price-uniform")
    env = gym.make("DynamAuction-v0", scenario=scenario, stochastic_reward=True)
    
    obs, info = env.reset(seed=42)
    
    done = False
    while not done:
        # Decide action (bid)
        bid = policy.bid(obs)
        
        obs, reward, done, truncated, info = env.step(bid)
    
    env.close()

Observation space:
    - t: Current auction time
    - ctx: Kernel context state
    - step: Current step count
    - user_features: Observable user feature vector
    - display_features: Current display features
    - y_now: Exogenous process value
    - cumulative_reward_since_last_step: Sum of delayed conversions that matured since the last step.
    - conversion_times_since_last_step: List of times for the matured conversions.
Action space:
    - bid: Non-negative float representing the bid amount.
Reward:
    - Stochastic reward based on the kernel and probe dynamics. Immediate conversions yield reward at the same step, while delayed conversions yield reward in future steps when they mature.

**Note** : To learn how to use more in depth the dynamauction gym environment, please refer to the `docs/dynamauction_tutorial.ipynb` notebook.


═══════════════════════════════════════════════════════════════════════════════
7. WRAPPERS
═══════════════════════════════════════════════════════════════════════════════

Three wrappers ship with dynamauction to bridge the raw AuctionEnv with RL
libraries and evaluation tools.

    from dynamauction import FlatAuctionObsWrapper, BidScaleWrapper
    from dynamauction.wrappers import SB3PolicyWrapper

FlatAuctionObsWrapper (ObservationWrapper)
──────────────────────────────────────────
Flattens the raw Dict observation space into a 1-D float32 vector.
Discrete fields are cast to float; Box fields are concatenated in insertion
order. Required by any MLP-based RL policy (PPO, SAC, TD3).

    env = gym.make("DynamAuction-v0", scenario=scenario)
    env = FlatAuctionObsWrapper(env)
    # env.observation_space → Box(-inf, inf, shape=(N,), float32)

BidScaleWrapper (ActionWrapper)
────────────────────────────────
Clips the action space from [0, 1e12] (raw env) to [0, max_bid].
Prevents exploration in an unbounded range that destabilises RL training.
Set max_bid to a value slightly above the expected optimal bid (e.g. 3× the
median competing bid).

    env = BidScaleWrapper(env, max_bid=3.0)
    # env.action_space → Box(0, 3.0, shape=(1,), float32)

SB3PolicyWrapper (BiddingPolicy adapter)
─────────────────────────────────────────
Adapts a trained SB3 model to the BiddingPolicy interface so it can be
evaluated in a raw-Dict env (no wrappers). Replicates FlatAuctionObsWrapper
internally and optionally applies VecNormalize statistics saved at train time.

    import pickle
    with open("models/ppo/obs_rms.pkl", "rb") as f:
        obs_rms = pickle.load(f)

    raw_env = gym.make("DynamAuction-v0", scenario=scenario)
    policy  = SB3PolicyWrapper(model, raw_env.observation_space, obs_rms=obs_rms)
    bid     = policy.bid(obs)   # obs is a raw Dict

MaskExpectedRewardWrapper (ObservationWrapper)
──────────────────────────────────────────────
Removes ``expected_reward`` from the observation dict before it reaches the
RL policy.

``expected_reward`` is the kernel-integrated oracle value κ + ∫λ ds — the
exact signal that ``FiPlaPolicy`` bids directly.  Exposing it to an RL agent
gives it privileged information a real advertiser would not have, making the
RL vs FiPla comparison unfair.  Apply this wrapper whenever RL policies
should be evaluated on equal footing with FiPla.

    env = gym.make("DynamAuction-v0", scenario=scenario)
    env = MaskExpectedRewardWrapper(env)   # ← drops expected_reward from obs
    env = FlatAuctionObsWrapper(env)
    # env.observation_space → Box of dim N-1  (expected_reward excluded)

``FiPlaPolicy`` reads ``obs["expected_reward"]`` from the raw env and is
unaffected by this wrapper.

When evaluating a model trained with this wrapper via ``SB3PolicyWrapper``,
pass ``exclude_keys=["expected_reward"]`` so evaluation flattening matches
training flattening:

    policy = SB3PolicyWrapper(
        model, raw_env.observation_space,
        obs_rms=obs_rms,
        exclude_keys=["expected_reward"],   # ← must match training wrappers
    )

Recommended training stack for fair RL / FiPla comparison:

    AuctionEnv → MaskExpectedRewardWrapper → FlatAuctionObsWrapper → BidScaleWrapper → Monitor → VecNormalize

Recommended training stack (without mask, oracle access):

    AuctionEnv → FlatAuctionObsWrapper → BidScaleWrapper → Monitor → VecNormalize


═══════════════════════════════════════════════════════════════════════════════
8. PLANNING BASELINES
═══════════════════════════════════════════════════════════════════════════════

Three planning-based bidding policies ship alongside the simple baselines.
All inherit from ``BiddingPolicy`` and are importable from ``dynamauction.baseline``.

    from dynamauction.baseline import (
        SimulationEstimator,
        MLPValuationHead,
        GreedyMyopicPolicy,
        ShadingPolicy,
        MPCPolicy,
    )

set_env() hook
──────────────
All three policies expose a ``set_env(env)`` hook that must be called once
after each ``env.reset()`` to bind the live environment for planning rollouts:

    obs, _ = env.reset(seed=seed)
    policy.set_env(env)          # required — stores reference to live env
    while not done:
        bid = policy.bid(obs)
        obs, reward, done, _, info = env.step(np.array([bid]))

The env reference is stored (not copied), so ``env.step()`` calls advance the
live episode normally while the policy deepcopies the env on demand inside
each ``bid()`` call.  This hook is called automatically by ``evaluate.py``
for any policy that exposes it.

Performance note: each ``bid()`` call performs O(n_rollouts) deepcopies.
For long episodes, keep ``n_rollouts`` small (≤ 10) or use ``FiPlaPolicy``
as a fast zero-rollout alternative.

SimulationEstimator
───────────────────
Monte Carlo estimator of the incremental display value:

    Δ(s_t, x) = V¹(s_t, x) − V⁰(s_t)

V¹ forces a win (bid = 1e12), adds back the payment to recover gross value,
then rolls out under π₀.  V⁰ forces a loss (bid = 0) and rolls out under π₀.

    estimator = SimulationEstimator(
        pi0=FiPlaPolicy(),   # continuation policy (default: FiPla)
        n_rollouts=20,       # MC trajectories per estimate
    )

MLPValuationHead
────────────────
MLP trained offline to approximate V₀(s, x) = E[reward | win display x in state s].
Training targets are the FiPla labels obs["expected_reward"] (kernel integral).
Used by MPCPolicy for fast value estimates inside CRN rollouts.

    # Collect FiPla rollouts and train in one call
    mlp = MLPValuationHead.from_scenario(scenario, n_episodes=200)
    mlp.save("v0.pt")

    # Load and inspect
    mlp = MLPValuationHead.load("v0.pt")
    value = mlp.predict(obs)   # float

GreedyMyopicPolicy
──────────────────
Bids the incremental value directly.  Truthful under second-price rules.

    b_t = Δ(s_t, x)

    estimator = SimulationEstimator(n_rollouts=20)
    policy    = GreedyMyopicPolicy(estimator)

ShadingPolicy
─────────────
Applies a multiplicative shading factor α to the incremental value estimate.
Required for first-price auctions; acts as a regulariser against MC noise.

    b_t = α · Δ(s_t, x)

    policy = ShadingPolicy(estimator, alpha=0.8)   # α ∈ (0, ∞)

MPCPolicy
─────────
Optimises the first-step shading multiplier α* via H-step CRN rollouts:

    α*_t = argmax_{α ∈ A₀} (1/M) Σ_m R^(m)(α)

Common Random Numbers (CRN): the M noise streams are shared across all
candidate α values, eliminating inter-candidate variance from common
stochastic factors.

    mlp    = MLPValuationHead.load("v0.pt")
    policy = MPCPolicy(
        v0               = mlp,
        scenario         = scenario,
        alpha_candidates = [0.6, 0.8, 1.0, 1.2, 1.5],
        alpha_cont       = 1.0,      # multiplier for continuation steps
        n_rollouts       = 10,       # CRN rollout pairs M
        planning_horizon = 7.0,      # time units H to simulate ahead
    )

Plugging into the benchmark
───────────────────────────
Planning policies cannot be selected via the CLI (they require constructed
objects).  Pass them via ``extra_policies``:

    from benchmark.evaluate import run_full_benchmark
    df = run_full_benchmark(extra_policies={
        "Greedy": GreedyMyopicPolicy(SimulationEstimator()),
        "MPC":    MPCPolicy(mlp, scenario),
    })