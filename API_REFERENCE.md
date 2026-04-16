"""
DynamicAuc API Reference & Best Practices
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

    import dynamicauc as dac
    from dynamicauc.simulator import Simulator
    
    # Create a scenario from preset
    scenario = dac.make("saturation-first-price-uniform")
    
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

    scenario = dac.make("saturation-first-price-uniform")
    
Available presets:
    - "saturation-first-price-uniform"
    - "recency-second-price-lognormal"
    - "mixed-first-price-exponential"

List all:
    print(dac.list_scenarios())
    print(dac.describe_scenarios())

Get preset info:
    info = dac.get_scenario_info("saturation-first-price-uniform")
    print(info["description"])


Method 2: BUILDER (Flexible, readable, recommended for custom setups)
──────────────────────────────────────────────────────────────────────

    scenario = (
        dac.ScenarioBuilder()
        .with_kernel("saturation", base_rate=0.5, saturation_scale=10.0)
        .with_auction("first-price")
        .with_competition("uniform", low=0.0, high=20.0)
        .with_arrival("poisson", breakpoints=[5.0], intensities=[0.1, 1.2])
        .with_display("iid", dim=3)
        .with_exogenous("trivial")
        .with_policy("fipla")
        .with_horizon("fixed", value=15.0)
        .with_user_features("uniform", full_dim=3, observable_dim=3)
        .with_max_history(10)
        .build()
    )

Available components:
    
    Kernels: "saturation", "recency", "competitor-extinction", "mixed"
    Auctions: "first-price", "second-price"
    Competition: "uniform", "lognormal", "exponential"
    Arrival: "poisson"
    Display: "iid", "catalogue", "fixed"
    Exogenous: "trivial"
    Policy: "fipla"


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

    from dynamicauc.registry import (
        create_kernel,
        create_auction,
        create_competition,
        create_arrival,
        create_display_sampler,
        create_exogenous,
        create_policy,
    )
    
    # Create individual primitives
    kernel = create_kernel("saturation", base_rate=0.5)
    auction = create_auction("first-price")
    
These are mostly used internally, but useful for custom scenarios.


═══════════════════════════════════════════════════════════════════════════════
4. RUNNING SIMULATIONS
═══════════════════════════════════════════════════════════════════════════════

Basic simulation:

    from dynamicauc.simulator import Simulator
    
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

    from dynamicauc.simulator import Simulator
    
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

For more sophisticated benchmarking workflows, see dynamicauc/benchmark.py (when implemented).


═══════════════════════════════════════════════════════════════════════════════
6. GYM INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

Use the Gym environment for step-by-step simulation:

    import gymnasium as gym
    import dynamicauc as dac
    
    scenario = dac.make("saturation-first-price-uniform")
    env = gym.make("DynamicAuction-v0", scenario=scenario, stochastic_reward=True)
    
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

# TODO : Enlever cette partie après
═══════════════════════════════════════════════════════════════════════════════
7. BEST PRACTICES FOR NeurIPS PAPERS
═══════════════════════════════════════════════════════════════════════════════

1. REPRODUCIBILITY
──────────────────
✓ Always set seeds:
        sim = Simulator(scenario, campaign_cfg={"seed": 42, ...})
        env.reset(seed=42)

✓ Use presets or exact builder specifications:
        Bad:  "We used a custom scenario..."
        Good: scenario = dac.make("saturation-first-price-uniform", horizon=20.0)


2. BENCHMARK DESIGN
────────────────────
✓ Start with presets for baseline comparisons:
        for preset_id in dac.list_scenarios():
            scenario = dac.make(preset_id)
            # run benchmarks
            
✓ Vary one component at a time:
        for kernel in ["saturation", "recency", "mixed"]:
            scenario = (dac.ScenarioBuilder()
                       .with_kernel(kernel)
                       .with_auction("first-price")
                       ...)

✓ Report means + standard deviations over multiple runs:
        summary_df = sim.run()[3]  # summary_df
        print(f"Conversions: {summary_df['n_conversions'].mean():.2f} ± "
              f"{summary_df['n_conversions'].std():.2f}")


3. DOCUMENTATION
──────────────────
✓ Document scenarios used:
        "We evaluated on three benchmark scenarios: saturation-first-price-uniform,
         recency-second-price-lognormal, and a custom configuration with mixed kernels."

✓ Include config in supplementary materials:
        scenario = dac.make(...)
        # Save to YAML or pickle for reproducibility

✓ Cite dynamicauc:
        @software{dynamicauc,
          title={DynamicAuc: Repeated Auction Simulation Library},
          author={...},
          year={...},
        }


4. SCALABILITY
────────────────
✓ For large experiments, batch runs:
        n_seeds = 10
        results = {}
        for seed in range(n_seeds):
            sim = Simulator(scenario, campaign_cfg={"seed": seed, ...})
            results[seed] = sim.run()

✓ Profile before optimizing:
        import time
        start = time.time()
        sim.run()
        print(f"Time: {time.time() - start:.2f}s")


5. VERSION CONTROL
────────────────────
✓ Pin dynamicauc version in your paper:
        # requirements.txt
        dynamicauc==X.Y.Z

✓ Store complete configs for reproducibility:
        # Include this in your code/data release
        scenario = dac.make("preset-id", horizon=...)
        # OR:
        # scenario_config.yaml


═══════════════════════════════════════════════════════════════════════════════
EXAMPLE: Complete NeurIPS Experiment
═══════════════════════════════════════════════════════════════════════════════

    import dynamicauc as dac
    from dynamicauc.simulator import Simulator
    import pandas as pd
    import numpy as np
    
    # Experiment setup
    scenarios = [
        dac.make("saturation-first-price-uniform"),
        dac.make("recency-second-price-lognormal"),
    ]
    
    seeds = [42, 123, 456, 789, 999]
    results = []
    
    # Run experiment
    for scenario_id, scenario in enumerate(scenarios):
        for seed in seeds:
            sim = Simulator(scenario, campaign_cfg={
                "horizon": 10.0,
                "n_runs": 10,
                "seed": seed,
            })
            _, _, _, summary_df = sim.run()
            
            results.append({
                "scenario": scenario_id,
                "seed": seed,
                "mean_conversions": summary_df["n_conversions"].mean(),
                "std_conversions": summary_df["n_conversions"].std(),
                "mean_spend": summary_df["total_spend"].mean(),
            })
    
    # Aggregate results
    results_df = pd.DataFrame(results)
    summary = results_df.groupby("scenario").agg({
        "mean_conversions": ["mean", "std"],
        "mean_spend": ["mean", "std"],
    })
    
    print(summary)


═══════════════════════════════════════════════════════════════════════════════
"""
