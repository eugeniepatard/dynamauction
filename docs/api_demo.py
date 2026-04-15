"""
Examples demonstrating professional API usage for dynamicauc.

This module shows recommended patterns for:
  1. Creating scenarios (presets, builder, YAML config)
  2. Running simulations
  3. Benchmarking policies
  4. Working with the Gym environment
"""

# ────────────────────────────────────────────────────────────────────────────
# Example 1: Quick start with presets
# ────────────────────────────────────────────────────────────────────────────

def example_preset_quick_start():
    """
    Quickest way to get started: use a built-in preset.
    """
    import dynamicauc as dac
    from dynamicauc.simulator import Simulator

    # List available presets
    print("Available scenarios:", dac.list_scenarios())
    
    # Create scenario from preset
    scenario = dac.make("saturation-first-price-uniform")
    policy = dac.FiPlaPolicy() # Policies are now separate from scenarios
    
    # Run a single simulation
    sim = Simulator(scenario, policy, campaign_cfg={
        "horizon": 10.0,
        "n_runs": 5,
        "seed": 42,
    })
    auctions_df, displays_df, conversions_df, summary_df = sim.run()
    
    print("Summary:", summary_df)
    return scenario


# ────────────────────────────────────────────────────────────────────────────
# Example 2: Programmatic construction with builder pattern
# ────────────────────────────────────────────────────────────────────────────

def example_builder_pattern():
    """
    Flexible construction using fluent builder API.
    """
    import dynamicauc as dac
    from dynamicauc.simulator import Simulator
    
    # Note: .with_policy() is no longer part of the builder
    scenario = (
        dac.ScenarioBuilder()
        .with_kernel("saturation", base_rate=0.5, saturation_scale=10.0)
        .with_auction("first-price")
        .with_competition("uniform", low=0.0, high=20.0)
        .with_arrival("poisson", breakpoints=[5.0], intensities=[0.1, 1.2])
        .with_display("iid", dim=3)
        .with_exogenous("trivial")
        .with_horizon(15.0)
        .with_user_features("uniform", full_dim=3, observable_dim=3)
        .build()
    )
    
    # Instantiate a policy to use with the simulator
    policy = dac.FiPlaPolicy()
    sim = Simulator(scenario, policy, campaign_cfg={
        "horizon": 15.0,
        "n_runs": 10,
        "seed": 123,
    })
    auctions_df, _, _, summary_df = sim.run()
    
    print("Auctions per run:", summary_df["n_auctions"].to_dict())
    return scenario


# ────────────────────────────────────────────────────────────────────────────
# Example 3: Custom scenario with overrides
# ────────────────────────────────────────────────────────────────────────────

def example_preset_with_overrides():
    """
    Start from a preset but customize specific parameters.
    """
    import dynamicauc as dac
    
    # Start from preset, override horizon
    scenario = dac.make(
        "saturation-first-price-uniform",
        horizon=20.0,
        user_params_observable_dim=2,
    )
    
    print("Scenario horizon:", scenario.horizon)
    print("Scenario user generator:", scenario.user_feature_generator)
    return scenario


# ────────────────────────────────────────────────────────────────────────────
# Example 4: Gym environment usage
# ────────────────────────────────────────────────────────────────────────────

def example_gym_environment():
    """
    Using the Gym-compatible environment for step-by-step simulation.
    """
    import gymnasium as gym
    import dynamicauc as dac
    import numpy as np
    
    # Create scenario
    scenario = dac.make("saturation-first-price-uniform")
    
    # Register and create environment
    # (Note: Adjust this if the environment is already registered elsewhere)
    env = gym.make("DynamicAuction-v0", scenario=scenario, stochastic_reward=True)
    obs, _ = env.reset(seed=42)
    
    # Run one episode
    done = False
    truncated = False
    total_reward = 0.0
    total_cost = 0.0
    step_count = 0
    
    # For the gym env, we can still use a policy object to decide bids
    policy = dac.FiPlaPolicy()
    
    while not done and not truncated and step_count < 10:
        # Use policy to decide bid
        bid = policy.bid(obs)
        
        obs, reward, done, truncated, info = env.step(bid)
        total_reward += reward
        cost = info["auction"]["payment"] # Accessing the info dict we built!
        total_cost += cost
        step_count += 1
    
    print(f"Completed {step_count} steps, total reward: {total_reward:.3f}")
    env.close()


# ────────────────────────────────────────────────────────────────────────────
# Example 5: Batch benchmarking of multiple policies
# ────────────────────────────────────────────────────────────────────────────

def example_policy_comparison():
    """
    Compare multiple policies on the same scenario.
    """
    import dynamicauc as dac
    from dynamicauc.simulator import Simulator
    from dynamicauc.policies import FiPlaPolicy
    from dynamicauc.policies import ConstantPolicy, RandomPolicy
    
    # Create base scenario
    scenario = dac.make("saturation-first-price-uniform")
    
    # Compare different policies
    policies = {
        "FiPla": FiPlaPolicy(),
        "ConstantBid": ConstantPolicy(0.5),
        "Random": RandomPolicy(),
    }
    
    # Pass the dictionary of policies directly to the Simulator
    sim = Simulator(scenario, policies, campaign_cfg={
        "horizon": 10.0,
        "n_runs": 20,
        "seed": 42,
    })
    _, _, _, summary_df = sim.run()
    
    # Group by policy to compute mean/std
    results = summary_df.groupby("policy").agg({
        "n_conversions": ["mean", "std"],
        "total_spend": ["mean"]
    })
    
    print("Policy comparison results:")
    print(results)
    
    return results


# ────────────────────────────────────────────────────────────────────────────
# Example 6: Scenario information and discovery
# ────────────────────────────────────────────────────────────────────────────

def example_scenario_discovery():
    """
    Discover and inspect available scenarios.
    """
    import dynamicauc as dac
    
    # Print all available scenarios
    print(dac.describe_scenarios())
    
    # Get info about a specific scenario
    info = dac.get_scenario_info("saturation-first-price-uniform")
    print("\nDetailed info for 'saturation-first-price-uniform':")
    print(f"  Description: {info['description']}")
    print(f"  Kernel: {info['kernel']['name']}")
    print(f"  Auction: {info['auction']['name']}")


# ────────────────────────────────────────────────────────────────────────────
# Main: Run all examples
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Quick Start with Presets")
    print("=" * 70)
    example_preset_quick_start()
    
    print("\n" + "=" * 70)
    print("Example 2: Builder Pattern")
    print("=" * 70)
    example_builder_pattern()
    
    print("\n" + "=" * 70)
    print("Example 3: Preset with Overrides")
    print("=" * 70)
    example_preset_with_overrides()
    
    print("\n" + "=" * 70)
    print("Example 4: Scenario Discovery")
    print("=" * 70)
    example_scenario_discovery()
    
    print("\n" + "=" * 70)
    print("Example 5: Policy Comparison")
    print("=" * 70)
    example_policy_comparison()
    
    # Skip Gym example if not available
    try:
        print("\n" + "=" * 70)
        print("Example 6: Gym Environment")
        print("=" * 70)
        example_gym_environment()
    except Exception as e:
        print(f"Gym example skipped: {e}")
