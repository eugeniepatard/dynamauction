"""Run before executing the script:
- cd ~/repeated-auction-dynamic-value
- pip install -e ."""

from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dynamicauc as dac
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt

def run_benchmark_anarchy(policies : Dict[str, dac.BiddingPolicy],
                          horizon : float = 50.0,
                          saturation_scales : List[float] =[2.0, 10.0, 50.0],
                          seeds : List[int] = [42, 123, 456]) -> pd.DataFrame:
    results = []

    for sat_scale in saturation_scales:
        scenario_base = dac.make(
            "mixed-second-price-exponential",
            kernel_base_kernel_saturation_scale=sat_scale,
            horizon=horizon # On allonge un peu pour voir l'effet de fatigue
        )

        for name, policy in policies.items():
            
            for s in seeds:
                env = gym.make("DynamicAuction-v0", scenario=scenario_base, stochastic_reward=True)
                obs, info = env.reset(seed=s)
                
                cum_utility = 0.0
                history = []
                done = False
                
                while not done:
                    bid = policy.bid(obs)
                    
                    obs, net_reward, done, _, info = env.step(bid)
                    cum_utility += net_reward
                    expected_value = env.unwrapped.kernel.expected_reward(
                        obs["user_features"], obs["display_features"], 
                        obs["ctx"].item(), env.unwrapped.state.y_func, 
                        obs["t"].item(), scenario_base.horizon
                    )
                    history.append({"t": obs["t"].item(), 
                                    "utility": cum_utility,
                                    "net_reward": net_reward,
                                    "context": obs["ctx"].item(),
                                    "expected_value": expected_value,
                                    "won": info["auction"]["won"]})
                
                run_df = pd.DataFrame(history)
                run_df['seed'] = s
                run_df['policy'] = name
                run_df['sat_scale'] = sat_scale
                results.append(run_df)

    return pd.concat(results)

def plot_anarchy_results(
    df: pd.DataFrame, 
    key: Literal['utility', 'net_reward'] = 'utility',
    policies: Optional[List[str]] = None,
    sat_scales: Optional[List[float]] = None
):
    # 1. Filtering Logic
    plot_df = df.copy()
    
    if policies is not None:
        plot_df = plot_df[plot_df['policy'].isin(policies)]
        
    if sat_scales is not None:
        plot_df = plot_df[plot_df['sat_scale'].isin(sat_scales)]

    if plot_df.empty:
        print("No data found for the selected filters.")
        return

    # 2. Plotting
    plt.figure(figsize=(10, 7), dpi=150)
    
    sns.lineplot(
        data=plot_df, 
        x='t', 
        y=key, 
        hue='policy', 
        style='sat_scale',
        errorbar='sd', 
        linewidth=2
    )

    # 3. Dynamic Labels
    ylabel = "Cumulative Net Utility" if key == 'utility' else "Instantaneous Net Reward"
    
    plt.title(f"Policy Performance Analysis ({ylabel})", fontsize=14, weight='bold', pad=15)
    plt.xlabel(r"Campaign Time $t$", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.legend(title="Policy & Saturation", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_anarchy_results_average(
    df: pd.DataFrame, 
    key: Literal['utility', 'net_reward'] = 'utility',
    policies: Optional[List[str]] = None,
    sat_scales: Optional[List[float]] = None,
    num_points: int = 200  # Number of points for the interpolation grid
):
    # 1. Filtering
    plot_df = df.copy()
    if policies:
        plot_df = plot_df[plot_df['policy'].isin(policies)]
    if sat_scales:
        plot_df = plot_df[plot_df['sat_scale'].isin(sat_scales)]

    if plot_df.empty:
        print("No data found for the selected filters.")
        return

    # 2. Setup Plot
    plt.figure(figsize=(10, 7), dpi=150)
    
    # 3. Interpolation and Averaging Logic
    # We define a common time grid from 0 to the maximum horizon
    t_min, t_max = plot_df['t'].min(), plot_df['t'].max()
    t_grid = np.linspace(t_min, t_max, num_points)

    # Use standard colors and styles for consistency
    palette = sns.color_palette("husl", len(plot_df['policy'].unique()))
    
    # Iterate through each combination to compute the mean curve
    for (policy, sat_scale), group in plot_df.groupby(['policy', 'sat_scale']):
        seeds = group['seed'].unique()
        interpolated_values = []

        for seed in seeds:
            seed_data = group[group['seed'] == seed].sort_values('t')
            # Interpolate the metric onto our common time grid
            # np.interp handles the alignment between different seed arrival times
            y_interp = np.interp(t_grid, seed_data['t'], seed_data[key])
            interpolated_values.append(y_interp)

        # Compute mean and standard deviation across seeds
        y_mean = np.mean(interpolated_values, axis=0)
        y_std = np.std(interpolated_values, axis=0)

        # Plotting the clean average
        label = f"{policy} (sat={sat_scale})"
        line, = plt.plot(t_grid, y_mean, label=label, linewidth=2.5)
        
        # Add a light shaded area for the standard deviation (no more blur!)
        plt.fill_between(t_grid, y_mean - y_std, y_mean + y_std, 
                         color=line.get_color(), alpha=0.15)

    # 4. Formatting
    ylabel = "Cumulative Net Utility" if key == 'utility' else "Instantaneous Net Reward"
    plt.title(f"Averaged Performance Analysis (N={len(plot_df['seed'].unique())} seeds)", 
              fontsize=14, weight='bold', pad=15)
    plt.xlabel(r"Campaign Time $t$", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.legend(title="Policy Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_value_dynamics(
    df: pd.DataFrame, 
    policies: Optional[List[str]] = None, 
    sat_scales: Optional[List[float]] = None,
    show_scatters: bool = True
) -> None:
    # 1. Filtering
    plot_df = df.copy()
    if policies:
        plot_df = plot_df[plot_df['policy'].isin(policies)]
    if sat_scales:
        plot_df = plot_df[plot_df['sat_scale'].isin(sat_scales)]
    
    if plot_df.empty:
        print("No data found for selected filters.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, dpi=150)
    
    # Define common lineplot arguments for consistency
    lp_args = {
        'x': 't',
        'errorbar': 'sd',
        'linewidth': 2.5
    }

    # --- PANEL 1: Evolution of User Saturation ---
    sns.lineplot(data=plot_df, y='context', hue='policy', style='sat_scale', ax=ax1, **lp_args)
    
    if show_scatters:
        wins = plot_df[plot_df['won']]
        # Use a very low alpha for scatters when multi-plotting to avoid "ink-soup"
        ax1.scatter(wins['t'], wins['context'], color='red', alpha=0.5, s=20, label='Auction Won', zorder=4)
        ax1.scatter(plot_df['t'], plot_df['context'], color='blue', alpha=0.5, s=20, label='Auctions', zorder=3)
    
    ax1.set_ylabel(r"Number of Won Auction", fontsize=11)
    ax1.set_xlabel(r"Time $t$", fontsize=11)
    ax1.set_title("Evolution of User Saturation (State Dynamics)", fontsize=13, weight='bold')
    ax1.legend(frameon=True, loc='upper left', fontsize=8, bbox_to_anchor=(1, 1))
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # --- PANEL 2: The Reward Kernel (Causal Response) ---
    sns.lineplot(data=plot_df, y='expected_value', hue='policy', style='sat_scale', ax=ax2, **lp_args, legend=False)
    
    if show_scatters:
        wins = plot_df[plot_df['won']]
        ax2.scatter(wins['t'], wins['expected_value'], color='red', alpha=0.5, s=20,label='Auctions Won', zorder=4)
        ax2.scatter(plot_df['t'], plot_df['expected_value'], color='blue', alpha=0.5, s=20, label='Auctions', zorder=3)
    
    ax2.set_xlabel(r"Time $t$", fontsize=11)
    ax2.set_ylabel(r"Expected Reward $R_i$", fontsize=11)
    ax2.set_title("Causal Reward Kernel: Diminishing Returns", fontsize=13, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_value_dynamics_average(
    df: pd.DataFrame, 
    policies: Optional[List[str]] = None, 
    sat_scales: Optional[List[float]] = None,
    show_scatters: bool = True,
    num_points: int = 300 # Slightly more points to capture state jumps
) -> None:
    # 1. Filtering
    plot_df = df.copy()
    if policies:
        plot_df = plot_df[plot_df['policy'].isin(policies)]
    if sat_scales:
        plot_df = plot_df[plot_df['sat_scale'].isin(sat_scales)]
    
    if plot_df.empty:
        print("No data found for selected filters.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, dpi=150)
    
    # Define common time grid
    t_min, t_max = plot_df['t'].min(), plot_df['t'].max()
    t_grid = np.linspace(t_min, t_max, num_points)

    # 2. Interpolation and Averaging
    # We group by policy and scale to compute the "Mean Trajectory"
    groups = plot_df.groupby(['policy', 'sat_scale'])
    
    for (name, scale), group in groups:
        seeds = group['seed'].unique()
        interp_context = []
        interp_net_reward = []
        interp_exp_value = []

        for s in seeds:
            seed_data = group[group['seed'] == s].sort_values('t')
            # Interpolate both the State (phi) and the Reward (R_i)
            interp_context.append(np.interp(t_grid, seed_data['t'], seed_data['context']))
            interp_net_reward.append(np.interp(t_grid, seed_data['t'], seed_data['net_reward']))
            interp_exp_value.append(np.interp(t_grid, seed_data['t'], seed_data['expected_value']))

        # Calculate Statistics
        ctx_mean = np.mean(interp_context, axis=0)
        ctx_std = np.std(interp_context, axis=0)
        
        exp_rew_mean = np.mean(interp_exp_value, axis=0)
        exp_rew_std = np.std(interp_exp_value, axis=0)

        # 3. Plotting Trend Lines (Panel 1 & 2)
        label = fr"{name} ($\eta={scale}$)"
        
        # Panel 1: Context
        line1, = ax1.plot(t_grid, ctx_mean, label=label, linewidth=2.5)
        ax1.fill_between(t_grid, ctx_mean - ctx_std, ctx_mean + ctx_std, 
                         color=line1.get_color(), alpha=0.1)

        # Panel 3: Net Reward
        line3, = ax2.plot(t_grid, exp_rew_mean, linewidth=2.5, color=line1.get_color())
        ax2.fill_between(t_grid, exp_rew_mean - exp_rew_std, exp_rew_mean + exp_rew_std, 
                         color=line3.get_color(), alpha=0.1)

    # 4. Heatmap Scatters (Win/Opportunity Density)
    if show_scatters:
        # We use a tiny alpha because 200 seeds create thousands of points
        # Blue mist = arrival process intensity
        # Red mist = win density
        wins = plot_df[plot_df['won']]
        
        # Panel 1 Scatters
        ax1.scatter(plot_df['t'], plot_df['context'], color='blue', alpha=0.01, s=2, zorder=1)
        ax1.scatter(wins['t'], wins['context'], color='red', alpha=0.03, s=5, zorder=2)

        ax2.scatter(plot_df['t'], plot_df['net_reward'], color='blue', alpha=0.01, s=2, zorder=1)
        ax2.scatter(wins['t'], wins['net_reward'], color='red', alpha=0.03, s=5, zorder=2)

    # 5. Formatting
    ax1.set_ylabel(r"Number of Won Auctions (Avg)", fontsize=11)
    ax1.set_xlabel(r"Campaign Time $t$", fontsize=11)
    ax1.set_title("Expected Evolution of User Saturation", fontsize=13, weight='bold')
    ax1.legend(title="Policies", frameon=True, loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.set_xlabel(r"Campaign Time $t$", fontsize=11)
    ax2.set_ylabel(r"Expected Reward $R_i$ (Avg)", fontsize=11)
    ax2.set_title("Mean Expected Reward (Diminishing Returns)", fontsize=13, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_net_reward_average(
    df: pd.DataFrame, 
    policies: Optional[List[str]] = None, 
    sat_scales: Optional[List[float]] = None,
    show_scatters: bool = True,
    num_points: int = 300 # Slightly more points to capture state jumps
) -> None:
    # 1. Filtering
    plot_df = df.copy()
    if policies:
        plot_df = plot_df[plot_df['policy'].isin(policies)]
    if sat_scales:
        plot_df = plot_df[plot_df['sat_scale'].isin(sat_scales)]
    
    if plot_df.empty:
        print("No data found for selected filters.")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 10), sharex=True, dpi=150)
    
    # Define common time grid
    t_min, t_max = plot_df['t'].min(), plot_df['t'].max()
    t_grid = np.linspace(t_min, t_max, num_points)

    # 2. Interpolation and Averaging
    # We group by policy and scale to compute the "Mean Trajectory"
    groups = plot_df.groupby(['policy', 'sat_scale'])
    
    for (name, scale), group in groups:
        seeds = group['seed'].unique()
        interp_context = []
        interp_net_reward = []
        interp_exp_value = []

        for s in seeds:
            seed_data = group[group['seed'] == s].sort_values('t')
            # Interpolate both the State (phi) and the Reward (R_i)
            interp_context.append(np.interp(t_grid, seed_data['t'], seed_data['context']))
            interp_net_reward.append(np.interp(t_grid, seed_data['t'], seed_data['net_reward']))
            interp_exp_value.append(np.interp(t_grid, seed_data['t'], seed_data['expected_value']))

        # Calculate Statistics        
        net_rew_mean = np.mean(interp_net_reward, axis=0)
        net_rew_std = np.std(interp_net_reward, axis=0)

        # 3. Plotting Trend Lines (Panel 1 & 2)
        label = fr"{name} ($\eta={scale}$)"
        
        # Panel 1: Context
        line1, = ax1.plot(t_grid, net_rew_mean, label=label, linewidth=2.5)
        ax1.fill_between(t_grid, net_rew_mean - net_rew_std, net_rew_mean + net_rew_std, 
                         color=line1.get_color(), alpha=0.1)

    # 4. Heatmap Scatters (Win/Opportunity Density)
    if show_scatters:
        # We use a tiny alpha because 200 seeds create thousands of points
        # Blue mist = arrival process intensity
        # Red mist = win density
        wins = plot_df[plot_df['won']]
        
        # Panel 1 Scatters
        ax1.scatter(plot_df['t'], plot_df['net_reward'], color='blue', alpha=0.01, s=2, zorder=1)
        ax1.scatter(wins['t'], wins['net_reward'], color='red', alpha=0.03, s=5, zorder=2)

    
    ax1.set_xlabel(r"Campaign Time $t$", fontsize=11)
    ax1.set_ylabel(r"Net Reward = Nb conversions - cost (Avg)", fontsize=11)
    ax1.legend(title="Policies", frameon=True, loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_title("Mean Causal Reward Kernel (Degradation)", fontsize=13, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()