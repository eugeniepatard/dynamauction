"""Run before executing the script:
- cd ~/repeated-auction-dynamic-value
- pip install -e ."""

from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import dynamicauc as dac
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt

def run_benchmark_anarchy(policies : Dict[str, dac.BiddingPolicy], saturation_scales : List[float] =[2.0, 10.0, 50.0]):
    seeds = [42, 123, 456] # Multiples seeds pour la robustesse NeurIPS
    results = []

    for sat_scale in saturation_scales:
        scenario_base = dac.make(
            "mixed-first-price-exponential",
            kernel_base_kernel_saturation_scale=sat_scale,
            horizon=50.0 # On allonge un peu pour voir l'effet de fatigue
        )

        for name, policy in policies.items():
            
            for s in seeds:
                env = gym.make("DynamicAuction-v0", scenario=scenario_base, stochastic_reward=True)
                obs, info = env.reset(seed=s)
                
                cum_utility = 0.0
                history = []
                done = False
                
                while not done:
                    bid = policy.bid(
                        obs["user_features"], obs["display_features"], 
                        obs["ctx"].item(), env.unwrapped.state.y_func, 
                        obs["t"].item(), scenario_base
                    )
                    
                    obs, net_reward, done, _, info = env.step(bid)
                    cum_utility += net_reward
                    history.append({"t": obs["t"].item(), 
                                    "utility": cum_utility,
                                    "net_reward": net_reward,
                                    "won": info["auction"]["won"]})
                
                run_df = pd.DataFrame(history)
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
    plt.figure(figsize=(12, 7), dpi=150)
    
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
    
    plt.legend(title="Policy & Saturation ($\phi$)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
