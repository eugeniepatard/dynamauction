"""Benchmarking policies for repeated auction environments."""
import pandas as pd

def evaluate_policy(env_fn, policy, n_episodes=10, seed=0):
    """
    Run a policy on your AuctionEnv and return performance metrics.
    """

    rewards = []

    for i in range(n_episodes):
        env = env_fn()
        env.reset(seed=seed + i)

        done = False
        total_reward = 0.0

        while not done:
            obs = env._obs() if hasattr(env, "_obs") else env.state
            action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "std_reward": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5,
        "rewards": rewards,
    }

def benchmark_policies(env_fn, policies, n_episodes=10, seed=0):
    """
    Compare multiple bidding policies.
    """

    results = []

    for name, policy in policies.items():

        metrics = evaluate_policy(
            env_fn=env_fn,
            policy=policy,
            n_episodes=n_episodes,
            seed=seed,
        )

        results.append({
            "policy": name,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })

    return pd.DataFrame(results).sort_values("mean_reward", ascending=False)