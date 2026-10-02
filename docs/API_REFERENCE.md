# DynamAuction API Reference

API reference and best practices for creating and running repeated auction simulations with dynamic value.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Creating Scenarios](#2-creating-scenarios-3-methods)
3. [Registry System](#3-registry-system)
4. [Running Simulations](#4-running-simulations)
5. [Benchmarking](#5-benchmarking)
6. [Gym Integration](#6-gym-integration)
7. [Wrappers](#7-wrappers)
8. [Planning Baselines](#8-planning-baselines)

---

## 1. Quick Start

```python
import dynamauction as dac
from dynamauction.simulator import Simulator

scenario = dac.make("survival-second-price-exponential", horizon_eta=50)

sim = Simulator(scenario, policy=dac.FiPlaPolicy(), campaign_cfg={"n_runs": 20, "seed": 42})
auctions_df, displays_df, conversions_df, summary_df = sim.run()
```

---

## 2. Creating Scenarios (3 Methods)

### Method 1: Preset

Fastest option — reproducible, good for benchmarks.

```python
scenario = dac.make("survival-second-price-exponential")
```

Available presets (all second price, Poisson arrivals of rate 1, campaign length $T \sim \mathrm{Exp}$ with mean $\eta = 50$ unless stated):

| Preset | Kernel | Carry-over context γ | Competition | Probe / sampler |
|--------|--------|----------------------|-------------|-----------------|
| `"survival-second-price-exponential"` | `survival` | `multiplicative` (c = 0.9, β = 1) | LogNormal(0, 1) | Bernoulli / single |
| `"survival-second-price-exponential-2"` | `survival` | `multiplicative` (c = 0.9, β = 1) | **LogNormal(−1, 1)** (lower competition) | Bernoulli / single |
| `"survival-second-price-exponential-immediate"` | `immediate-survival` (reward at bid time) | `multiplicative` (c = 0.9, β = 1) | LogNormal(0, 1) | none / single (κ ~ Bernoulli) |
| `"saturation-second-price-lognormal"` | `saturation` | `additive` (number of wins) | LogNormal(0, 1) | Bernoulli / single |
| `"recency-second-price-lognormal"` | `recency` | `last-won-time` | LogNormal(0, 1) | Bernoulli / single |
| `"mixed-second-price-exponential"` | `mixed` (immediate + saturation) | `additive` | Exponential(1); fixed $T = 20$, arrival rate 0.5 | Poisson / multi |

Scenario 2 (`-2`) is the survival scenario with lower competition: the agent can win more often, so the fatigue of
winning shortly after a previous win (see [Carry-over context](#carry-over-context-γ)) matters more.

```python
print(dac.list_scenarios())       # list all preset IDs
print(dac.describe_scenarios())   # descriptions and kernel types

info = dac.get_scenario_info("survival-second-price-exponential-2")
print(info["description"])
```

### Method 2: Builder

Flexible and readable — recommended for custom setups.

```python
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
    .with_gamma("additive")
    .with_probe("poisson")
    .with_delay_sampler("single")
    .with_max_history(10)
    .build()
)
```

Available components:

| Type | Options |
|------|---------|
| Kernels | `"saturation"`, `"recency"`, `"competitor-extinction"`, `"mixed"`, `"survival"`, `"immediate-survival"` |
| Carry-over context (γ) | `"additive"`, `"multiplicative"`, `"last-won-time"`, `"constant"` |
| Auctions | `"first-price"`, `"second-price"` |
| Competition | `"uniform"`, `"lognormal"`, `"exponential"` |
| Arrival | `"pwpoisson"`, `"cstpoisson"` |
| Display | `"iid"`, `"catalogue"`, `"fixed"`, `"normal"` |
| User features | `"uniform"`, `"normal"`, `"fixed"` |
| Horizon | `"fixed"`, `"exponential"` |
| Exogenous | `"trivial"` |
| Probe | `"poisson"`, `"bernoulli"`, `"null"` |
| DelaySampler | `"single"`, `"multi"` |

### Method 3: Hybrid

Start from a preset, override selected parameters.

```python
scenario = dac.make(
    "survival-second-price-exponential",
    horizon_eta=100,             # override horizon.params.eta
    competition_mu=-1.0,         # override competition.params.mu
    gamma_beta=0.5,              # override gamma.params.beta (recovery rate, see below)
)
```

### Carry-over context γ

The context `ctx` summarises the past won auctions and enters the conversion kernel. A `Gamma` object defines it:

| Method | Role |
|--------|------|
| `initiate_context()` | stored context at the start of a campaign |
| `current_context(ctx, t, t_past_won_auctions)` | context **used to value the auction at time t** and shown in the observation; not stored (default: the stored context) |
| `update_context(ctx, x, u, t_display, t_past_won_auctions)` | new stored context after a **won** auction (lost auctions leave it unchanged) |

| γ | Stored context | Current context at time $t$ |
|---|----------------|-----------------------------|
| `additive` | number of won auctions | same |
| `last-won-time` | time of the last won auction | same |
| `constant` | a constant | same |
| `multiplicative(c, initial_value=1, beta=1)` | $S(t_{\rm last}^-)$ at the last win | $S(t^-) = S(t_{\rm last}^-)\,\big(1 - c\,e^{-\beta (t - t_{\rm last})}\big)$ |

For `multiplicative` (survival presets), $S(0^-) = $ `initial_value` and there is **no factor before the first win**.
A win at time $t$ stores $S(t^-)$, so a win shortly after the previous one locks a small factor
$\rho(\Delta) = 1 - c\,e^{-\beta\Delta}$ ($\Delta = t - t_{\rm last}$) into the context for the rest of the campaign.
`beta` is the recovery rate: $\beta < 1$ makes the fatigue of a win last longer, $\beta > 1$ shorter. The environment
values every auction (expected reward, conversion probability, conversion draw) with `current_context`.

```python
from dynamauction.primitives import MultiplicativeGamma
g = MultiplicativeGamma(c=0.9, beta=1.0)
ctx = g.initiate_context()                                    # 1.0
g.current_context(ctx, t=2.0, t_past_won_auctions=[])         # 1.0   (no factor before the first win)
g.current_context(0.5, t=4.0, t_past_won_auctions=[1.0, 3.0]) # 0.5 · (1 − 0.9 e^{−1}) ≈ 0.334
```

---

## 3. Registry System

The registry provides factory functions for each primitive type:

```python
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

kernel  = create_kernel("saturation", base_rate=0.5)
auction = create_auction("first-price")
```

These are used internally but exposed for custom scenario construction.

---

## 4. Running Simulations

```python
from dynamauction.simulator import Simulator

sim = Simulator(scenario, policy=dac.FiPlaPolicy(), campaign_cfg={"n_runs": 20, "seed": 42})
auctions_df, displays_df, conversions_df, summary_df = sim.run()
```

**`campaign_cfg` parameters:**

| Key | Description |
|-----|-------------|
| `n_runs` | Number of independent campaign runs |
| `seed` | Random seed for reproducibility |

The campaign length comes from the scenario's horizon generator (e.g. `horizon_eta` for the exponential horizon) and the
observation history length from the scenario's `max_history`.

**Output DataFrames:**

| DataFrame | Granularity | Key columns |
|-----------|-------------|-------------|
| `auctions_df` | One row per auction opportunity | `run_id`, `auction_id`, `time`, `bid`, `competing_bid`, `won`, `payment`, `context` (current context used to value the auction), `y` |
| `displays_df` | One row per won auction | `run_id`, `auction_id`, `time`, `context`, `y`, `x_0`, `x_1`, … |
| `conversions_df` | One row per conversion event | `run_id`, `auction_id`, `display_time`, `conversion_time`, `delay`, `is_immediate` |
| `summary_df` | One row per run | `run_id`, `n_auctions`, `n_wins`, `win_rate`, `total_spend`, `n_conversions`, `n_immediate`, `n_delayed` |

---

## 5. Benchmarking

Compare multiple policies on the same scenario using paired runs (same noise streams per `run_id`):

```python
from dynamauction.simulator import Simulator

policies = {
    "FiPla":   dac.FiPlaPolicy(),
    "Constant": dac.ConstantPolicy(0.5),
}

sim = Simulator(scenario, policies, campaign_cfg={"n_runs": 50, "seed": 42})
_, _, _, summary_df = sim.run()

results = summary_df.groupby("policy").agg({
    "n_conversions": ["mean", "std"],
    "total_spend":   ["mean"],
})
print(results)
```

For large-scale evaluation with RL agents and planning baselines, see [`benchmark/BENCHMARK.md`](../benchmark/BENCHMARK.md).

---

## 6. Gym Integration

```python
import gymnasium as gym
import dynamauction as dac

scenario = dac.make("survival-second-price-exponential")
env = gym.make("DynamAuction-v0", scenario=scenario)

obs, info = env.reset(seed=42)
done = False
while not done:
    bid = policy.bid(obs)
    obs, reward, done, truncated, info = env.step(bid)

env.close()
```

**Observation space keys:**

| Key | Description |
|-----|-------------|
| `t` | Current auction time |
| `ctx` | Current carry-over context at this auction (`gamma.current_context`; for survival: the fatigue factor $S(t^-)$) |
| `step` | Current step count |
| `user_features` | Observable user feature vector |
| `display_features` | Current display features |
| `y_now` | Exogenous process value |
| `expected_reward` | Kernel-integrated oracle value of winning this auction (κ + ∫₀^{T−t} λ ds), with the current context |
| `cumulative_reward_since_last_step` | Sum of delayed conversions that matured since the last step |
| `conversion_times_since_last_step` | Times of the matured conversions |

**Action space:** non-negative float bid.

**Reward:** immediate conversions minus the payment at the current step; a delayed conversion is paid at the step that follows
its maturation. Conversions that mature between the last auction and the end of the campaign $T$ are not paid in any step.

> For a detailed walkthrough of the environment interaction loop, see [`docs/dynamauction_tutorial.ipynb`](dynamauction_tutorial.ipynb).

---

## 7. Wrappers

```python
from dynamauction import FlatAuctionObsWrapper, BidScaleWrapper, MaskExpectedRewardWrapper
from dynamauction.wrappers import SB3PolicyWrapper
```

### FlatAuctionObsWrapper

Flattens the raw Dict observation space into a 1-D `float32` vector.
Required by any MLP-based RL policy (PPO, SAC, TD3).

```python
env = gym.make("DynamAuction-v0", scenario=scenario)
env = FlatAuctionObsWrapper(env)
# env.observation_space → Box(-inf, inf, shape=(N,), float32)
```

### BidScaleWrapper

Clips the action space from `[0, 1e12]` to `[0, max_bid]`.
Prevents exploration in an unbounded range that destabilises training.
Set `max_bid` slightly above the expected optimal bid (e.g. 3× the median competing bid).

```python
env = BidScaleWrapper(env, max_bid=3.0)
# env.action_space → Box(0, 3.0, shape=(1,), float32)
```

### SB3PolicyWrapper

Adapts a trained SB3 model to the `BiddingPolicy` interface for evaluation in a raw-Dict env.
Replicates `FlatAuctionObsWrapper` internally and optionally applies `VecNormalize` statistics.

```python
import pickle
with open("models/ppo/obs_rms.pkl", "rb") as f:
    obs_rms = pickle.load(f)

raw_env = gym.make("DynamAuction-v0", scenario=scenario)
policy  = SB3PolicyWrapper(model, raw_env.observation_space, obs_rms=obs_rms)
bid     = policy.bid(obs)
```

### MaskExpectedRewardWrapper

Removes `expected_reward` from the observation before it reaches the RL policy.

`expected_reward` is the oracle value that `FiPlaPolicy` bids directly — exposing it to an RL agent gives privileged information a real advertiser would not have, making the RL vs FiPla comparison unfair.

```python
env = gym.make("DynamAuction-v0", scenario=scenario)
env = MaskExpectedRewardWrapper(env)
env = FlatAuctionObsWrapper(env)
# env.observation_space → Box of dim N-1  (expected_reward excluded)
```

When evaluating a model trained with this wrapper via `SB3PolicyWrapper`, pass `exclude_keys=["expected_reward"]` so the flattening at evaluation matches training:

```python
policy = SB3PolicyWrapper(
    model, raw_env.observation_space,
    obs_rms=obs_rms,
    exclude_keys=["expected_reward"],
)
```

**Recommended training stacks:**

```
# Fair RL / FiPla comparison (RL has no oracle access):
AuctionEnv → MaskExpectedRewardWrapper → FlatAuctionObsWrapper → BidScaleWrapper → Monitor → VecNormalize

# Oracle access (upper bound):
AuctionEnv → FlatAuctionObsWrapper → BidScaleWrapper → Monitor → VecNormalize
```

---

## 8. Planning Baselines

```python
from dynamauction.baseline.baseline_algo import (
    SimulationEstimator,
    GreedyMyopicPolicy,
    ShadingPolicy,
    MPCPolicy,
    MPCOraclePolicy,
    MPCTimeBinnedPolicy,
)
from dynamauction.baseline.estimators import MLPValuationHead  # requires torch
```

### `set_env()` hook

All planning policies expose a `set_env(env)` hook that must be called once after each `env.reset()` to bind the live environment for planning rollouts:

```python
obs, _ = env.reset(seed=seed)
policy.set_env(env)
while not done:
    bid = policy.bid(obs)
    obs, reward, done, _, info = env.step(np.array([bid]))
```

The env reference is stored (not copied), so `env.step()` calls advance the live episode normally while the policy deepcopies the env on demand inside each `bid()` call. This hook is called automatically by `evaluate.py` for any policy that exposes it.

> **Performance:** each `bid()` call performs O(n_rollouts) deepcopies. For long episodes, keep `n_rollouts` small (≤ 10).

### SimulationEstimator

Monte Carlo estimator of the incremental display value Δ(s_t, x) = V¹(s_t, x) − V⁰(s_t).

```python
estimator = SimulationEstimator(
    pi0=FiPlaPolicy(),  # continuation policy
    n_rollouts=20,
)
```

> **`pi0` receives the raw observation — `expected_reward` is always present.**
>
> The rollout env is never wrapped, so `pi0.bid(obs)` sees the full observation
> dict including the oracle `expected_reward` field. `FiPlaPolicy` (the default)
> uses it on purpose. If you use an RL model as `pi0`, wrap it with
> `SB3PolicyWrapper(model, obs_space, exclude_keys=["expected_reward"])` so the
> field is dropped internally before flattening — do **not** apply
> `MaskExpectedRewardWrapper` to the env, which would break `FiPlaPolicy` if
> it runs in the same benchmark.
>
> ```python
> from dynamauction.wrappers import SB3PolicyWrapper
> pi0 = SB3PolicyWrapper(model, obs_space, exclude_keys=["expected_reward"])
> estimator = SimulationEstimator(pi0=pi0, n_rollouts=10)
> ```

### MLPValuationHead

MLP trained offline to approximate V₀(s, x) = E[reward | win display x in state s].
Required by `MPCPolicy`.

```python
mlp = MLPValuationHead.from_scenario(scenario, n_episodes=200)
mlp.save("v0.pt")

mlp   = MLPValuationHead.load("v0.pt")
value = mlp.predict(obs)  # float
```

### GreedyMyopicPolicy

Bids the incremental value directly: b_t = Δ(s_t, x).

```python
estimator = SimulationEstimator(n_rollouts=20)
policy    = GreedyMyopicPolicy(estimator)
```

### ShadingPolicy

Applies a multiplicative shading factor α: b_t = α · Δ(s_t, x).

```python
policy = ShadingPolicy(estimator, alpha=0.8)  # α ∈ (0, ∞)
```

### MPCPolicy

Optimises the first-step shading multiplier α* via H-step CRN rollouts:
α*_t = argmax_{α ∈ A₀} (1/M) Σ_m R^(m)(α), and bids α* · V₀(s_t, x_t).
Rollout m bids α · V₀ at the current auction, then `alpha_cont` · V₀, and stops after `planning_horizon` time units.

```python
mlp    = MLPValuationHead.load("v0.pt")
policy = MPCPolicy(
    v0               = mlp,
    alpha_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    alpha_cont       = 1.0,     # multiplier of the bids after the first one, inside the rollouts
    n_rollouts       = 20,      # M
    planning_horizon = 25.0,    # H
    seed             = 0,
    tie_alpha        = None,    # ties go to the first candidate; or to the candidate closest to tie_alpha
)
```

**Rollouts never see the real future.** Each rollout deep-copies the live env, resamples the remaining horizon and
the future auction times (`_resample_remaining_horizon`), and draws its noise from a fresh generator assigned to
`env_copy.unwrapped.rng` (setting `env_copy.rng` on the Gymnasium wrapper would leave the copied real random stream in
place, so the rollout would replay the real competing bid). The same `M` (horizon seed, noise seed) pairs are shared by
all candidates (common random numbers).

**Ties.** In a second-price auction a candidate only changes a rollout if the simulated competing bid falls between two
candidate bids, so candidates often get exactly the same Q̂ (more often with few rollouts). Ties go to the first candidate
(`tie_alpha=None`) or to the candidate closest to `tie_alpha`.

### MPCTimeBinnedPolicy and MPCOraclePolicy

`MPCTimeBinnedPolicy(v0, alpha_candidates, n_bins=K, n_rollouts, planning_horizon, seed, bin_edges=None)` searches one
multiplier per time bin (|A₀|^K vectors) and bids the first-bin multiplier. Bins have width H/K by default; `bin_edges`
gives the interior boundaries instead (time elapsed since the decision; K = len(bin_edges) + 1). With
`bin_edges=(1.5,)` the first multiplier covers the current auction and the next 1.5 time units ("wait now, bid later"
at the cost of K = 2); with `bin_edges=(0.0,)` it covers only the current auction, i.e. `MPCPolicy` with α_cont chosen
jointly at every decision. `MPCOraclePolicy` is an ablation that keeps
the real future auction schedule and horizon in its rollouts (privileged information).

### Plugging into the benchmark

Planning policies cannot be selected via the CLI — pass them via `extra_policies`:

```python
from benchmark.evaluate import run_full_benchmark

df = run_full_benchmark(extra_policies={
    "Greedy": GreedyMyopicPolicy(SimulationEstimator()),
    "MPC":    MPCPolicy(mlp),
})
```
