# Benchmark: Experimental Setup and Hyperparameters

This document details the full experimental protocol used to produce the results reported in the paper.

---

## Scenario

All experiments use the built-in preset scenario **`survival-second-price-exponential`**, accessible via:

```python
import dynamicauc as dac
scenario = dac.make("survival-second-price-exponential")
```

This scenario features:
- **Auction mechanism**: Second-price (Vickrey) auction
- **Value kernel**: Survival / exponential decay (value erodes over repeated exposures)
- **Campaign horizon**: Exponentially distributed (geometric stopping time)
- **Competition model**: Stochastic competing bids

---

## Environment Wrappers

The raw `DynamicAuction-v0` Gymnasium environment is wrapped in the following order for RL training:

| Wrapper | Purpose |
|---|---|
| `FlatAuctionObsWrapper` | Flattens the Dict observation into a 1-D vector |
| `BidScaleWrapper(max_bid=5.0)` | Maps the agent's action from `[-1, 1]` to `[0, max_bid]` |
| `Monitor` (SB3) | Records episode statistics for TensorBoard |
| `VecNormalize(norm_obs=True, norm_reward=True)` | Online running-mean normalization of observations and rewards |

> **Note**: At evaluation time `VecNormalize` is used with `norm_reward=False` and `training=False`. Observation statistics are synchronized from the training environment before each evaluation rollout.

---

## Training Protocol

| Parameter | Value |
|---|---|
| Total timesteps | 2 000 000 |
| Parallel environments (`n_envs`) | 4 |
| Random seed | 42 |
| Max bid (`max_bid`) | 5.0 |
| VecNormalize | enabled |
| Evaluation frequency | every 10 000 steps (2 500 per sub-env) |
| Evaluation episodes | 20 |

Training is launched via [Hydra](https://hydra.cc/):

```bash
# Single algorithm (default: PPO)
python -m benchmark.train_rl

# Specific algorithm
python -m benchmark.train_rl algo=sac
python -m benchmark.train_rl algo=td3

# All three in parallel (one process per algo)
python -m benchmark.train_rl 'algo=ppo,sac,td3' --multirun
```

Models are saved to `benchmark/models/<algo>/` (best and final checkpoints). TensorBoard logs are written to `benchmark/logs/<algo>/`.

---

## Algorithm Hyperparameters

All algorithms use Stable Baselines3 with an `MlpPolicy` (two hidden layers of 64 units, `tanh` activations).

### PPO (Proximal Policy Optimization)

| Hyperparameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ |
| Discount factor γ | 0.99 |
| Steps per rollout (`n_steps`) | 1 024 |
| Mini-batch size | 128 |
| Epochs per update | 10 |
| GAE λ | 0.90 |

### SAC (Soft Actor-Critic)

| Hyperparameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ |
| Discount factor γ | 0.99 |
| Replay buffer mini-batch size | 256 |
| Learning starts (warm-up steps) | 1 000 |

### TD3 (Twin Delayed Deep Deterministic Policy Gradient)

| Hyperparameter | Value |
|---|---|
| Learning rate | 1 × 10⁻⁴ |
| Discount factor γ | 0.99 |
| Replay buffer mini-batch size | 256 |
| Learning starts (warm-up steps) | 5 000 |
| Exploration noise | Gaussian, σ = 0.1 (≈ 10 % of the normalized action range) |

---

## Evaluation Protocol

Evaluation is run after training via `benchmark/evaluate.py`:

```bash
python -m benchmark.evaluate --algos ppo sac td3 --n-seeds 30
```

| Parameter | Value |
|---|---|
| Evaluation seeds | 0 – 29 (30 independent episodes) |
| Scenario | `survival-second-price-exponential` |
| Max bid | 5.0 |

Per-step metrics collected for each (policy, seed) pair: `reward`, `cumulative reward`, `won`, `bid`, `payment`, `ctx`, `expected_reward`, `competing_bid`.

---

## Baseline Policies

Three non-learning baselines are included in every benchmark run:

| Name | Description |
|---|---|
| **FiPla** | First-price look-ahead policy — bids the expected immediate value |
| **Constant(0.5)** | Always bids 0.5 (10 % of `max_bid`) |
| **Random** | Bids uniformly at random in `[0, 1.0]` (20 % of `max_bid`) |

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -e .
pip install -r benchmark/requirements_benchmark.txt
```

### 2. Train RL agents

```bash
python -m benchmark.train_rl 'algo=ppo,sac,td3' --multirun
```

Expected wall-clock time: ~30–60 min per algorithm on a single CPU.

### 3. Run evaluation

```bash
python -m benchmark.evaluate --algos ppo sac td3 --n-seeds 30
```

Results are written to `benchmark/results/benchmark_results.csv`.

### 4. Generate plots

Open and execute `benchmark/benchmark_results.ipynb`, or run:

```bash
python -m benchmark.plots
```

Output PDFs are written to `benchmark/plots/`.

---

## Pretrained Models

Pretrained model weights (PPO, SAC, TD3) are available at:

> **[TODO: add Zenodo / HuggingFace DOI link here]**

Download and place them under `benchmark/models/` to skip training and jump directly to step 3 above.

---

## Software Versions

| Library | Version used |
|---|---|
| Python | 3.10+ |
| Stable Baselines3 | see `benchmark/requirements_benchmark.txt` |
| Gymnasium | see `benchmark/requirements_benchmark.txt` |
| Hydra-core | see `benchmark/requirements_benchmark.txt` |
| rliable | see `benchmark/requirements_benchmark.txt` |
