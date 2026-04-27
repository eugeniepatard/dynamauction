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
| `BidScaleWrapper(max_bid=5.0)` | Maps the agent's action from `[0, 1e12]` to `[0, max_bid]` for more stability in exploration during training|
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

## Planning Policies

Three simulation-based planning policies are included alongside the RL agents.

| Name | Description |
|---|---|
| **GreedyMyopic** | Bids the incremental value Δ(s, x) = V¹(s, x) − V⁰(s) estimated via Monte Carlo rollouts |
| **Shading(α)** | Bid-shaded variant: bids α · Δ(s, x); α < 1 reduces overpayment |
| **MPC** | Model-Predictive Control: selects the shading multiplier α* that maximises simulated cumulative reward over a planning horizon, using a pre-trained MLP valuation head V₀(s, x) |

### MLP Valuation Head (for MPC)

MPC requires a pre-trained `MLPValuationHead` that approximates V₀(s, x) = E[reward | win display x in state s]. It is trained offline on FiPla rollouts and **must be trained before evaluating MPC**.

```bash
# Train with defaults (500 episodes, 500 epochs, hidden dims 64-64)
python -m benchmark.train_mlp

# Larger model / more data
python -m benchmark.train_mlp --n-episodes 1000 --n-epochs 800 --hidden-dims 128 128

# Monitor training in TensorBoard (logs written automatically)
tensorboard --logdir benchmark/logs/mlp_valuation_head
```

| Parameter | Default | Description |
|---|---|---|
| `--n-episodes` | 500 | FiPla rollout episodes for data collection |
| `--val-frac` | 0.15 | Fraction of samples held out for validation |
| `--hidden-dims` | 64 64 | MLP hidden layer widths |
| `--n-epochs` | 500 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 256 | Mini-batch size |
| `--seed` | 0 | Base random seed |
| `--save-dir` | `benchmark/models/mlp_valuation_head/` | Override model output directory |
| `--log-dir` | `benchmark/logs/mlp_valuation_head/<scenario_id>/` | Override TensorBoard log directory |

Outputs written to `benchmark/models/mlp_valuation_head/`:

| File | Content |
|---|---|
| `<scenario_id>.pt` | Model weights |
| `<scenario_id>.json` | Training config + final train/val loss |
| `<scenario_id>_loss.pdf` | Train / val MSE loss curves |
| `<scenario_id>_pred.pdf` | Predicted vs actual scatter plot (val set) |

---

## Evaluation Protocol

Evaluation is run after training via `benchmark/evaluate.py`:

```bash
# Full benchmark (all RL algos + all baselines, 30 seeds)
python -m benchmark.evaluate

# Select specific RL algorithms and baselines
python -m benchmark.evaluate --algos ppo sac --baselines fipla constant

# Tune parameterized baselines
python -m benchmark.evaluate --baselines constant random --constant-bid 0.3 --random-max-bid 2.0

# RL policies only (no baselines)
python -m benchmark.evaluate --baselines none

# Include simulation-based planning policies (train_mlp.py must be run first for MPC)
python -m benchmark.evaluate --sim-policies greedy shading mpc

# All policies
python -m benchmark.evaluate --sim-policies all

# Custom number of seeds
python -m benchmark.evaluate --n-seeds 50
```

| Parameter | Value |
|---|---|
| Evaluation seeds | 0 – 29 (30 independent episodes) |
| Scenario | `survival-second-price-exponential` |
| Max bid | 5.0 |
| Default baselines | `fipla`, `constant` (bid=0.5), `random` (max=1.0) |

Per-step metrics collected for each (policy, seed) pair: `reward`, `cumulative reward`, `won`, `bid`, `payment`, `ctx`, `expected_reward`, `competing_bid`.

To add a fully custom policy, use the Python API (the CLI cannot accept arbitrary Python objects):

```python
from benchmark.evaluate import run_full_benchmark, EvalConfig
from dynamicauc.baseline.policies import BiddingPolicy

class MyPolicy(BiddingPolicy):
    def bid(self, obs):
        return obs["expected_reward"].item() * 0.8

df = run_full_benchmark(
    algos=["ppo", "sac", "td3"],
    cfg=EvalConfig(seeds=list(range(30))),
    extra_policies={"MyPolicy": MyPolicy()},
)
```

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

### 3. Train MLP valuation head (required for MPC)

```bash
python -m benchmark.train_mlp
```

Expected wall-clock time: a few minutes on CPU.

### 4. Run evaluation

```bash
python -m benchmark.evaluate --algos ppo sac td3 --n-seeds 30
```

Results are written to `benchmark/results/benchmark_results.csv`.

### 5. Generate plots

Open and execute `benchmark/benchmark_results.ipynb`.

Output PDFs are written to `benchmark/plots/`.

---

## Pretrained Models

Pretrained model weights (PPO, SAC, TD3) are available at:

> **[TODO: add Zenodo / HuggingFace DOI link here]**

Download and unpack them so the directory layout matches what `evaluate.py` expects:

```
benchmark/models/
├── ppo/
│   ├── best_model.zip   # SB3 model weights (loaded by evaluate.py)
│   └── obs_rms.pkl      # VecNormalize observation statistics
├── sac/
│   ├── best_model.zip
│   └── obs_rms.pkl
├── td3/
│   ├── best_model.zip
│   └── obs_rms.pkl
└── mlp_valuation_head/
    ├── <scenario_id>.pt    # MLPValuationHead weights (loaded by evaluate.py for MPC)
    └── <scenario_id>.json  # Training metadata
```

`best_model.zip` is the checkpoint saved by SB3's `EvalCallback` at the timestep with the highest mean evaluation reward. `obs_rms.pkl` contains the running-mean/variance statistics of `VecNormalize` and must be loaded alongside the model to reproduce the correct observation normalization at inference time.

Once in place, skip directly to step 4 (evaluation).

---

## Software Versions

| Library | Version used |
|---|---|
| Python | 3.10+ |
| Stable Baselines3 | see `benchmark/requirements_benchmark.txt` |
| Gymnasium | see `benchmark/requirements_benchmark.txt` |
| Hydra-core | see `benchmark/requirements_benchmark.txt` |
| rliable | see `benchmark/requirements_benchmark.txt` |
