# Benchmark: Experimental Setup and Hyperparameters

This document details the full experimental protocol used to produce the results reported in the paper.

---

## Scenario

All experiments use the built-in preset scenario **`survival-second-price-exponential`**, accessible via:

```python
import dynamauction as dac
scenario = dac.make("survival-second-price-exponential", horizon_eta=<η>)
```

This scenario features:
- **Auction mechanism**: Second-price (Vickrey) auction
- **Value kernel**: Survival / exponential decay (value erodes over repeated exposures)
- **Campaign horizon**: Exponentially distributed (geometric stopping time) with rate η
- **Competition model**: Stochastic competing bids

---

## Environment Wrappers

The raw `DynamAuction-v0` Gymnasium environment is wrapped in the following order for RL training:

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
| Total timesteps | 5 000 000 |
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

Hydra configuration lives in `benchmark/conf/train.yaml` and `benchmark/conf/algo/`.
Models are saved to `benchmark/models/<algo>/eta<η>/` (best and final checkpoints). TensorBoard logs are written to `benchmark/logs/<algo>/`.

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

Seven simulation-based planning policies are included in the benchmark runs.

### Overview

| Name | Description |
|---|---|
| **FiPla** | Bids the expected marginal reward of winning: b = E[reward \| win]. Truthful under second-price rules; serves as the main non-learning reference. |
| **GreedyMyopic** | Bids the incremental value Δ(s, x) = V¹(s, x) − V⁰(s) estimated via Monte Carlo rollouts |
| **Shading(α)** | Bid-shaded variant of Greedy: bids α · Δ(s, x) |
| **MPC(H)** | Model-Predictive Control: selects α* = argmax_{α ∈ A₀} Q̂(α) over a planning horizon H using a pre-trained MLP valuation head V₀(s, x). Horizon T' is resampled per rollout (no oracle access to the true future). |
| **MPCOracle(H)** | Same as MPC but uses the true pre-sampled future auction schedule (oracle access). Upper-bound ablation. |
| **MPCTimeBinned(H, K)** | Time-binned MPC: splits [t, t+H) into K equal bins and jointly optimises K shading multipliers α = (α₀, …, α_{K−1}) via grid search over A₀^K. Bids α*₀ · V₀(s, x) at the current step. |

### Shared Simulation Parameters

All simulation-based policies use the following parameters:

| Parameter | Value |
|---|---|
| Monte Carlo rollouts per decision step (`n_rollouts`) | 10 |
| Policy seed | 0 |

### Constant Bid

The `ConstantPolicy` always bids the same fixed value regardless of state. The optimal constant bid can be found via grid search:

```bash
python benchmark/tune_constant_bid.py --horizon-eta 50 --n-bids 20 --bid-min 0.1 --bid-max 3.0
```

| Parameter | Value |
|---|---|
| `--n-bids` | Number of values evenly spaced in `[bid-min, bid-max]` |
| `--bid-min` / `--bid-max` | Search range (default: 0.1 – 2.0) |
| `--n-seeds` | Evaluation seeds (default: 30) |

Results are saved to `benchmark/benchmark_results/` as a CSV.

### Shading Factors

The optimal shading multiplier α can be found via grid search:

```bash
python benchmark/tune_shading_alpha.py --horizon-eta 50 --n-alphas 20
```

| Policy family | Values evaluated |
|---|---|
| Shading α | 0.2, 0.4, 0.6, 0.8 |

### MPC, MPCOracle, MPCTimeBinned Parameters

| Parameter | Value |
|---|---|
| α candidates A₀ | {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5} |
| Continuation multiplier α_cont | 1.0 |
| Planning horizons H | 0.0, 4.0, 7.0, 10.0, 15.0, 25.0 |
| Number of time bins K (MPCTimeBinned) | 2, 3, 5, 10 |

All evaluation hyperparameters are set via CLI flags to `benchmark/evaluate.py` (see the Evaluation Protocol section below).

### MLP Valuation Head (for MPC, MPCOracle, MPCTimeBinned)

These three policy families require a pre-trained `MLPValuationHead` that approximates V₀(s, x) = E[reward | win display x in state s]. It is trained offline on FiPla rollouts and **must be trained before evaluating these policies**.

```bash
# Train with the configuration used in the paper
python -m benchmark.train_mlp --n-episodes 500 --n-epochs 500 --hidden-dims 128 128 --seed 42

# Train and launch TensorBoard live (open http://localhost:6006)
python -m benchmark.train_mlp --n-episodes 500 --n-epochs 500 --hidden-dims 128 128 --seed 42 --tensorboard
```

| Parameter | Value used in paper |
|---|---|
| `--n-episodes` | 500 |
| `--val-frac` | 0.15 |
| `--hidden-dims` | 128 128 |
| `--n-epochs` | 500 |
| `--lr` | 1e-3 |
| `--batch-size` | 256 |
| `--seed` | 42 |

The model is saved to:
`benchmark/models/mlp_valuation_head/survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt`

Outputs written to `benchmark/models/mlp_valuation_head/`:

| File | Content |
|---|---|
| `<scenario_id>.pt` | Model weights |
| `<scenario_id>.json` | Training config + final train/val loss |
| `<scenario_id>_loss.pdf` | Train / val MSE loss curves |
| `<scenario_id>_pred.pdf` | Predicted vs actual scatter plot (val set) |

---

## Evaluation Protocol

The evaluation entry-point is `benchmark/evaluate.py`. It evaluates all policies (planning baselines and, optionally, RL agents) sequentially.

### Shared evaluation parameters

| Parameter | Value |
|---|---|
| Evaluation seeds | 0 – 29 (30 independent episodes) |
| Scenario | `survival-second-price-exponential` |
| Max bid | 5.0 |
| Baseline | FiPla only |

Per-step metrics collected for each (policy, seed) pair: `reward`, `cumulative reward`, `won`, `bid`, `payment`, `ctx`, `expected_reward`, `competing_bid`.

To add a fully custom policy, use the Python API (the CLI cannot accept arbitrary Python objects):

```python
from benchmark.evaluate import run_full_benchmark, EvalConfig
from dynamauction.baseline.policies import BiddingPolicy

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

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -e ".[benchmark]"
```

### 2. Train RL agents *(optional — pretrained weights are provided)*

```bash
python -m benchmark.train_rl 'algo=ppo,sac,td3' --multirun
```

Expected wall-clock time: ~30–60 min per algorithm on a single CPU.
If you skip this step, `evaluate.py` automatically falls back to the pretrained weights in `benchmark/models/`.

### 3. Train MLP valuation head *(optional — pretrained weights are provided)*

```bash
python -m benchmark.train_mlp --n-episodes 500 --n-epochs 500 --hidden-dims 128 128 --seed 42
```

Expected wall-clock time: a few minutes on CPU.
If you skip this step, the pretrained `.pt` file in `benchmark/models/mlp_valuation_head/` is used automatically.

### 4. Run evaluation

```bash
# RL agents + simple baselines only (default), η=50
python -m benchmark.evaluate --horizon-eta 50

# Add all simulation-based planning policies (Greedy, Shading, MPC variants)
python -m benchmark.evaluate --horizon-eta 50 --sim-policies all

# Full paper benchmark: all policies, η=100
python -m benchmark.evaluate --horizon-eta 100 --sim-policies all
```

RL model weights are resolved automatically from `benchmark/models/<algo>/eta<η>/`.
To use a specific checkpoint instead of the auto-resolved one, pass its path explicitly:

```bash
python -m benchmark.evaluate --horizon-eta 100 \
    --ppo-model benchmark/models/ppo/eta100/lr=.../best_model \
    --sac-model benchmark/models/sac/eta100/lr=.../best_model \
    --td3-model benchmark/models/td3/eta100/lr=.../best_model
```

Results are written to `benchmark/benchmark_results/`.

### 5. Generate plots

```bash
python -m benchmark.plots --horizon-eta 50
python -m benchmark.plots --horizon-eta 100
```

Output PDFs are written to `benchmark/plots/` (gitignored — generated on demand).

---

## Pretrained Models

All pretrained weights used to produce the paper results are **shipped directly in this repository** under `benchmark/models/` — no download required.

```
benchmark/models/
├── ppo/
│   └── eta50/lr=0.0003_ent=0.02_steps=512_gamma=0.95_epochs=5/
│       ├── best_model.zip   # SB3 checkpoint (highest eval reward)
│       └── obs_rms.pkl      # VecNormalize running-mean/variance statistics
├── sac/
│   └── eta50/lr=5e-05_bs=512_warmup=500_gamma=0.95/
│       ├── best_model.zip
│       └── obs_rms.pkl
├── td3/
│   └── eta50/lr=0.0001_bs=512_noise=0.1_warmup=500_gamma=0.99/
│       ├── best_model.zip
│       └── obs_rms.pkl
└── mlp_valuation_head/
    └── survival-second-price-exponential_h128-128_ep500_e500_lr0.001_s42.pt
```

`best_model.zip` is the SB3 `EvalCallback` checkpoint saved at the timestep with the highest mean evaluation reward. `obs_rms.pkl` contains the `VecNormalize` statistics and must be loaded alongside the model to reproduce the correct observation normalisation at inference time.

`evaluate.py` resolves these paths automatically — if no freshly trained model is found it falls back to the pretrained weights above. You can therefore **skip steps 2 and 3** and go straight to step 4.

---

## Software Versions

| Library | Version used |
|---|---|
| Python | 3.10+ |
| Stable Baselines3 | see `pyproject.toml` |
| Gymnasium | see `pyproject.toml` |
| Hydra-core | see `pyproject.toml` |
| rliable | see `pyproject.toml` |
