# Benchmark: Experimental Setup and Hyperparameters

This document details the experimental protocol used to produce the benchmark results reported in the paper, on the
main survival scenario.

---

## Scenario

All benchmark results use the built-in preset **`survival-second-price-exponential`** (the main survival scenario):

```python
import dynamauction as dac
scenario = dac.make("survival-second-price-exponential", horizon_eta=50)
```

This scenario features:
- **Auctions**: a Poisson process of rate 1 over the campaign; each auction is a second-price auction against a competing
  bid $B \sim \mathrm{LogNormal}(-1, 1)$.
- **Campaign horizon**: $T \sim \mathrm{Exp}$ with mean $\eta = 50$, unknown to the bidder (47.7 auctions per campaign
  on average over the test campaigns). The benchmark is also reported for $\eta = 100$ (95.4 auctions per campaign), see
  [Results](#results).
- **Users and displays**: one user $u \sim \mathcal N(0, I_3)$ per campaign (observed) and a new display
  $x \sim \mathcal N(0, I_3)$ at every auction.
- **Value kernel** (survival): a won display converts at most once, with probability
  $f = \mathrm{clip}(S\,(\mathbf 1^\top u)(\mathbf 1^\top x), 0, 1)$, after a delay $\mathrm{Exp}(1)$; a conversion is
  worth 1 if it happens before $T$.
- **User fatigue** (multiplicative carry-over context): $S(t^-) = S(t_{\rm last}^-)\,\rho(\Delta)$,
  $\rho(\Delta) = 1 - 0.9\,e^{-\Delta}$, with $\Delta$ the time since the last won auction; $S = 1$ before the first win.
  Winning shortly after a previous win permanently lowers the value of the user's later displays.
- **Myopic value** of a display: $V_0 = f\,(1 - e^{-(T - t)})$ (`expected_reward` in the observation; only FiPla uses it).

`survival-second-price-exponential-1` is the same scenario with higher competition ($B \sim \mathrm{LogNormal}(0, 1)$).

---

## Evaluation Protocol

| Parameter | Value |
|---|---|
| Test campaigns | environment seeds 0 – 999 (1 000 campaigns), identical for every policy: all comparisons are paired |
| Policy randomness | a fresh policy per campaign, seeded with the campaign seed + 1 000 (rollout random numbers) |
| Validation campaigns | seeds 20 000 – 20 199, used only to select the RL hyperparameters and checkpoints |
| Training data of the MLP valuation head | FiPla campaigns on seeds 10 000 – 10 999 |
| Max bid | 5.0 |
| `expected_reward` | hidden from the RL policies (`MaskExpectedRewardWrapper`) |

Per-step metrics collected for each (policy, campaign): `reward`, `cum_reward`, `won`, `bid`, `payment`, `ctx`,
`expected_reward`, `competing_bid`, `alpha_chosen`. Results are reported as mean ± 95% confidence interval over the
1 000 campaigns, and as paired differences between policies.

---

## Environment Wrappers

The raw `DynamAuction-v0` Gymnasium environment is wrapped in the following order for RL training:

| Wrapper | Purpose |
|---|---|
| `MaskExpectedRewardWrapper` | Removes the oracle `expected_reward` from the observation |
| `FlatAuctionObsWrapper` | Flattens the Dict observation into a 1-D vector |
| `BidScaleWrapper(max_bid=5.0)` | Restricts the bid to `[0, max_bid]` |
| `Monitor` (SB3) | Records episode statistics for TensorBoard |
| `VecNormalize(norm_obs=True, norm_reward=True)` | Online running-mean normalization of observations and rewards |

> **Note**: At evaluation time `VecNormalize` is used with `norm_reward=False` and `training=False`. Observation statistics
> are synchronized from the training environment before each evaluation rollout.

---

## RL Training Protocol

All algorithms use Stable Baselines3 with the default `MlpPolicy` networks (PPO: two hidden layers of 64 units, `tanh`;
SAC and TD3: two hidden layers of 256 units, `ReLU`), 4 parallel environments and an evaluation every 10 000 steps on
20 campaigns.

1. **Hyperparameter search**: every configuration of the grid below is trained for 5M steps (training seed 42).
2. **Selection**: the best (configuration, checkpoint) pair on the 200 validation campaigns, where the checkpoint is either
   the best evaluation during training or the end of training.
3. **Final training**: the selected configuration is retrained for 10M steps with training seeds 42, 43 and 44; the
   checkpoint of each retraining is again chosen on the validation campaigns, then tested on the 1 000 test campaigns.
   The benchmark reports the mean of the 3 trainings on every campaign.

| Algorithm | Search grid | Selected configuration |
|---|---|---|
| PPO (144 configurations) | learning rate {5e-5, 1e-4, 2e-4, 3e-4} × `n_steps` {256, 512} × `n_epochs` {2, 5, 10} × entropy coefficient {0, 0.01, 0.02} × γ {0.95, 0.99}; mini-batch 128, GAE λ 0.90 | learning rate 5e-5, `n_steps` 256, `n_epochs` 10, entropy 0, γ 0.99 |
| SAC (48 configurations) | learning rate {5e-5, 1e-4, 3e-4, 1e-3} × batch size {256, 512} × learning starts {500, 1000, 1500} × γ {0.95, 0.99} | learning rate 5e-5, batch 512, learning starts 500, γ 0.99 |
| TD3 (96 configurations) | the SAC grid × Gaussian action noise σ {0.1, 0.2} | learning rate 1e-3, batch 512, learning starts 500, σ 0.2, γ 0.99 |

Training is launched via [Hydra](https://hydra.cc/); `benchmark/conf/train.yaml` (10M steps, scenario, evaluation) and
`benchmark/conf/algo/` (the selected configurations above) hold the settings of the final trainings:

```bash
# One training (default: PPO, training seed 42)
python -m benchmark.train_rl
python -m benchmark.train_rl algo=sac training.seed=43

# The 3 algorithms × the 3 training seeds
python -m benchmark.train_rl 'algo=ppo,sac,td3' 'training.seed=42,43,44' --multirun
```

Each training is saved to `benchmark/models/<algo>/eta50/seed<k>/`: `best_model.zip` (best evaluation during training)
with its `VecNormalize` statistics `obs_rms.pkl`, and `final_model.zip` with `final_obs_rms.pkl`. The benchmark keeps, for each
training, the checkpoint (best or final) with the higher mean reward on the 200 validation campaigns. TensorBoard logs are
written to `benchmark/logs/<algo>/eta50/seed<k>/`.

---

## Planning Policies

| Name | Description | Benchmark setting |
|---|---|---|
| **FiPla** | Bids the oracle myopic value $V_0$ of the display (truthful for a single second-price auction) | — |
| **Constant** | Bids the same value $b$ at every auction | $b = 0.11$ |
| **GreedyMyopic** | Bids the incremental value $\Delta(s, x) = V^1(s, x) - V^0(s)$, estimated by $M$ Monte-Carlo rollouts | $M = 20$ |
| **Shading(α)** | Bids $\alpha \cdot \Delta(s, x)$ | $\alpha = 0.4$, $M = 20$ |
| **MPC(H)** | Model-predictive control: at every auction, selects $\alpha^* = \arg\max_{\alpha \in A_0} \hat Q(\alpha)$ over $M$ simulated futures of $H$ time units, then bids $\alpha^* \widehat V_0(s, x)$ with the MLP valuation head; the bids after the first one in each simulated future are $\alpha_{\rm cont} \widehat V_0$ | $H \in \{10, 25, 35\}$, $M = 60$ (and $M = 80$ for $H = 25$), $\alpha_{\rm cont} \in \{1, 0.4\}$ |
| **MPCTimeBinned(H, K)** | MPC with one multiplier per time bin of $[t, t + H)$, jointly optimised over $A_0^K$ (equal bins, or explicit `bin_edges`) | additional results |

`MPCOraclePolicy` (MPC with access to the true future auction schedule) is also available in the library as an
ablation; it is not part of the benchmark.

### Shared simulation settings

- The Monte-Carlo value of GreedyMyopic / Shading simulates each future to the end of the campaign with continuation bid 0.
- In every simulated future (Shading, GreedyMyopic, MPC), the remaining horizon is resampled from the horizon distribution
  (no access to the true $T$), and the $M$ rollouts use common random numbers across the candidate actions.
- MPC: candidates $A_0 = \{0.1, 0.2, \dots, 1.0\}$; ties go to the first candidate.

### Constant bid

Sweep over $b \in \{0, 0.01, \dots, 1.5\}$ on the 1 000 test campaigns; best $b = 0.11$:

```bash
python benchmark/tune_constant_bid.py --horizon-eta 50 --bid-min 0.0 --bid-max 1.5 --n-bids 151 --n-seeds 1000
```

### Shading factor

Sweep over $\alpha \in \{0.1, \dots, 1.0\}$ with $M = 20$ on the 1 000 test campaigns; best $\alpha = 0.4$:

```bash
python benchmark/tune_shading_alpha.py --horizon-eta 50 --n-alphas 10 --alpha-min 0.1 --alpha-max 1.0 --n-rollouts 20 --n-seeds 1000
```

### MLP valuation head (for MPC and MPCTimeBinned)

MPC requires a pre-trained `MLPValuationHead` that approximates $V_0(s, x)$ from the observation (without
`expected_reward`). It is trained offline on FiPla rollouts and **must be trained before evaluating MPC** (a trained head is
provided, see below).

```bash
python -m benchmark.train_mlp --n-episodes 1000 --n-epochs 500 --hidden-dims 32 32 --activation relu --lr 1e-3 \
    --batch-size 256 --val-frac 0.15 --seed 10000
```

| Parameter | Value |
|---|---|
| `--n-episodes` | 1000 (FiPla campaigns, environment seeds 10 000 – 10 999) |
| `--hidden-dims`, `--activation` | 32 32, relu |
| `--n-epochs`, `--lr`, `--batch-size` | 500, 1e-3, 256 |
| `--val-frac` | 0.15 |
| `--seed` | 10000 |

The model is saved to
`benchmark/models/mlp_valuation_head/survival-second-price-exponential_h32-32_actrelu_ep1000_e500_lr0.001_s10000.pt`
(validation MSE 0.0018). Larger heads (64-64, 128-128) and Leaky ReLU fit the training set better but not the validation
set.

Outputs written to `benchmark/models/mlp_valuation_head/`:

| File | Content |
|---|---|
| `<stem>.pt` | Model weights |
| `<stem>.json` | Training config + final train/val loss |
| `<stem>_loss.pdf` | Train / val MSE loss curves |
| `<stem>_pred.pdf` | Predicted vs actual scatter plot (val set) |

---

## Evaluation

The evaluation entry-point is `benchmark/evaluate.py`. `benchmark/eval_config.yaml` holds the benchmark setting (main
scenario, 1 000 campaigns, the policies and parameters of the tables above):

```bash
python -m benchmark.evaluate --config benchmark/eval_config.yaml
```

- RL models: the configuration file points to training seed 42 of each algorithm; evaluate the other trainings, or freshly
  trained models, with `--ppo-model`, `--sac-model`, `--td3-model` (e.g. `benchmark/models/ppo/eta50/<...>_seed43/best_model`,
  `benchmark/models/ppo/eta50/seed42/best_model`).
- `evaluate.py` uses one policy seed (`policy_seed`) for all campaigns instead of a fresh seed per campaign (GreedyMyopic
  and Shading both use $M = 20$, as in the reported results); the stochastic planners then differ from the reported
  results by sampling noise only. FiPla, Constant and the RL policies are reproduced exactly.
- `eval_config.yaml` runs every MPC with $M = 60$; for MPC($H = 25$, $M = 80$), set `horizons: [25]`, `n_rollouts: 80`
  and `alpha_cont: [1.0]` in its `mpc` block.
- Cost: MPC($H = 25$, $M = 60$) takes about 2–4 s per decision on one CPU core (25–50 CPU-hours for 1 000 campaigns).
- The command-line defaults (30 campaigns, `Constant(0.5)`, …) are meant for quick runs; use the configuration file for the
  benchmark.

To add a fully custom policy, use the Python API (the CLI cannot accept arbitrary Python objects):

```python
from benchmark.evaluate import run_full_benchmark, EvalConfig
from dynamauction.baseline.policies import BiddingPolicy

class MyPolicy(BiddingPolicy):
    def bid(self, obs):
        return obs["expected_reward"].item() * 0.8

df = run_full_benchmark(
    algos=["ppo", "sac", "td3"],
    cfg=EvalConfig(seeds=list(range(1000))),
    extra_policies={"MyPolicy": MyPolicy()},
)
```

---

## Results

### η = 50

| Policy | Mean reward (1 000 campaigns) |
|---|---|
| MPC($H = 35$, $M = 60$, $\alpha_{\rm cont} = 0.4$) | 4.236 ± 0.271 |
| MPC($H = 25$, $M = 60$, $\alpha_{\rm cont} = 0.4$) | 4.169 ± 0.262 |
| MPC($H = 35$, $M = 60$) | 4.130 ± 0.267 |
| MPC($H = 25$, $M = 80$) | 4.127 ± 0.258 |
| MPC($H = 25$, $M = 60$) | 4.108 ± 0.261 |
| MPC($H = 10$, $M = 60$) | 3.697 ± 0.219 |
| Shading($\alpha = 0.4$, $M = 20$) | 3.680 ± 0.234 |
| PPO (mean of 3 trainings) | 2.965 ± 0.182 |
| GreedyMyopic | 2.826 ± 0.167 |
| FiPla | 2.821 ± 0.160 |
| TD3 (mean of 3 trainings) | 2.006 ± 0.121 |
| Constant($b = 0.11$) | 1.026 ± 0.099 |
| SAC (mean of 3 trainings) | 0.707 ± 0.120 |

### η = 100

Same policies and protocol with a mean campaign length $\eta = 100$, on the same 1 000 test campaigns (environment seeds
0–999): the MLP valuation head is retrained at $\eta = 100$ (same settings); the Constant and Shading sweeps are redone (best
$b = 0.11$ and best $\alpha = 0.4$ again); PPO, TD3 and SAC keep the hyperparameters selected at $\eta = 50$ (no new search)
and are retrained 3 times at $\eta = 100$, the checkpoint being chosen on 200 validation campaigns of $\eta = 100$. To
evaluate at $\eta = 100$, set `horizon_eta: 100` in `eval_config.yaml` and use an MLP head and RL models trained at
$\eta = 100$ (`scenario_overrides={"horizon_eta": 100}` in `benchmark.train_mlp.train_mlp` and in `conf/train.yaml`).

| Policy | Mean reward (1 000 campaigns) |
|---|---|
| MPC($H = 35$, $M = 60$, $\alpha_{\rm cont} = 0.4$) | 7.110 ± 0.399 |
| MPC($H = 35$, $M = 60$) | 6.801 ± 0.379 |
| MPC($H = 25$, $M = 80$) | 6.788 ± 0.368 |
| MPC($H = 25$, $M = 60$, $\alpha_{\rm cont} = 0.4$) | 6.704 ± 0.354 |
| MPC($H = 25$, $M = 60$) | 6.659 ± 0.371 |
| Shading($\alpha = 0.4$, $M = 20$) | 5.991 ± 0.330 |
| MPC($H = 10$, $M = 60$) | 5.705 ± 0.287 |
| FiPla | 3.865 ± 0.190 |
| GreedyMyopic | 3.833 ± 0.194 |
| PPO (mean of 3 trainings) | 3.555 ± 0.191 |
| TD3 (mean of 3 trainings) | 2.416 ± 0.149 |
| Constant($b = 0.11$) | 1.643 ± 0.142 |
| SAC (mean of 3 trainings) | −0.362 ± 0.210 |

- [`benchmark_results.ipynb`](benchmark_results.ipynb): the main results (tables with paired differences, rliable aggregate
  metrics and performance profiles, per-step plots), for $\eta = 50$ and $\eta = 100$, from `benchmark/benchmark_results/main/`.
- [`additional_bench_results.ipynb`](additional_bench_results.ipynb): decision time, MPC with an imperfect value estimate,
  immediate rewards, and the saturation, recency, effective-frequency and reminder kernels, from
  `benchmark/benchmark_results/additional/`.
- Pre-generated figures of the main results: [`../figures/eta50/`](../figures/eta50/) and
  [`../figures/eta100/`](../figures/eta100/).

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -e ".[benchmark]"
```

### 2. Train RL agents *(optional — pretrained weights are provided)*

```bash
python -m benchmark.train_rl 'algo=ppo,sac,td3' 'training.seed=42,43,44' --multirun
```

Expected wall-clock time: about 2 h (PPO) to 16 h (SAC, TD3) per training of 10M steps on CPU.
If you skip this step, `evaluate.py` uses the pretrained weights in `benchmark/models/`.

### 3. Train the MLP valuation head *(optional — pretrained weights are provided)*

```bash
python -m benchmark.train_mlp --n-episodes 1000 --n-epochs 500 --hidden-dims 32 32 --activation relu --seed 10000
```

Expected wall-clock time: about 6 minutes on CPU.

### 4. Run the evaluation

```bash
python -m benchmark.evaluate --config benchmark/eval_config.yaml
```

Results are written to `benchmark/benchmark_results/`.

### 5. Results and plots

Open [`benchmark_results.ipynb`](benchmark_results.ipynb) and
[`additional_bench_results.ipynb`](additional_bench_results.ipynb). Output PDFs are written to `benchmark/plots/`
(gitignored — generated on demand).

---

## Pretrained Models

The pretrained weights used to produce the results are **shipped in this repository** under `benchmark/models/` — no
download required.

```
benchmark/models/
├── ppo/eta50/lr=5e-05_ent=0_steps=256_gamma=0.99_epochs=10_seed{42,43,44}/
│   ├── best_model.zip    # SB3 checkpoint selected on the validation campaigns
│   ├── obs_rms.pkl       # VecNormalize statistics of that checkpoint
│   └── checkpoint.json   # hyperparameters, training seed, checkpoint, validation and test means
├── sac/eta50/lr=5e-05_bs=512_warmup=500_gamma=0.99_seed{42,43,44}/
├── td3/eta50/lr=0.001_bs=512_noise=0.2_warmup=500_gamma=0.99_seed{42,43,44}/
├── mlp_valuation_head/
│   └── survival-second-price-exponential_h32-32_actrelu_ep1000_e500_lr0.001_s10000.pt
└── survival-second-price-exponential-1/   # models of the original setup on survival-second-price-exponential-1
```

`best_model.zip` is the checkpoint of each training selected on the validation campaigns (the best evaluation during
training, or the end of training; see `checkpoint.json`). `obs_rms.pkl` contains the matching `VecNormalize` statistics and
must be loaded alongside the model to reproduce the observation normalisation at inference time.

`evaluate.py` resolves these paths automatically (training seed 42 by default) — you can therefore **skip steps 2 and 3**
and go straight to step 4.

---

## Software Versions

| Library | Version used |
|---|---|
| Python | 3.10+ |
| Stable Baselines3 | see `pyproject.toml` |
| Gymnasium | see `pyproject.toml` |
| Hydra-core | see `pyproject.toml` |
| rliable | see `pyproject.toml` |
