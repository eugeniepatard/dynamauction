# Planning experiments: FiPla vs Shading vs MPC

This folder holds one script, [`run_experiment.py`](run_experiment.py), which compares three bidding policies on any
DynamAuction scenario, on a single machine:

- **FiPla**: bids the exact myopic value $V_0$ of the display;
- **Shading**: bids $\alpha V_0$, for a sweep of $\alpha$;
- **MPC** (`MPCPolicy`): plans with simulated rollouts, after training its MLP valuation head $\widehat V_0$.

All policies play the same campaigns, so every comparison is paired. The full documentation of the script is in the
docstring at its top: commands, every parameter, outputs and running times. This README covers the installation and
summarises the study of 2026-11-03.

## 1. Installation

Requirements: Python ≥ 3.10, git, and a few GB of disk for PyTorch.

```bash
git clone <repository-url> dynamauction
cd dynamauction
git checkout simulation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools
pip install -e ".[benchmark]"      # the library + what the MLP training needs (Stable-Baselines3, TensorBoard, ...)
```

Optional check, which takes about 30 seconds:

```bash
pip install -e ".[dev]"
pytest -q -m "not slow" tests/
```

## 2. Running a simulation

Run everything from the repository root.

```bash
# 1. quick test (about 2 minutes): checks that everything runs
python simulation/run_experiment.py --n-campaigns 8 --mpc-horizons 5 --mpc-rollouts 4 --mlp-episodes 50 --mlp-epochs 20

# 2. a real run: main survival scenario, 200 campaigns, MPC with H = 15 and 25, all the cores
python simulation/run_experiment.py --scenario survival-second-price-exponential

# 3. the most promising scenario of the study: a burst of auctions, MPC with two rollout continuations
python simulation/run_experiment.py --scenario survival-second-price-exponential-burst-arrivals-beta0.3 \
    --mpc-horizons 5 15 25 --alpha-cont 0:0.3,10:0.7 best

# 4. change any parameter of a preset with --set (the overrides of dynamauction.make)
python simulation/run_experiment.py --set gamma_beta=0.1 --set competition_mu=-2
```

The script runs, in order:
1. FiPla;
2. the Shading sweep (`--alphas`);
3. the MLP training, once per output folder;
4. MPC, one run per planning horizon (`--mpc-horizons`) and per rollout continuation (`--alpha-cont`);
5. a summary table.

**Outputs** go to `simulation/results/<scenario>[_<overrides>]/`, a folder git ignores:
- **`summary.csv`**: the table, also printed. It gives each policy's mean reward ± 95% CI, its paired difference with
  FiPla and with the best Shading, and its win rate.
- **One CSV per policy**, with one row per campaign.
- **`mlp/`**: the trained valuation head.

**Interrupted?** Run the same command again. Finished runs and finished campaigns are skipped, and the MLP is reused.
`--summary-only --out <folder>` only rebuilds the table.

**Parameters that matter most:**

| Option | Default | What it changes |
|---|---|---|
| `--n-campaigns` | 200 | precision: the 95% CI is about ±0.15–0.3 at 200 campaigns; use 1000 for final numbers |
| `--workers` | all cores | parallelism over campaigns |
| `--mpc-rollouts` | 60 | MPC noise against cost: 20 is about 3× faster but noisier |
| `--mpc-horizons` | 15 25 | planning window H, in time units |
| `--alpha-cont` | 1 | how MPC's rollouts continue after the first bid: a number, `best` (the best Shading α of the run), or a schedule `T0:A0,T1:A1`. Use the best Shading α when auctions are dense (see §3) |
| `--shading` | oracle | `oracle` bids α·V0, which is fast. `simulated` is the paper's Monte-Carlo ShadingPolicy, much slower |
| `--mpc-v0` | mlp | `oracle` uses the exact V0 in MPC, with no MLP training (quicker tests) |

**Running time on one core, main survival preset** (about 48 auctions per campaign):
- FiPla and oracle Shading: a few seconds per 100 campaigns;
- MLP training: about 5 minutes;
- **MPC(H=25, M=60): about 1–2 minutes per campaign.**

Denser arrival schedules cost more: the burst preset is about 10× slower for MPC. On a SLURM cluster, run the same
command in one job with `--workers` set to the job's cores, for example:

```bash
sbatch --cpus-per-task=32 --mem=32G --time=24:00:00 --wrap="python simulation/run_experiment.py ... --workers 32"
```

## 3. What was done on 2026-11-03 (summary)

**Starting point.** The survival fatigue context of the code didn't follow Eq. (carry-over) of the paper. It was fixed to
$S_u(t^-)=\prod_{i:\,t_i<t}\bigl(1-c\,e^{-\beta(t-t_i)}\bigr)$: each win's fatigue now fades with time. The old code
froze the factor of each pair of consecutive wins forever.

That change removed the planning advantage of the main scenario:

| Main preset (c = 0.9, β = 1, competition LogNormal(−1, 1)) | FiPla | Best Shading | MPC − best Shading |
|---|---|---|---|
| old fatigue formula (1000 campaigns) | 2.82 | 3.68 (α = 0.4) | **+0.43 ± 0.12** (H = 25, M = 60) |
| corrected formula (200 campaigns) | 6.55 | 6.50 (α = 0.8) | −0.61 ± 0.25 (H = 25, M = 20) |

The reason: with permanent fatigue, each win was a long-term decision, so a policy had to space its wins out. With
fatigue that fades in a few time units, bidding close to the myopic value is nearly optimal.

**The search for a scenario where planning pays,** with 200 paired campaigns each:

- **Fatigue strength c ∈ {0.8, 0.9, 0.95, 0.99} at β = 1:** no effect. Shading ≈ FiPla and MPC < Shading.
- **Recovery rate β × competition μ:** Shading only pays with a slow recovery, for example +21% over FiPla at β = 0.1
  with μ = −1, and +41% with μ = −2.
- **β = 0.1, with many MPC variants** (H = 5–35, M = 20–100, the rollout continuation, the tie rule; η = 50 and 100):
  at best MPC ≈ Shading, within ±0.1–0.2.
- **First-price auction:** MPC is clearly below Shading (−0.4 to −0.65): its noisy choices overpay.
- **Decaying arrival rate,** a new `StepDecayPoisson` with $\lambda(t)=\lambda_0 r^{\lfloor t/N\rfloor}$:
  - campaigns get shorter, and most of the value sits in the dense first phase, where a constant low α is already right;
  - with the default continuation, MPC is below Shading;
  - with its rollouts continuing at the best Shading α (`--alpha-cont`), MPC rises to about Shading's level
    (−0.04 ± 0.15 and −0.16 ± 0.20) but not above;
  - an *increasing* rate cannot be run with $T\sim\mathrm{Exp}(50)$: the expected number of auctions is infinite as soon
    as factor $> e^{N/50}$.
- **Longer-memory fatigue** (slower recovery, partly or fully permanent fatigue): longer memory makes shading more
  valuable, but proportional shading stays the best simple rule, so there is no room left for planning.
- **Burst of auctions** (rate 10 on [0, 10), then 1 until the end, β = 0.3; preset
  `survival-second-price-exponential-burst-arrivals-beta0.3`): **the most promising scenario.**
  - The dense burst and the long normal phase need very different bids: α ≈ 0.3 during the burst, ≈ 0.7 after. A constant
    α must compromise between them.
  - First MPC results: with H = 5 and a continuation of 0.3 then 0.7, **+0.37 ± 0.26 over the best Shading**, on 121 of
    200 campaigns.
  - Still to confirm on all campaigns, then on 1000.

**How "planning room" was measured cheaply.** Before running MPC, a scenario was screened with an oracle shading that
uses a different α in each time period. Its α were chosen on 200 campaigns, and it was evaluated on 400 others against
the best constant α. It's an upper bound for what changing α over time can bring, and it's how the burst scenario was
found.

**Code added to the library on this branch** (all tested):
- `StepDecayPoisson` (`"decaypoisson"`), and planning rollouts that sample future auctions with the rate at the
  *absolute* time (`ArrivalProcess.sample_times_from`). `PiecewiseConstantPoisson` with breakpoints had the same bug,
  now fixed. Constant-rate scenarios are unchanged, bit for bit.
- The presets `survival-second-price-exponential-decaying-arrivals`, `…-decaying-arrivals-beta0.1` and
  `…-burst-arrivals-beta0.3`.
- `MPCPolicy(alpha_cont=...)` now also accepts a time schedule `[(t_from, alpha), ...]`.

**Open points:**
- **Confirm the burst results:** all 200 campaigns, then 1000 campaigns, and other burst sizes; a burst of 20 seemed to
  leave as much room.
- **Known high-value windows** (for example a sale period where conversions are worth ×5): an idea that is not
  implemented yet. It needs a time-dependent conversion value in the survival kernel.
- **A latent bug:** the planning rollouts resample the remaining horizon as if it were exponential, which is wrong for a
  fixed horizon. That affects the `mixed` preset; it's not fixed yet.
