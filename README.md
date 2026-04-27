# repeated-auction-dynamic-value

## Overview

**DynamicAuc** is a Python library for simulating repeated auctions where the value of a display is dynamic and depends on the history of previous wins. It is the official implementation accompanying the paper: *"Benchmarking Bidding Strategies in Repeated Auctions with Dynamic Conversion Values"*. # TODO: mettre à jour nom papier

The library models realistic advertiser behavior where conversion probabilities evolve based on frequency (saturation), recency, or competitor actions. It provides a standardized Gymnasium environment and simulator for evaluating bidding policies in non-stationary auction settings.

## Installation

```bash
git clone https://github.com/your-repo/repeated-auction-dynamic-value.git
cd repeated-auction-dynamic-value
pip install -e .                    # core library
pip install -e ".[benchmark]"       # + benchmark dependencies (SB3, TensorBoard, …)
pip install -e ".[dev]"             # + development tools (pytest, black, flake8)
```

Requirements: Python 3.9+.

> **Note — editable install requires recent tooling.** `pip install -e .` uses PEP 660, which requires **pip ≥ 21.3** and **setuptools ≥ 64**. If the command fails with *"missing the `build_editable` hook"*, upgrade first:
> ```bash
> pip install --upgrade pip setuptools
> ```

## Package Structure

```
dynamicauc/
├── primitives/         Auction rules, kernels, competition, arrival, display,
│                       exogenous process, gamma, user features, probe & delay samplers
├── baseline/           Bidding policies (FiPla, Constant, Random, GreedyMyopic,
│                       Shading, MPC) and the MLPValuationHead estimator
├── scenarios/          Preset scenario registry and ScenarioBuilder
├── checker/            Gymnasium-style sanity checker (check_env) for AuctionEnv
├── env.py              Gymnasium-compatible AuctionEnv
├── simulator.py        Simulator : runs multi-policy repeated auction campaigns
├── wrappers.py         Observation and action wrappers (Flat, BidScale, SB3, Mask)
└── dataclasses.py      Scenario and AuctionState dataclasses
```

## Quick Start

```python
import dynamicauc as dac

# Create a preset scenario
scenario = dac.make("survival-second-price-exponential")

# Use as a Gymnasium environment
env = dac.AuctionEnv(scenario)
obs, info = env.reset(seed=42)

# Sanity-check your environment or a custom scenario
dac.check_env(env)

# Run a multi-policy simulation campaign
from dynamicauc.simulator import Simulator

sim = Simulator(scenario, {"FiPla": dac.FiPlaPolicy()}, campaign_cfg={"n_runs": 20, "seed": 42})
auctions_df, displays_df, conversions_df, summary_df = sim.run()
```

## Benchmark

The `benchmark/` directory contains a full experimental pipeline for reproducing the paper results: RL agent training (PPO, SAC, TD3 via Stable Baselines3), MLP valuation head training for the MPC policy, multi-policy evaluation across seeds, and result plotting. See [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md) for the complete experimental protocol, hyperparameters, and reproduction steps.

## Documentation

- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — full API reference: scenario creation (preset, builder, hybrid), simulation, Gymnasium integration, wrappers, and planning baselines.
- [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md) — experimental setup, training protocol, hyperparameters, and evaluation procedure.
- `docs/dynamicauc_tutorial.ipynb` — step-by-step tutorial for the Gymnasium environment.
- `docs/simulator_tutorial.ipynb` — tutorial for the simulation API.
