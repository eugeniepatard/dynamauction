# repeated-auction-dynamic-value

## Overview

**DynamicAuc** is a high-performance Python library designed for simulating repeated auctions where the value of a display is dynamic and depends on the history of previous wins. This library serves as the official implementation for the benchmark paper: *"Benchmarking Bidding Strategies in Repeated Auctions with Dynamic Conversion Values"*.

The library allows researchers to model realistic user behavior where conversion probabilities evolve based on frequency (saturation), recency, or competitor actions. It provides a standardized framework for evaluating bidding policies in non-stationary environments.

## Installation

You can install the library directly from the source:

```bash
git clone https://github.com/your-repo/repeated-auction-dynamic-value.git
cd repeated-auction-dynamic-value
pip install -e .
```

Requirements:
- Python 3.8+
- numpy
- pandas
- gymnasium
- scipy

## Project Structure

The repository is organized as follows:

- `dynamicauc/`: Core library source code.
  - `primitives/`: Basic data structures and utilities.
    - `kernel.py`: Implementations of dynamic value functions (Saturation, Recency, etc.).
    - `auctions.py`: Auction mechanisms (First-price, Second-price).
    - `competition.py`: Models for external bidding competition.
    - `arrival_process.py`: Stochastic processes for auction arrival times.
    - `display_sampler.py`: Samplers for display and user features.
    - `exogenous_process.py`: Models for exogenous factors affecting conversion probabilities.
  - `policies.py`: Bidding strategies and baseline policies.
  - `simulator.py`: Main engine for running repeated auction campaigns.
  - `env.py`: Gymnasium-compatible environment for Reinforcement Learning.
  - `registry.py`: Factory system for scenario components.
  - `dataclasses.py`: Core data structures for auction state and results.
- `docs/`: Tutorials and detailed documentation.


## Documentation and Usage

For a detailed breakdown of the API, please refer to [API_REFERENCE.md](./API_REFERENCE.md), or explore the `examples/` directory for practical demonstrations.

You can also run the provided Jupyter notebooks to see how to set up and execute simulations, as well as how to train and evaluate bidding strategies using the Gymnasium environments with `docs/dynamicauc_tutorial.ipynb` or `docs/simulation_tutorial.ipynb`.

