"""
Entry point for campaign data generation.

Usage
-----
    python generate.py                                     # default config
    python generate.py kernel=recency_display              # swap kernel
    python generate.py campaign.n_runs=50 campaign.seed=0  # override params
    python generate.py auction=first_price competition=uniform
"""
from __future__ import annotations

import os
import sys

# Ensure local modules are importable when Hydra changes the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from simulator import Scenario, Simulator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    scenario = Scenario(
        auction=instantiate(cfg.auction),
        competition=instantiate(cfg.competition),
        arrival=instantiate(cfg.arrival),
        kernel=instantiate(cfg.kernel),
        display_sampler=instantiate(cfg.display),
        exogenous=instantiate(cfg.exogenous),
        policy=instantiate(cfg.policy),
    )

    auctions_df, displays_df, conversions_df, summary_df = Simulator(scenario, cfg.campaign).run()

    out_dir = os.path.join(get_original_cwd(), cfg.output.dir)
    os.makedirs(out_dir, exist_ok=True)

    auctions_df.to_csv(os.path.join(out_dir, "auctions.csv"), index=False)
    displays_df.to_csv(os.path.join(out_dir, "displays.csv"), index=False)
    conversions_df.to_csv(os.path.join(out_dir, "conversions.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"\nOutputs saved to: {out_dir}")
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
