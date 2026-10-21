"""The trained models shipped in benchmark/models/ load and reproduce the test results stored in
benchmark/benchmark_results/main/ (marked slow: run with `pytest -m slow`)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dynamauction as dac
from benchmark.evaluate import (evaluate_policy, get_planning_policies, get_rl_policies, get_simulation_policies,
                                load_eval_config)

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parents[2]
SEEDS = list(range(5))
STORED = REPO / "benchmark/benchmark_results/main/survival-second-price-exponential_eta50_results_final.csv.gz"


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.chdir(REPO)   # eval_config.yaml gives the model paths from the repository root
    cfg, algos, model_paths = load_eval_config("benchmark/eval_config.yaml")
    cfg.seeds = SEEDS
    return cfg, algos, model_paths, dac.make(cfg.scenario_id, **cfg.scenario_overrides)


def final_reward(df):
    return df.sort_values("step").groupby("seed")["cum_reward"].last()


def test_rl_models_and_baselines_reproduce_the_stored_results(setup):
    cfg, algos, model_paths, scenario = setup
    policies = get_rl_policies(scenario, algos, cfg.mask_expected_reward, eta=50, model_paths=model_paths)
    policies.update(get_planning_policies(cfg))
    stored_name = {"PPO": "PPO (training 1)", "SAC": "SAC (training 1)", "TD3": "TD3 (training 1)", "FiPla": "FiPla",
                   "Constant(0.11)": "Constant"}
    assert set(policies) == set(stored_name)
    stored = pd.read_csv(STORED, usecols=["policy", "seed", "step", "cum_reward"])
    for name, policy in policies.items():
        got = final_reward(evaluate_policy(policy, scenario, cfg, name))
        expected = final_reward(stored[(stored["policy"] == stored_name[name]) & stored["seed"].isin(SEEDS)])
        np.testing.assert_allclose(got.to_numpy(), expected.reindex(got.index).to_numpy(), rtol=1e-5, atol=1e-5,
                                   err_msg=name)   # the stored results have 6 significant digits


def test_mpc_runs_with_the_shipped_valuation_head(setup):
    cfg, _, _, scenario = setup
    cfg.seeds = [42]
    cfg.include_greedy = cfg.include_shading = cfg.include_mpc_timebin = False
    cfg.mpc_planning_horizons, cfg.mpc_alpha_cont, cfg.mpc_n_rollouts = [25.0], [1.0], 2
    (name, policy), = get_simulation_policies(cfg).items()
    df = evaluate_policy(policy, scenario, cfg, name)
    assert np.isfinite(df["cum_reward"]).all()
    assert df["alpha_chosen"].between(0.1, 1.0).all()
