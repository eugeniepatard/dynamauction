import sys
from pathlib import Path

import pytest

import benchmark.evaluate as ev

CONFIG = Path(__file__).resolve().parents[2] / "benchmark" / "eval_config.yaml"


@pytest.fixture
def captured_cfg(monkeypatch):
    """Run evaluate.main() with --config and return the EvalConfig passed to run_full_benchmark (nothing is evaluated)."""
    captured = {}
    monkeypatch.setattr(ev, "run_full_benchmark", lambda algos, cfg, save, model_paths: captured.update(cfg=cfg))

    def run(*cli_args):
        monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(CONFIG), "--no-save", *cli_args])
        ev.main()
        return captured["cfg"]

    return run


class TestEvalConfigFile:

    def test_mpc_timebin_uses_its_own_horizons(self, monkeypatch):
        monkeypatch.setattr(ev, "load_mlp", lambda *args, **kwargs: object())   # MPC heads are not needed here
        cfg, _, _ = ev.load_eval_config(CONFIG)
        assert cfg.mpc_planning_horizons == [10, 25, 35]
        assert cfg.mpc_timebin_planning_horizons == [25.0]
        cfg.include_greedy = cfg.include_shading = cfg.include_mpc = False
        assert list(ev.get_simulation_policies(cfg)) == ["MPCTimeBinned(H=25,edges=1.5)"]

    def test_values_of_the_file_are_kept(self, captured_cfg):
        cfg = captured_cfg()
        assert cfg.policy_seed == 1000
        assert cfg.constant_bid == 0.11
        assert len(cfg.seeds) == 1000

    def test_cli_flags_override_the_file(self, captured_cfg):
        cfg = captured_cfg("--policy-seed", "0", "--constant-bid", "0.3", "--random-max-bid", "2.0", "--n-seeds", "5")
        assert cfg.policy_seed == 0
        assert cfg.constant_bid == 0.3
        assert cfg.random_max_bid == 2.0
        assert cfg.seeds == list(range(5))
