# TODO : Rechecker les tests
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dynamauction.simulator import Simulator

# --- Fixtures ---

@pytest.fixture
def mock_scenario():
    """A minimal mock of the scenario."""
    return MagicMock()

@pytest.fixture
def campaign_cfg():
    """Standard configuration for simulator tests."""
    return {
        "n_runs": 2,
        "seed": 42
    }

@pytest.fixture
def mock_env_output():
    """Generic data returned by info dictionary in env.step."""
    return {
        "auction": {"time": 0.1, "bid": 1.0, "won": True, "payment": 0.5},
        "display": {"time": 0.1, "x_0": 0.5},
        "actual_conversions": [
            {"conversion_time": 0.5, "is_immediate": False},
            {"conversion_time": 0.1, "is_immediate": True}
        ]
    }

# --- Tests ---

class TestSimulator:
    
    @patch("dynamauction.simulator.AuctionEnv")
    def test_run_single_flow(self, mock_env_class, mock_scenario, mock_env_output):
        """Verify that _run_single correctly iterates through an environment."""
        # Setup Mock Environment
        mock_env = mock_env_class.return_value
        mock_env.reset.return_value = ({}, {})
        # Step 1: returns info, Step 2: sets done=True
        mock_env.step.side_effect = [
            ({}, 0.0, False, False, mock_env_output),
            ({}, 0.0, True, False, mock_env_output)
        ]

        sim = Simulator(mock_scenario, lambda obs: 1.0, {"n_runs": 1, "seed": 0})
        auctions, displays, conversions = sim._run_single(run_id=0, policy=lambda x: 1.0, seed=42)

        assert len(auctions) == 2
        assert len(displays) == 2
        assert len(conversions) == 4 # 2 steps * 2 conversions
        assert auctions[0]["run_id"] == 0
        mock_env.close.assert_called_once()

    @patch.object(Simulator, "_run_single")
    def test_multi_policy_aggregation(self, mock_run_single, mock_scenario):
        """Verify that results from multiple policies are concatenated with labels."""
        # Mock _run_single to return simplified data
        mock_run_single.return_value = (
            [{"won": True, "payment": 0.5, "bid": 1.0, "run_id": 0}], 
            [{"won": True, "run_id": 0}],                             
            [{"is_immediate": True, "run_id": 0}]                    
        )

        policies = {
            "p1": lambda x: 1.0,
            "p2": lambda x: 2.0
        }
        sim = Simulator(mock_scenario, policies, {"n_runs": 1, "seed": 0})
        a_df, d_df, c_df, s_df = sim.run()

        # Check policy labeling
        assert "policy" in a_df.columns
        assert set(a_df["policy"].unique()) == {"p1", "p2"}
        # Check aggregation
        assert len(a_df) == 2 # 1 run * 2 policies
        assert len(s_df) == 2

    def test_summary_logic(self):
        """Verify the statistical aggregation in _summarise."""
        auctions = pd.DataFrame([
            {"run_id": 0, "won": True, "payment": 1.5},
            {"run_id": 0, "won": False, "payment": 0.0},
        ])
        conversions = pd.DataFrame([
            {"run_id": 0, "is_immediate": True},
            {"run_id": 0, "is_immediate": False},
        ])

        summary = Simulator._summarise(auctions, conversions, n_runs=1)

        assert summary.loc[0, "n_wins"] == 1
        assert summary.loc[0, "win_rate"] == 0.5
        assert summary.loc[0, "total_spend"] == 1.5
        assert summary.loc[0, "n_conversions"] == 2
        assert summary.loc[0, "n_immediate"] == 1
        assert summary.loc[0, "n_delayed"] == 1

    def test_policy_input_types(self, mock_scenario):
        """Test that simulator accepts both callables and BiddingPolicy objects."""
        # Test Object with .bid()
        mock_policy_obj = MagicMock()
        mock_policy_obj.bid.return_value = 5.0
        sim_obj = Simulator(mock_scenario, mock_policy_obj, {"n_runs": 1, "seed": 0})
        assert "default" in sim_obj.policies
        
        # Test Raw Callable
        sim_func = Simulator(mock_scenario, lambda x: 10.0, {"n_runs": 1, "seed": 0})
        assert callable(sim_func.policies["default"])

    @patch("dynamauction.simulator.AuctionEnv")
    def test_seed_determinism(self, mock_env_class, mock_scenario):
        """Ensure that the simulator passes the correct seeds to ensure pairing."""
        mock_env = mock_env_class.return_value
        mock_env.reset.return_value = ({}, {})
        valid_info = {
            "auction": {"won": True, "payment": 0.1},
            "display": {"x": 0},
            "actual_conversions": [{"is_immediate": True}]
        }
        mock_env.step.return_value = ({}, 0.0, True, False, valid_info)
        base_seed = 100
        n_runs = 3
        sim = Simulator(mock_scenario, lambda x: 1.0, {"n_runs": n_runs, "seed": base_seed})
        sim.run()

        # Check if reset was called with base_seed + run_id
        called_seeds = [call.kwargs['seed'] for call in mock_env.reset.call_args_list]
        assert called_seeds == [100, 101, 102]

    def test_empty_results_handling(self):
        """Verify simulator doesn't crash if no auctions occur."""
        empty_a = pd.DataFrame(columns=["run_id", "won", "payment"])
        empty_c = pd.DataFrame(columns=["run_id", "is_immediate"])
        
        summary = Simulator._summarise(empty_a, empty_c, n_runs=1)
        assert summary.loc[0, "n_auctions"] == 0
        assert summary.loc[0, "total_spend"] == 0.0