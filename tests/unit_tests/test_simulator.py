import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dynamauction.simulator import Simulator
from dynamauction.env import AuctionEnv

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


class TestSimulatorWrapperSupport:
    """Tests for env_wrappers parameter and gymnasium-conformant action format."""

    @pytest.fixture
    def full_env(self):
        """AuctionEnv with a fully configured mock scenario (mirrors conftest)."""
        scenario = MagicMock()
        scenario.user_feature_generator.observable_dim = 3
        scenario.user_feature_generator.generate.return_value = np.array([1.0, 0.5, 0.2])
        scenario.user_feature_generator.get_observable_features.return_value = np.array([1.0, 0.5, 0.2])
        scenario.display_sampler.dim = 3
        scenario.display_sampler.sample.return_value = np.array([0.1, 0.2, 0.3])
        scenario.horizon_generator.sample.return_value = 10.0
        scenario.arrival.sample_times.return_value = np.array([1.0, 2.0, 5.0])
        scenario.exogenous.y = lambda t: 0.0
        scenario.exogenous.generate = MagicMock()
        scenario.kernel.kappa.return_value = 0.0
        scenario.kernel.lambda_.return_value = 0.5
        scenario.kernel.lambda_upper_bound.return_value = 1.0
        scenario.kernel.expected_reward.return_value = 0.5
        scenario.gamma.initiate_context.return_value = 0.0
        scenario.gamma.update_context.side_effect = lambda ctx, x, u, t, t_past: ctx + 1.0
        scenario.gamma.current_context.side_effect = lambda ctx, t, t_past: ctx
        scenario.competition.sample.return_value = 0.3
        scenario.auction.win.side_effect = lambda bid, comp: bid > comp
        scenario.auction.payment.side_effect = lambda bid, comp: comp
        scenario.delay_sampler.sample.return_value = []
        scenario.max_history = 1
        scenario.conversion_value = 1.0
        return AuctionEnv(scenario)

    @patch("dynamauction.simulator.AuctionEnv")
    def test_env_wrapper_applied(self, mock_env_class, mock_scenario, mock_env_output):
        """env_wrappers callables are applied to the env in _make_env."""
        inner_env = mock_env_class.return_value
        wrapped_env = MagicMock()
        wrapped_env.reset.return_value = ({}, {})
        wrapped_env.step.return_value = ({}, 0.0, True, False, mock_env_output)

        mock_wrapper = MagicMock(return_value=wrapped_env)

        sim = Simulator(
            mock_scenario, lambda x: 1.0, {"n_runs": 1, "seed": 0},
            env_wrappers=[mock_wrapper],
        )
        sim._run_single(run_id=0, policy=lambda x: 1.0, seed=0)

        mock_wrapper.assert_called_once_with(inner_env)
        wrapped_env.step.assert_called()

    @patch("dynamauction.simulator.AuctionEnv")
    def test_env_wrappers_applied_in_order(self, mock_env_class, mock_scenario, mock_env_output):
        """Multiple env_wrappers are composed left-to-right."""
        call_order = []

        base = mock_env_class.return_value
        mid = MagicMock()
        outer = MagicMock()
        outer.reset.return_value = ({}, {})
        outer.step.return_value = ({}, 0.0, True, False, mock_env_output)

        def wrap1(env):
            call_order.append(("wrap1", env))
            return mid

        def wrap2(env):
            call_order.append(("wrap2", env))
            return outer

        sim = Simulator(
            mock_scenario, lambda x: 0.5, {"n_runs": 1, "seed": 0},
            env_wrappers=[wrap1, wrap2],
        )
        sim._run_single(run_id=0, policy=lambda x: 0.5, seed=0)

        assert call_order[0] == ("wrap1", base)
        assert call_order[1] == ("wrap2", mid)

    @patch("dynamauction.simulator.AuctionEnv")
    def test_action_passed_as_numpy_array(self, mock_env_class, mock_scenario, mock_env_output):
        """env.step receives the bid wrapped in a float32 numpy array of shape (1,)."""
        mock_env = mock_env_class.return_value
        mock_env.reset.return_value = ({}, {})
        mock_env.step.return_value = ({}, 0.0, True, False, mock_env_output)

        sim = Simulator(mock_scenario, lambda x: 1.5, {"n_runs": 1, "seed": 0})
        sim._run_single(run_id=0, policy=lambda x: 1.5, seed=0)

        action = mock_env.step.call_args[0][0]
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
        assert action.shape == (1,)
        assert action[0] == pytest.approx(1.5)

    def test_sb3_policy_wrapper_end_to_end(self, full_env):
        """SB3PolicyWrapper drives a complete episode through the simulator."""
        from dynamauction.wrappers import SB3PolicyWrapper

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([1.0], dtype=np.float32), None)

        policy = SB3PolicyWrapper(mock_model, full_env.observation_space)
        sim = Simulator(full_env.scenario, policy, {"n_runs": 2, "seed": 0})
        a_df, d_df, c_df, s_df = sim.run()

        assert len(a_df) > 0
        assert len(s_df) == 2
        assert mock_model.predict.call_count > 0

    def test_sb3_policy_wrapper_with_mask_wrapper(self, full_env):
        """SB3PolicyWrapper(exclude_keys) + MaskExpectedRewardWrapper on the env is consistent."""
        from dynamauction.wrappers import SB3PolicyWrapper, MaskExpectedRewardWrapper

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.8], dtype=np.float32), None)

        policy = SB3PolicyWrapper(
            mock_model,
            full_env.observation_space,
            exclude_keys=["expected_reward"],
        )
        sim = Simulator(
            full_env.scenario, policy, {"n_runs": 1, "seed": 0},
            env_wrappers=[MaskExpectedRewardWrapper],
        )
        a_df, _, _, _ = sim.run()

        assert len(a_df) > 0
        assert mock_model.predict.call_count > 0