import numpy as np
import pytest
from unittest.mock import MagicMock

from dynamauction import AuctionEnv


@pytest.fixture
def mock_scenario():
    scenario = MagicMock()

    scenario.user_feature_generator.generate.return_value = np.array([1.0, 0.5, 0.2])
    scenario.user_feature_generator.get_observable_features.return_value = np.array([1.0, 0.5, 0.2])
    scenario.user_feature_generator.observable_dim = 3

    scenario.display_sampler.sample.return_value = np.array([0.1, 0.2, 0.3])
    scenario.display_sampler.dim = 3

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

    return scenario


@pytest.fixture
def base_env(mock_scenario):
    return AuctionEnv(mock_scenario)
