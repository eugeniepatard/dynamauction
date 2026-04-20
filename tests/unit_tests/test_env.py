import pytest
import numpy as np
from unittest.mock import MagicMock
from gymnasium.utils.env_checker import check_env
from dynamicauc import AuctionEnv, OBSERVATION_KEYS
from dynamicauc.primitives import PoissonProbe

@pytest.fixture
def mock_scenario():
    # Use a plain MagicMock or a loosely defined spec
    scenario = MagicMock()

    # 1. Manually create the nested mocks that the Env expects
    scenario.user_feature_generator = MagicMock()
    scenario.display_sampler = MagicMock()
    scenario.horizon_generator = MagicMock()
    scenario.arrival = MagicMock()
    scenario.exogenous = MagicMock()
    scenario.kernel = MagicMock()
    scenario.competition = MagicMock()
    scenario.auction = MagicMock()
    scenario.probe = MagicMock()
    scenario.delay_sampler = MagicMock()

    # 2. Setup simple return values
    scenario.user_feature_generator.generate.return_value = np.array([1.0, 0.5])
    scenario.user_feature_generator.get_observable_features.return_value = np.array([1.0])
    scenario.user_feature_generator.observable_dim = 1

    scenario.display_sampler.sample.return_value = np.array([0.1])
    scenario.display_sampler.dim = 1

    scenario.horizon_generator.sample.return_value = 10.0
    scenario.arrival.sample_times.return_value = np.array([1.0, 2.0, 5.0])

    scenario.exogenous.y = lambda t: 0.0
    scenario.exogenous.generate = MagicMock()

    # Kernel properties
    scenario.kernel.kappa.return_value = 0.0
    scenario.kernel.lambda_.return_value = 1.0
    scenario.kernel.lambda_upper_bound.return_value = 1.0
    scenario.kernel.expected_reward.return_value = 0.5

    scenario.gamma = MagicMock()
    scenario.gamma.initiate_context.return_value = 0.0
    scenario.gamma.update_context.side_effect = lambda ctx, x, u, t, t_past: ctx + 1.0

    # Competition & Auction
    scenario.competition.sample.return_value = 0.5
    scenario.auction.win.side_effect = lambda bid, comp: bid > comp
    scenario.auction.payment.side_effect = lambda bid, comp: comp

    # Delay sampler returns no delays by default
    scenario.delay_sampler.sample.return_value = []

    scenario.max_history = 5
    scenario.conversion_value = 10.0

    return scenario

def test_gym_api_conformity(mock_scenario):
    """Check if the environment follows Gymnasium standards."""
    env = AuctionEnv(mock_scenario)
    # This utility checks for observation/action space consistency
    check_env(env)

def test_reset_logic(mock_scenario):
    env = AuctionEnv(mock_scenario)
    obs, info = env.reset(seed=42)
    
    for key in OBSERVATION_KEYS:
        assert key in obs
    assert obs["t"] == 1.0
    assert obs["ctx"] == 0.0
    assert obs["user_features"].shape == (1,)
    assert obs["user_features"][0] == 1.0
    assert obs["display_features"].shape == (1,)
    assert obs["display_features"][0] == 0.1
    assert obs["y_now"] == 0.0
    assert obs["step"] == 0
    assert info["horizon"] == 10.0
    assert len(info["auction_times"]) == 3

    assert env.rng is not None
    assert env.state is not None
    assert env.user_features is not None
    assert env.pending_conversions == []
    assert env.matured_conversions_reward == 0.0
    assert env.matured_conversions_times == []
    assert env.auction_times is not None
    assert env.horizon == 10.0


def test_delayed_reward_out_of_window(mock_scenario):
    """
    Test that a reward scheduled far in the future doesn't mature immediately.
    """
    env = AuctionEnv(mock_scenario)
    env.reset(seed=42)
    
    # 1. Place a high bid to ensure a win
    # We force the delayed sampler to return a delay of 0.5s
    # so conversion occurs at 1.0 (t) + 0.5 (s) = 1.5.
    # This falls between t=1.0 and t_next=2.0.
    env.delay_sampler = MagicMock()
    env.delay_sampler.sample.return_value = [0.5]
    
    obs, reward, _, _, info = env.step(action=1.0)
    
    # Since the conversion happens at 1.5, it matures during the step transition to t=2.0
    assert info["auction"]["won"] is True
    assert reward == - 0.5  # Immediate conversion - Payment
    assert obs["t"] == 2.0
    assert obs["cumulative_reward_since_last_step"] == mock_scenario.conversion_value

def test_episode_termination(mock_scenario):
    env = AuctionEnv(mock_scenario)
    env.reset()
    
    # There are 3 auction times. 3 steps should end it.
    _, _, done1, _, _ = env.step(1.0)
    _, _, done2, _, _ = env.step(1.0)
    obs, _, done3, _, _ = env.step(1.0)
    
    assert done1 is False
    assert done2 is False
    assert done3 is True
    assert obs["t"] == env.horizon

# --- Statistical Test for PoissonProbe (Ogata thinning) ---

def test_poisson_probe_statistical_properties():
    """
    PoissonProbe with constant lambda=1.0 over a large horizon should produce
    counts that follow Poisson(1.0 * horizon).
    """
    kernel = MagicMock()
    kernel.lambda_.return_value = 1.0
    kernel.lambda_upper_bound.return_value = 1.0

    probe = PoissonProbe(kernel=kernel)
    rng = np.random.default_rng(42)
    u, x = np.array([]), np.array([])
    y_func = lambda t: 0
    horizon = 100.0

    sample_counts = []
    for _ in range(200):
        delays = probe.sample_all(u, x, 0.0, y_func, 0.0, horizon, rng)
        sample_counts.append(len(delays))

    mean_samples = np.mean(sample_counts)
    assert 95 <= mean_samples <= 105