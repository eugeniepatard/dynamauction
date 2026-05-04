"""
Tests for dynamicauc/baseline/baseline_algo.py:
  - GreedyMyopicPolicy
  - ShadingPolicy
  - MPCPolicy
  - MPCOraclePolicy
  - MPCTimeBinnedPolicy
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from dynamicauc import AuctionEnv
from dynamicauc.baseline.baseline_algo import (
    GreedyMyopicPolicy,
    MPCOraclePolicy,
    MPCPolicy,
    MPCTimeBinnedPolicy,
    ShadingPolicy,
)
from dynamicauc.baseline.estimators import SimulationEstimator, MLPValuationHead
from dynamicauc.baseline.policies import ConstantPolicy

@pytest.fixture
def mock_estimator():
    est = MagicMock(spec=SimulationEstimator)
    est.estimate.return_value = 0.6
    return est


@pytest.fixture
def live_env(base_env):
    obs, _ = base_env.reset(seed=0)
    return base_env, obs


@pytest.fixture
def mock_mlp():
    mlp = MagicMock(spec=MLPValuationHead)
    mlp.predict.return_value = 0.5
    return mlp

class TestGreedyMyopicPolicy:

    def test_bid_equals_delta(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.6
        policy = GreedyMyopicPolicy(mock_estimator)
        assert policy.bid({}) == pytest.approx(0.6)

    def test_bid_clips_negative_delta_to_zero(self, mock_estimator):
        mock_estimator.estimate.return_value = -0.4
        policy = GreedyMyopicPolicy(mock_estimator)
        assert policy.bid({}) == pytest.approx(0.0)

    def test_bid_returns_float(self, mock_estimator):
        policy = GreedyMyopicPolicy(mock_estimator)
        assert isinstance(policy.bid({}), float)

    def test_set_env_delegates_to_estimator(self, mock_estimator, base_env):
        policy = GreedyMyopicPolicy(mock_estimator)
        policy.set_env(base_env)
        mock_estimator.set_env.assert_called_once_with(base_env)

    def test_estimate_called_with_obs(self, mock_estimator):
        obs = {"key": 1}
        policy = GreedyMyopicPolicy(mock_estimator)
        policy.bid(obs)
        mock_estimator.estimate.assert_called_once_with(obs)

    def test_end_to_end_with_real_env(self, live_env):
        env, obs = live_env
        est = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=1)
        policy = GreedyMyopicPolicy(est)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert isinstance(bid, float)
        assert bid >= 0.0

class TestShadingPolicy:

    def test_bid_is_alpha_times_delta(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.6
        policy = ShadingPolicy(mock_estimator, alpha=0.5)
        assert policy.bid({}) == pytest.approx(0.3)

    def test_bid_clips_negative_to_zero(self, mock_estimator):
        mock_estimator.estimate.return_value = -1.0
        policy = ShadingPolicy(mock_estimator, alpha=0.8)
        assert policy.bid({}) == pytest.approx(0.0)

    def test_alpha_one_equals_greedy(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.6
        greedy = GreedyMyopicPolicy(mock_estimator)
        shading = ShadingPolicy(mock_estimator, alpha=1.0)
        assert shading.bid({}) == pytest.approx(greedy.bid({}))

    def test_alpha_gt_one_amplifies_bid(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.5
        assert ShadingPolicy(mock_estimator, alpha=2.0).bid({}) == pytest.approx(1.0)

    def test_invalid_alpha_raises(self, mock_estimator):
        with pytest.raises(ValueError, match="alpha must be positive"):
            ShadingPolicy(mock_estimator, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            ShadingPolicy(mock_estimator, alpha=-1.0)

    def test_default_alpha_is_one(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.4
        policy = ShadingPolicy(mock_estimator)
        assert policy.alpha == pytest.approx(1.0)
        assert policy.bid({}) == pytest.approx(0.4)

    def test_set_env_delegates_to_estimator(self, mock_estimator, base_env):
        policy = ShadingPolicy(mock_estimator, alpha=0.8)
        policy.set_env(base_env)
        mock_estimator.set_env.assert_called_once_with(base_env)

    def test_bid_returns_float(self, mock_estimator):
        assert isinstance(ShadingPolicy(mock_estimator, alpha=0.8).bid({}), float)

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0, 1.2])
    def test_bid_always_nonnegative(self, mock_estimator, alpha):
        mock_estimator.estimate.return_value = 0.5
        assert ShadingPolicy(mock_estimator, alpha=alpha).bid({}) >= 0.0


class TestMPCPolicy:

    def test_bid_raises_without_set_env(self, mock_mlp):
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1)
        with pytest.raises(RuntimeError, match="set_env"):
            policy.bid({})

    def test_set_env_stores_env(self, mock_mlp, base_env):
        policy = MPCPolicy(v0=mock_mlp)
        policy.set_env(base_env)
        assert policy._env is base_env

    def test_bid_returns_float(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        assert isinstance(policy.bid(obs), float)

    def test_bid_is_nonnegative(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        assert policy.bid(obs) >= 0.0

    def test_alpha_log_appended_on_each_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        assert policy.alpha_log == []
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        policy.bid(obs)
        assert len(policy.alpha_log) == 2

    def test_chosen_alpha_is_from_candidates(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0, 1.5]
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=candidates, n_rollouts=2,
                           planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        assert policy.alpha_log[-1] in candidates

    def test_bid_equals_alpha_star_times_v0(self, mock_mlp, live_env):
        env, obs = live_env
        mock_mlp.predict.return_value = 0.4
        # Single candidate: alpha* must be that candidate
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[2.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert bid == pytest.approx(2.0 * 0.4, abs=1e-5)

    def test_alpha_candidates_stored(self, mock_mlp):
        candidates = [0.6, 0.8, 1.0]
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=candidates)
        assert policy.alpha_candidates == candidates

    def test_default_alpha_candidates(self, mock_mlp):
        policy = MPCPolicy(v0=mock_mlp)
        assert len(policy.alpha_candidates) > 0

    def test_v0_predict_called_during_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        mock_mlp.predict.assert_called()

    def test_planning_horizon_zero_stops_after_first_step(self, mock_mlp, live_env):
        # With H=0, rollout terminates after 1 step, so MPC only sees immediate reward.
        # This is tested indirectly: bid should still return a valid float and not hang.
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert np.isfinite(bid)

    def test_end_to_end_real_mlp(self, live_env):
        from dynamicauc.baseline.utils import _obs_to_features
        env, obs = live_env
        dim = _obs_to_features(obs).shape[0]
        mlp = MLPValuationHead(input_dim=dim, hidden_dims=(8,))
        policy = MPCPolicy(v0=mlp, alpha_candidates=[0.8, 1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert isinstance(bid, float)
        assert bid >= 0.0


class TestMPCOraclePolicy:

    def test_bid_raises_without_set_env(self, mock_mlp):
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1)
        with pytest.raises(RuntimeError, match="set_env"):
            policy.bid({})

    def test_set_env_stores_env(self, mock_mlp, base_env):
        policy = MPCOraclePolicy(v0=mock_mlp)
        policy.set_env(base_env)
        assert policy._env is base_env

    def test_bid_returns_float(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        assert isinstance(policy.bid(obs), float)

    def test_bid_is_nonnegative(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        assert policy.bid(obs) >= 0.0

    def test_alpha_log_grows_with_each_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        assert policy.alpha_log == []
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        policy.bid(obs)
        assert len(policy.alpha_log) == 2

    def test_chosen_alpha_is_from_candidates(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0, 1.5]
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=candidates, n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        assert policy.alpha_log[-1] in candidates

    def test_bid_equals_alpha_star_times_v0(self, mock_mlp, live_env):
        env, obs = live_env
        mock_mlp.predict.return_value = 0.4
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[2.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert bid == pytest.approx(2.0 * 0.4, abs=1e-5)

    def test_alpha_candidates_stored(self, mock_mlp):
        candidates = [0.6, 0.8, 1.0]
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=candidates)
        assert policy.alpha_candidates == candidates

    def test_v0_predict_called_during_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        mock_mlp.predict.assert_called()


class TestMPCTimeBinnedPolicy:

    def test_bid_raises_without_set_env(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1)
        with pytest.raises(RuntimeError):
            policy.bid({})

    def test_set_env_stores_env(self, mock_mlp, base_env):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp)
        policy.set_env(base_env)
        assert policy._env is base_env

    def test_bid_returns_float(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_bins=2,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        assert isinstance(policy.bid(obs), float)

    def test_bid_is_nonnegative(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_bins=2,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        assert policy.bid(obs) >= 0.0

    def test_alpha_log_contains_tuples(self, mock_mlp, live_env):
        env, obs = live_env
        n_bins = 2
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_bins=n_bins,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        assert isinstance(policy.alpha_log[0], tuple)
        assert len(policy.alpha_log[0]) == n_bins

    def test_chosen_alpha_vec_from_candidates(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0]
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=candidates, n_bins=2,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        alpha_vec = policy.alpha_log[-1]
        assert all(a in candidates for a in alpha_vec)

    def test_n_bins_stored(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=3)
        assert policy.n_bins == 3

    def test_alpha_log_grows_with_each_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_bins=1,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        assert policy.alpha_log == []
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        policy.bid(obs)
        assert len(policy.alpha_log) == 2
