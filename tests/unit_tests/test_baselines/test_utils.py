"""
Tests for dynamicauc/baseline/utils.py:
  - _obs_to_features
  - _rollout_to_end

Mock scenario setup (from conftest):
  competing_bid = 0.3, payment = competing_bid
  kappa = 0.0, delay_sampler returns [] → no conversions
  → winning a step yields reward = 0 - 0.3 = -0.3
  → losing a step yields reward = 0.0
"""
import copy

import numpy as np
import pytest
from unittest.mock import MagicMock

from dynamicauc import AuctionEnv, FlatAuctionObsWrapper
from dynamicauc.baseline.policies import ConstantPolicy
from dynamicauc.baseline.utils import _obs_to_features, _rollout_to_end


# ── _obs_to_features ──────────────────────────────────────────────────────────

class TestObsToFeatures:

    @pytest.fixture
    def obs(self, base_env):
        o, _ = base_env.reset(seed=0)
        return o

    def test_output_is_1d(self, obs):
        assert _obs_to_features(obs).ndim == 1

    def test_output_dtype_is_float32(self, obs):
        assert _obs_to_features(obs).dtype == np.float32

    def test_output_dim_excludes_expected_reward(self, obs, base_env):
        flat_dim = FlatAuctionObsWrapper(base_env).observation_space.shape[0]
        assert _obs_to_features(obs).shape[0] == flat_dim - 1

    def test_first_element_is_t(self, obs):
        out = _obs_to_features(obs)
        assert out[0] == pytest.approx(float(obs["t"].ravel()[0]))

    def test_user_features_present(self, obs):
        out = _obs_to_features(obs)
        user_feat = obs["user_features"].ravel().astype(np.float32)
        # user_features are at a fixed offset; verify all values appear somewhere
        for v in user_feat:
            assert v in out

    def test_display_features_present(self, obs):
        out = _obs_to_features(obs)
        disp_feat = obs["display_features"].ravel().astype(np.float32)
        for v in disp_feat:
            assert v in out

    def test_expected_reward_value_not_in_output(self, obs):
        # expected_reward = 0.5 from mock, and it is the only field with that exact value.
        # After excluding it, 0.5 must not appear anywhere in the feature vector.
        assert 0.5 not in _obs_to_features(obs)

    def test_output_is_finite(self, obs):
        assert np.all(np.isfinite(_obs_to_features(obs)))


class TestRolloutToEnd:

    @pytest.fixture
    def env_and_obs(self, base_env):
        obs, _ = base_env.reset(seed=0)
        return base_env, obs

    def test_returns_float(self, env_and_obs):
        env, obs = env_and_obs
        result = _rollout_to_end(copy.deepcopy(env), obs, ConstantPolicy(0.0))
        assert isinstance(result, float)

    def test_always_losing_gives_zero_total(self, env_and_obs):
        # bid=0.0 < competing_bid=0.3 → always lose → no payment, no conversion → reward=0
        env, obs = env_and_obs
        total = _rollout_to_end(copy.deepcopy(env), obs, ConstantPolicy(0.0))
        assert total == pytest.approx(0.0)

    def test_first_bid_overrides_policy_on_step_one(self, env_and_obs):
        # first_bid=0.0 → lose first step; compare with first_bid=1e12 → win first step
        env, obs = env_and_obs
        total_lose = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0), first_bid=0.0
        )
        total_win = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0), first_bid=1e12
        )
        # They differ because winning pays 0.3 with no conversion → total_win < total_lose
        assert total_lose != pytest.approx(total_win)

    def test_first_bid_none_uses_policy(self, env_and_obs):
        # With first_bid=None and ConstantPolicy(0.0), behaviour identical to first_bid=0.0
        env, obs = env_and_obs
        total_default = _rollout_to_end(copy.deepcopy(env), obs, ConstantPolicy(0.0))
        total_explicit = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0), first_bid=0.0
        )
        assert total_default == pytest.approx(total_explicit)

    def test_gross_first_step_adds_back_payment(self, env_and_obs):
        # When winning, gross_first_step=True adds back the payment (0.3)
        # → gross total = net total + payment
        env, obs = env_and_obs
        total_net   = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0),
            first_bid=1e12, gross_first_step=False,
        )
        total_gross = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0),
            first_bid=1e12, gross_first_step=True,
        )
        assert total_gross == pytest.approx(total_net + 0.3, abs=1e-5)

    def test_gross_first_step_irrelevant_when_losing(self, env_and_obs):
        # Payment is only added back on a win; if we lose, gross=True has no effect
        env, obs = env_and_obs
        total_net   = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0),
            first_bid=0.0, gross_first_step=False,
        )
        total_gross = _rollout_to_end(
            copy.deepcopy(env), obs, ConstantPolicy(0.0),
            first_bid=0.0, gross_first_step=True,
        )
        assert total_net == pytest.approx(total_gross)

    def test_policy_bid_called_after_first_step(self, env_and_obs):
        env, obs = env_and_obs
        mock_policy = MagicMock()
        mock_policy.bid.return_value = 0.0

        _rollout_to_end(copy.deepcopy(env), obs, mock_policy, first_bid=0.0)

        # Episode has 3 arrival steps; first is overridden by first_bid,
        # so policy.bid should be called for the remaining steps.
        assert mock_policy.bid.call_count >= 1
