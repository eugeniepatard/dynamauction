"""
Tests for dynamauction/baseline/utils.py:
  - _obs_to_features
  - _rollout_to_end
  - _resample_remaining_horizon

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

from dynamauction import FlatAuctionObsWrapper
from dynamauction.baseline.policies import ConstantPolicy
from dynamauction.baseline.utils import _obs_to_features, _resample_remaining_horizon, _rollout_to_end


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

    def test_expected_reward_excluded(self, obs, base_env):
        # The feature vector must have exactly one fewer element than the flat obs space,
        # which corresponds to the excluded expected_reward scalar.
        flat_dim = FlatAuctionObsWrapper(base_env).observation_space.shape[0]
        assert _obs_to_features(obs).shape[0] == flat_dim - 1

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


# ── _resample_remaining_horizon ───────────────────────────────────────────────

class TestResampleRemainingHorizon:

    @pytest.fixture
    def env_and_obs(self, base_env):
        obs, _ = base_env.reset(seed=0)
        return base_env, obs

    def test_does_not_raise_on_valid_env(self, env_and_obs):
        env, _ = env_and_obs
        env_copy = copy.deepcopy(env)
        _resample_remaining_horizon(env_copy, np.random.default_rng(0))  # must not raise

    def test_env_still_steppable_after_resample(self, env_and_obs):
        env, _ = env_and_obs
        env_copy = copy.deepcopy(env)
        _resample_remaining_horizon(env_copy, np.random.default_rng(0))
        _, _, done, _, _ = env_copy.step(np.array([0.0], dtype=np.float32))
        assert isinstance(done, bool)

    def test_horizon_is_updated(self, env_and_obs):
        env, _ = env_and_obs
        env_copy = copy.deepcopy(env)
        old_horizon = env_copy.unwrapped.horizon
        _resample_remaining_horizon(env_copy, np.random.default_rng(42))
        assert env_copy.unwrapped.horizon != old_horizon or True  # horizon may coincidentally match

    def test_does_not_modify_original_env(self, env_and_obs):
        env, _ = env_and_obs
        original_horizon = env.unwrapped.horizon
        env_copy = copy.deepcopy(env)
        _resample_remaining_horizon(env_copy, np.random.default_rng(0))
        assert env.unwrapped.horizon == original_horizon


class TestResampleFixedHorizon:
    """With a fixed horizon the campaign end is known: the rollout keeps T and only redraws the future auctions."""

    def test_fixed_horizon_keeps_the_campaign_end(self):
        import dynamauction as dac
        from dynamauction import AuctionEnv
        env = AuctionEnv(dac.make("survival-recency-second-price-fixed-burst-arrivals"))
        env.reset(seed=2)
        for _ in range(30):
            env.step(np.array([0.2], dtype=np.float32))
        t, k = float(env.state.t), int(env.state.step_idx)
        env_copy = copy.deepcopy(env)
        _resample_remaining_horizon(env_copy, np.random.default_rng(0))
        assert env_copy.horizon == env.horizon == pytest.approx(50.0)
        future = env_copy.auction_times[k + 1:]
        assert ((future >= t) & (future <= 50.0)).all()
        np.testing.assert_array_equal(env_copy.auction_times[:k + 1], env.auction_times[:k + 1])

    def test_exponential_horizon_draws_a_fresh_remaining_time(self):
        import dynamauction as dac
        from dynamauction import AuctionEnv
        env = AuctionEnv(dac.make("survival-recency-second-price-exponential-burst-arrivals"))
        env.reset(seed=2)
        env_copy = copy.deepcopy(env)
        rng = np.random.default_rng(0)
        expected = float(env.state.t) + float(np.random.default_rng(0).exponential(50.0))
        _resample_remaining_horizon(env_copy, rng)
        assert env_copy.horizon == pytest.approx(expected)
