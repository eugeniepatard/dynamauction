import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from unittest.mock import MagicMock

from dynamauction import AuctionEnv, FlatAuctionObsWrapper, BidScaleWrapper, MaskExpectedRewardWrapper
from dynamauction.wrappers import SB3PolicyWrapper


@pytest.fixture
def flat_env(base_env):
    return FlatAuctionObsWrapper(base_env)


@pytest.fixture
def scaled_env(flat_env):
    return BidScaleWrapper(flat_env, max_bid=5.0)

class TestFlatAuctionObsWrapperSpace:

    def test_observation_space_is_box(self, flat_env):
        assert isinstance(flat_env.observation_space, spaces.Box)

    def test_observation_space_dtype(self, flat_env):
        assert flat_env.observation_space.dtype == np.float32

    def test_observation_space_is_1d(self, flat_env):
        assert flat_env.observation_space.shape == (len(flat_env.observation_space.low),)

    def test_observation_space_dim_correct(self, flat_env, base_env):
        """
        Expected flat dim:
          t(1) + ctx(1) + step(1, not one-hot) + user_features(3)
          + display_features(3) + y_now(1) + expected_reward(1)
          + cumulative_reward_since_last_step(1) + conversion_times_since_last_step(1)
          = 13
        """
        orig = base_env.observation_space
        expected_dim = sum(
            1 if isinstance(sp, spaces.Discrete) else int(np.prod(sp.shape))
            for sp in orig.spaces.values()
        )
        assert flat_env.observation_space.shape[0] == expected_dim

    def test_step_not_one_hot_encoded(self, flat_env, base_env):
        """Discrete(100000) must map to 1 dim, not 100000."""
        n_discrete = sum(
            1 for sp in base_env.observation_space.spaces.values()
            if isinstance(sp, spaces.Discrete)
        )
        orig_box_dim = sum(
            int(np.prod(sp.shape))
            for sp in base_env.observation_space.spaces.values()
            if isinstance(sp, spaces.Box)
        )
        assert flat_env.observation_space.shape[0] == orig_box_dim + n_discrete

class TestFlatAuctionObsWrapperValues:

    def test_reset_returns_flat_array(self, flat_env):
        obs, _ = flat_env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert obs.dtype == np.float32

    def test_reset_shape_matches_space(self, flat_env):
        obs, _ = flat_env.reset(seed=0)
        assert obs.shape == flat_env.observation_space.shape

    def test_step_returns_flat_array(self, flat_env):
        flat_env.reset(seed=0)
        obs, _, _, _, _ = flat_env.step(np.array([0.5], dtype=np.float32))
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert obs.shape == flat_env.observation_space.shape

    def test_step_field_preserved(self, flat_env, base_env):
        """The 'step' value (discrete index) must appear as a float, not one-hot."""
        obs_flat, _ = flat_env.reset(seed=42)
        obs_raw, _  = base_env.reset(seed=42)

        # Reconstruct which index corresponds to 'step' in the flat vector
        keys = list(base_env.observation_space.spaces.keys())
        offset = 0
        for key in keys:
            sp = base_env.observation_space.spaces[key]
            dim = 1 if isinstance(sp, spaces.Discrete) else int(np.prod(sp.shape))
            if key == "step":
                assert obs_flat[offset] == float(obs_raw["step"])
                break
            offset += dim

    def test_box_values_preserved(self, flat_env, base_env):
        """Box values must be concatenated without modification."""
        obs_flat, _ = flat_env.reset(seed=7)
        obs_raw, _  = base_env.reset(seed=7)

        keys   = list(base_env.observation_space.spaces.keys())
        offset = 0
        for key in keys:
            sp  = base_env.observation_space.spaces[key]
            if isinstance(sp, spaces.Discrete):
                offset += 1
                continue
            dim = int(np.prod(sp.shape))
            chunk = obs_flat[offset: offset + dim]
            np.testing.assert_array_almost_equal(
                chunk, np.asarray(obs_raw[key]).ravel().astype(np.float32)
            )
            offset += dim

    def test_gymnasium_conformity(self, flat_env):
        check_env(flat_env)

class TestBidScaleWrapperSpace:

    def test_action_space_is_box(self, scaled_env):
        assert isinstance(scaled_env.action_space, spaces.Box)

    def test_action_space_lower_bound(self, scaled_env):
        assert scaled_env.action_space.low[0] == pytest.approx(0.0)

    def test_action_space_upper_bound_default(self, scaled_env):
        assert scaled_env.action_space.high[0] == pytest.approx(5.0)

    def test_action_space_upper_bound_custom(self, flat_env):
        env = BidScaleWrapper(flat_env, max_bid=2.0)
        assert env.action_space.high[0] == pytest.approx(2.0)

    def test_action_space_shape(self, scaled_env):
        assert scaled_env.action_space.shape == (1,)

    def test_action_space_dtype(self, scaled_env):
        assert scaled_env.action_space.dtype == np.float32

class TestBidScaleWrapperClipping:

    @pytest.mark.parametrize("bid", [0.0, 0.5, 2.5, 5.0])
    def test_bid_in_range_passes_through(self, scaled_env, bid):
        """Bids within [0, max_bid] should be unchanged."""
        scaled_env.reset(seed=0)
        clipped = scaled_env.action(np.array([bid], dtype=np.float32))
        assert clipped[0] == pytest.approx(bid)

    def test_bid_above_max_clipped(self, scaled_env):
        scaled_env.reset(seed=0)
        clipped = scaled_env.action(np.array([100.0], dtype=np.float32))
        assert clipped[0] == pytest.approx(5.0)

    def test_bid_below_zero_clipped(self, scaled_env):
        scaled_env.reset(seed=0)
        clipped = scaled_env.action(np.array([-3.0], dtype=np.float32))
        assert clipped[0] == pytest.approx(0.0)

    def test_action_output_dtype(self, scaled_env):
        scaled_env.reset(seed=0)
        clipped = scaled_env.action(np.array([2.0], dtype=np.float32))
        assert clipped.dtype == np.float32


class TestBidScaleWrapperEndToEnd:

    def test_reset_returns_flat_obs(self, scaled_env):
        obs, _ = scaled_env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1

    def test_step_with_valid_bid(self, scaled_env):
        scaled_env.reset(seed=0)
        obs, reward, done, truncated, info = scaled_env.step(
            np.array([1.0], dtype=np.float32)
        )
        assert obs.shape == scaled_env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_with_oversized_bid_clipped(self, scaled_env):
        """An oversized bid is clipped; env must not raise."""
        scaled_env.reset(seed=0)
        obs, reward, done, _, info = scaled_env.step(
            np.array([1e6], dtype=np.float32)
        )
        assert info["auction"]["bid"] == pytest.approx(5.0)

    def test_gymnasium_conformity(self, scaled_env):
        check_env(scaled_env)

class TestMaskExpectedRewardWrapper:

    _MASKED_KEY = "expected_reward"

    @pytest.fixture
    def masked_env(self, base_env):
        return MaskExpectedRewardWrapper(base_env)

    @pytest.fixture
    def masked_flat_env(self, masked_env):
        return FlatAuctionObsWrapper(masked_env)

    # ── observation space ────────────────────────────────────────

    def test_masked_key_absent_from_space(self, masked_env):
        assert self._MASKED_KEY not in masked_env.observation_space.spaces

    def test_other_keys_present_in_space(self, masked_env, base_env):
        remaining = set(base_env.observation_space.spaces) - {self._MASKED_KEY}
        assert remaining == set(masked_env.observation_space.spaces)

    def test_space_dim_reduced_by_one(self, masked_env, base_env):
        """expected_reward is Box(shape=(1,)) so flat dim shrinks by exactly 1."""
        raw_dim = sum(
            1 if isinstance(sp, spaces.Discrete) else int(np.prod(sp.shape))
            for sp in base_env.observation_space.spaces.values()
        )
        masked_dim = sum(
            1 if isinstance(sp, spaces.Discrete) else int(np.prod(sp.shape))
            for sp in masked_env.observation_space.spaces.values()
        )
        assert masked_dim == raw_dim - 1

    # ── obs dict returned by reset / step ───────────────────────

    def test_reset_obs_excludes_masked_key(self, masked_env):
        obs, _ = masked_env.reset(seed=0)
        assert self._MASKED_KEY not in obs

    def test_step_obs_excludes_masked_key(self, masked_env):
        masked_env.reset(seed=0)
        obs, _, _, _, _ = masked_env.step(np.array([0.5], dtype=np.float32))
        assert self._MASKED_KEY not in obs

    def test_reset_other_values_unchanged(self, masked_env, base_env):
        """Values for the remaining keys must equal the raw env values."""
        obs_masked, _ = masked_env.reset(seed=42)
        obs_raw,    _ = base_env.reset(seed=42)
        for key in obs_masked:
            np.testing.assert_array_equal(obs_masked[key], obs_raw[key])

    def test_step_other_values_unchanged(self, masked_env):
        # masked_env.env IS base_env — they share the same underlying object.
        # Step the raw env first (capturing the raw obs), then reset to the same
        # state and step through the wrapper to get the masked obs.
        masked_env.reset(seed=7)
        obs_raw, _, _, _, _ = masked_env.env.step(np.array([1.0], dtype=np.float32))

        masked_env.reset(seed=7)
        obs_masked, _, _, _, _ = masked_env.step(np.array([1.0], dtype=np.float32))

        for key in obs_masked:
            np.testing.assert_array_equal(obs_masked[key], obs_raw[key])

    # ── stacked with FlatAuctionObsWrapper ───────────────────────

    def test_flat_dim_excludes_expected_reward(self, masked_flat_env, flat_env):
        """Flat obs after masking must be 1 element shorter than without masking."""
        assert masked_flat_env.observation_space.shape[0] == \
               flat_env.observation_space.shape[0] - 1

    def test_flat_reset_shape_matches_space(self, masked_flat_env):
        obs, _ = masked_flat_env.reset(seed=0)
        assert obs.shape == masked_flat_env.observation_space.shape

    def test_flat_step_shape_matches_space(self, masked_flat_env):
        masked_flat_env.reset(seed=0)
        obs, _, _, _, _ = masked_flat_env.step(np.array([0.5], dtype=np.float32))
        assert obs.shape == masked_flat_env.observation_space.shape

    # ── SB3PolicyWrapper with exclude_keys ───────────────────────

    def test_sb3_wrapper_exclude_matches_mask_dim(self, masked_flat_env, base_env):
        """SB3PolicyWrapper(exclude_keys=[...]) must produce the same flat dim
        as MaskExpectedRewardWrapper → FlatAuctionObsWrapper."""
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([1.0]), None)

        wrapper = SB3PolicyWrapper(
            mock_model,
            base_env.observation_space,
            exclude_keys=[self._MASKED_KEY],
        )
        obs_raw, _ = base_env.reset(seed=0)
        flat = wrapper._flatten(obs_raw)
        assert flat.shape[0] == masked_flat_env.observation_space.shape[0]

    def test_sb3_wrapper_exclude_values_match(self, masked_flat_env, base_env):
        """The flat vector from SB3PolicyWrapper must equal the one from
        MaskExpectedRewardWrapper → FlatAuctionObsWrapper."""
        from unittest.mock import MagicMock
        mock_model = MagicMock()

        wrapper = SB3PolicyWrapper(
            mock_model,
            base_env.observation_space,
            exclude_keys=[self._MASKED_KEY],
        )
        obs_raw,    _ = base_env.reset(seed=5)
        obs_masked, _ = masked_flat_env.reset(seed=5)

        np.testing.assert_array_almost_equal(wrapper._flatten(obs_raw), obs_masked)

    # ── gymnasium conformity ─────────────────────────────────────

    def test_gymnasium_conformity(self, masked_flat_env):
        check_env(masked_flat_env)


class TestStackedWrappers:

    def test_full_episode_runs(self, scaled_env):
        obs, _ = scaled_env.reset(seed=99)
        done = False
        steps = 0
        while not done:
            action = scaled_env.action_space.sample()
            obs, _, done, _, _ = scaled_env.step(action)
            steps += 1
        assert steps > 0

    def test_obs_always_in_space(self, scaled_env):
        """Every observation returned during an episode must lie inside the space."""
        obs, _ = scaled_env.reset(seed=3)
        assert scaled_env.observation_space.contains(obs)
        done = False
        while not done:
            action = scaled_env.action_space.sample()
            obs, _, done, _, _ = scaled_env.step(action)
            assert scaled_env.observation_space.contains(obs), \
                f"Observation {obs} out of space bounds"

    def test_action_always_in_space(self, scaled_env):
        """Sampled actions must always be contained in the action space."""
        for _ in range(50):
            action = scaled_env.action_space.sample()
            assert scaled_env.action_space.contains(action)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_vec_env(mock_scenario, n=1):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make():
        env = AuctionEnv(mock_scenario)
        env = FlatAuctionObsWrapper(env)
        env = BidScaleWrapper(env)
        env = Monitor(env)
        return env

    return DummyVecEnv([make] * n)


# ── SB3PolicyWrapper ──────────────────────────────────────────────────────────

class TestSB3PolicyWrapper:

    @pytest.fixture
    def mock_model(self):
        m = MagicMock()
        m.predict.return_value = (np.array([1.5]), None)
        return m

    @pytest.fixture
    def wrapper(self, mock_model, base_env):
        return SB3PolicyWrapper(mock_model, base_env.observation_space)

    # ── _flatten ─────────────────────────────────────────────────────────────

    def test_flatten_shape_matches_flat_wrapper(self, wrapper, base_env, flat_env):
        obs, _ = base_env.reset(seed=0)
        expected, _ = flat_env.reset(seed=0)
        assert wrapper._flatten(obs).shape == expected.shape

    def test_flatten_values_match_flat_wrapper(self, wrapper, base_env, flat_env):
        obs, _ = base_env.reset(seed=0)
        expected, _ = flat_env.reset(seed=0)
        np.testing.assert_array_almost_equal(wrapper._flatten(obs), expected)

    def test_flatten_exclude_key_reduces_dim_by_one(self, mock_model, base_env, flat_env):
        w = SB3PolicyWrapper(mock_model, base_env.observation_space,
                             exclude_keys=["expected_reward"])
        obs, _ = base_env.reset(seed=0)
        assert w._flatten(obs).shape[0] == flat_env.observation_space.shape[0] - 1

    def test_flatten_output_is_float32(self, wrapper, base_env):
        obs, _ = base_env.reset(seed=0)
        assert wrapper._flatten(obs).dtype == np.float32

    # ── bid without obs_rms ──────────────────────────────────────────────────

    def test_bid_returns_float(self, wrapper, base_env):
        obs, _ = base_env.reset(seed=0)
        assert isinstance(wrapper.bid(obs), float)

    def test_bid_matches_model_output(self, wrapper, mock_model, base_env):
        mock_model.predict.return_value = (np.array([2.7]), None)
        obs, _ = base_env.reset(seed=0)
        assert wrapper.bid(obs) == pytest.approx(2.7)

    def test_bid_calls_predict_with_deterministic_true(self, wrapper, mock_model, base_env):
        obs, _ = base_env.reset(seed=0)
        wrapper.bid(obs)
        mock_model.predict.assert_called_once()
        assert mock_model.predict.call_args[1]["deterministic"] is True

    def test_bid_predict_input_is_1d_float32(self, wrapper, mock_model, base_env):
        obs, _ = base_env.reset(seed=0)
        wrapper.bid(obs)
        inp = mock_model.predict.call_args[0][0]
        assert inp.ndim == 1
        assert inp.dtype == np.float32

    def test_bid_without_obs_rms_passes_raw_flat_obs(self, wrapper, mock_model, base_env, flat_env):
        obs_raw,  _ = base_env.reset(seed=0)
        obs_flat, _ = flat_env.reset(seed=0)
        wrapper.bid(obs_raw)
        np.testing.assert_array_almost_equal(mock_model.predict.call_args[0][0], obs_flat)

    # ── bid with obs_rms (VecNormalize statistics) ───────────────────────────

    @pytest.fixture
    def obs_rms(self, flat_env):
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        flat_dim = flat_env.observation_space.shape[0]
        rms = RunningMeanStd(shape=(flat_dim,))
        rms.mean = np.full(flat_dim, 0.5, dtype=np.float64)
        rms.var  = np.full(flat_dim, 4.0, dtype=np.float64)   # std = 2
        return rms

    def test_bid_with_obs_rms_normalizes_obs(self, mock_model, base_env, obs_rms):
        wrapper = SB3PolicyWrapper(mock_model, base_env.observation_space,
                                   obs_rms=obs_rms, clip_obs=10.0)
        obs_raw, _ = base_env.reset(seed=0)
        wrapper.bid(obs_raw)

        raw_flat = SB3PolicyWrapper(mock_model, base_env.observation_space)._flatten(obs_raw)
        expected = np.clip(
            (raw_flat - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
            -10.0, 10.0,
        ).astype(np.float32)
        np.testing.assert_array_almost_equal(mock_model.predict.call_args[0][0], expected)

    def test_bid_obs_rms_clips_extreme_values(self, mock_model, base_env, flat_env):
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        flat_dim = flat_env.observation_space.shape[0]
        rms = RunningMeanStd(shape=(flat_dim,))
        rms.mean = np.zeros(flat_dim, dtype=np.float64)
        rms.var  = np.full(flat_dim, 1e-10, dtype=np.float64)  # tiny var → huge normalised values

        wrapper = SB3PolicyWrapper(mock_model, base_env.observation_space,
                                   obs_rms=rms, clip_obs=3.0)
        obs_raw, _ = base_env.reset(seed=0)
        wrapper.bid(obs_raw)

        inp = mock_model.predict.call_args[0][0]
        assert np.all(np.abs(inp) <= 3.0 + 1e-6)

    def test_bid_custom_clip_obs_respected(self, mock_model, base_env, flat_env):
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        flat_dim = flat_env.observation_space.shape[0]
        rms = RunningMeanStd(shape=(flat_dim,))
        rms.mean = np.zeros(flat_dim)
        rms.var  = np.full(flat_dim, 1e-10)

        for clip in (1.0, 5.0, 20.0):
            mock_model.reset_mock()
            w = SB3PolicyWrapper(mock_model, base_env.observation_space,
                                 obs_rms=rms, clip_obs=clip)
            obs_raw, _ = base_env.reset(seed=0)
            w.bid(obs_raw)
            inp = mock_model.predict.call_args[0][0]
            assert np.all(np.abs(inp) <= clip + 1e-6)


# ── VecNormalize ──────────────────────────────────────────────────────────────

class TestVecNormalize:

    @pytest.fixture
    def vec_env(self, mock_scenario):
        env = _make_vec_env(mock_scenario)
        yield env
        env.close()

    @pytest.fixture
    def vn_train(self, vec_env):
        from stable_baselines3.common.vec_env import VecNormalize
        vn = VecNormalize(vec_env, norm_obs=True, norm_reward=True, training=True)
        yield vn
        vn.close()

    # ── obs normalisation statistics ─────────────────────────────────────────

    def test_obs_rms_updates_during_training(self, vn_train):
        initial_var = vn_train.obs_rms.var.copy()
        vn_train.reset()
        for _ in range(20):
            vn_train.step(vn_train.action_space.sample())
        # Variance must be non-trivial after seeing observations
        assert vn_train.obs_rms.var.sum() > 0.0
        assert not np.allclose(vn_train.obs_rms.var, initial_var)

    def test_obs_rms_frozen_when_training_false(self, vec_env):
        from stable_baselines3.common.vec_env import VecNormalize
        vn = VecNormalize(vec_env, norm_obs=True, training=False)
        vn.obs_rms.mean[:] = 99.0
        vn.obs_rms.var[:]  = 99.0
        vn.reset()
        for _ in range(20):
            vn.step(vn.action_space.sample())
        assert np.all(vn.obs_rms.mean == 99.0)
        assert np.all(vn.obs_rms.var  == 99.0)

    def test_normalised_obs_dtype_is_float32(self, vn_train):
        obs = vn_train.reset()
        assert obs.dtype == np.float32

    def test_normalised_obs_shape_matches_flat_env(self, vn_train, flat_env):
        obs = vn_train.reset()
        # VecEnv returns (n_envs, obs_dim); one env here
        assert obs.shape == (1, flat_env.observation_space.shape[0])

    # ── reward normalisation ─────────────────────────────────────────────────

    def test_norm_reward_true_attribute(self, vn_train):
        assert vn_train.norm_reward is True

    def test_norm_reward_false_attribute(self, vec_env):
        from stable_baselines3.common.vec_env import VecNormalize
        vn = VecNormalize(vec_env, norm_reward=False)
        assert vn.norm_reward is False

    def test_norm_reward_false_flag_is_propagated(self, vec_env):
        from stable_baselines3.common.vec_env import VecNormalize
        vn = VecNormalize(vec_env, norm_reward=False)
        assert vn.norm_reward is False
