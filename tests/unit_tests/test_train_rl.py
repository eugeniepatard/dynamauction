import numpy as np
import pytest
from unittest.mock import patch

from dynamauction import AuctionEnv, FlatAuctionObsWrapper, BidScaleWrapper


def _make_vec_normalize(mock_scenario, norm_reward=True, training=True):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    def make():
        env = AuctionEnv(mock_scenario)
        env = FlatAuctionObsWrapper(env)
        env = BidScaleWrapper(env)
        env = Monitor(env)
        return env

    return VecNormalize(
        DummyVecEnv([make]),
        norm_obs=True,
        norm_reward=norm_reward,
        training=training,
    )


class TestSyncNormEvalCallback:

    @pytest.fixture
    def train_vn(self, mock_scenario):
        vn = _make_vec_normalize(mock_scenario, norm_reward=True, training=True)
        yield vn
        vn.close()

    @pytest.fixture
    def eval_vn(self, mock_scenario):
        vn = _make_vec_normalize(mock_scenario, norm_reward=False, training=False)
        yield vn
        vn.close()

    # ── class contract ───────────────────────────────────────────────────────

    def test_is_subclass_of_eval_callback(self, train_vn, eval_vn):
        from benchmark.train_rl import _SyncNormEvalCallback
        from stable_baselines3.common.callbacks import EvalCallback
        cb = _SyncNormEvalCallback(eval_vn, train_env=train_vn, warn=False)
        assert isinstance(cb, EvalCallback)

    # ── obs_rms synchronisation ──────────────────────────────────────────────

    def test_syncs_obs_rms_mean_to_eval_env(self, train_vn, eval_vn):
        from benchmark.train_rl import _SyncNormEvalCallback
        from stable_baselines3.common.callbacks import EvalCallback

        train_vn.obs_rms.mean[:] = 42.0
        cb = _SyncNormEvalCallback(eval_vn, train_env=train_vn, warn=False)

        with patch.object(EvalCallback, "_on_step", return_value=True):
            cb._on_step()

        np.testing.assert_array_equal(eval_vn.obs_rms.mean, 42.0)

    def test_syncs_obs_rms_var_to_eval_env(self, train_vn, eval_vn):
        from benchmark.train_rl import _SyncNormEvalCallback
        from stable_baselines3.common.callbacks import EvalCallback

        train_vn.obs_rms.var[:] = 7.0
        cb = _SyncNormEvalCallback(eval_vn, train_env=train_vn, warn=False)

        with patch.object(EvalCallback, "_on_step", return_value=True):
            cb._on_step()

        np.testing.assert_array_equal(eval_vn.obs_rms.var, 7.0)

    def test_sync_is_a_deepcopy(self, train_vn, eval_vn):
        from benchmark.train_rl import _SyncNormEvalCallback
        from stable_baselines3.common.callbacks import EvalCallback

        train_vn.obs_rms.mean[:] = 1.0
        cb = _SyncNormEvalCallback(eval_vn, train_env=train_vn, warn=False)

        with patch.object(EvalCallback, "_on_step", return_value=True):
            cb._on_step()

        # Mutating train after sync must NOT affect eval
        train_vn.obs_rms.mean[:] = 999.0
        assert not np.all(eval_vn.obs_rms.mean == 999.0)

    # ── graceful behaviour when envs are not VecNormalize ────────────────────

    def test_no_sync_when_train_env_is_plain_vec_env(self, eval_vn, mock_scenario):
        from benchmark.train_rl import _SyncNormEvalCallback
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        plain = DummyVecEnv([lambda: Monitor(BidScaleWrapper(
            FlatAuctionObsWrapper(AuctionEnv(mock_scenario))))])

        eval_vn.obs_rms.mean[:] = 5.0
        initial = eval_vn.obs_rms.mean.copy()

        cb = _SyncNormEvalCallback(eval_vn, train_env=plain, warn=False)

        with patch.object(EvalCallback, "_on_step", return_value=True):
            cb._on_step()  # must not raise

        np.testing.assert_array_equal(eval_vn.obs_rms.mean, initial)
        plain.close()

    def test_no_sync_when_eval_env_is_plain_vec_env(self, train_vn, mock_scenario):
        from benchmark.train_rl import _SyncNormEvalCallback
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        plain = DummyVecEnv([lambda: Monitor(BidScaleWrapper(
            FlatAuctionObsWrapper(AuctionEnv(mock_scenario))))])

        cb = _SyncNormEvalCallback(plain, train_env=train_vn, warn=False)

        with patch.object(EvalCallback, "_on_step", return_value=True):
            cb._on_step()  # must not raise
        plain.close()
