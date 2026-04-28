import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dynamicauc.baseline.estimators import SimulationEstimator, MLPValuationHead
from dynamicauc.baseline.policies import ConstantPolicy, FiPlaPolicy


class TestSimulationEstimator:

    def test_default_pi0_is_fipla(self):
        est = SimulationEstimator()
        assert isinstance(est.pi0, FiPlaPolicy)

    def test_custom_pi0_stored(self):
        policy = ConstantPolicy(0.5)
        est = SimulationEstimator(pi0=policy)
        assert est.pi0 is policy

    def test_default_n_rollouts(self):
        assert SimulationEstimator().n_rollouts == 20

    def test_set_env_stores_env(self, base_env):
        est = SimulationEstimator()
        est.set_env(base_env)
        assert est._env is base_env

    def test_estimate_raises_without_env(self):
        est = SimulationEstimator()
        with pytest.raises(RuntimeError, match="set_env"):
            est.estimate({})

    def test_estimate_returns_float(self, base_env):
        obs, _ = base_env.reset(seed=0)
        est = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=1)
        est.set_env(base_env)
        result = est.estimate(obs)
        assert isinstance(result, float)

    def test_estimate_is_deterministic_with_seed(self, base_env):
        obs, _ = base_env.reset(seed=0)
        est1 = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=2, seed=123)
        est2 = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=2, seed=123)
        est1.set_env(base_env)
        est2.set_env(base_env)
        assert est1.estimate(obs) == pytest.approx(est2.estimate(obs))

    def test_estimate_is_finite(self, base_env):
        obs, _ = base_env.reset(seed=0)
        est = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=2)
        est.set_env(base_env)
        assert np.isfinite(est.estimate(obs))

    def test_estimate_does_not_mutate_live_env(self, base_env):
        obs, _ = base_env.reset(seed=0)
        est = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=1)
        est.set_env(base_env)

        step_before = int(obs["step"])
        est.estimate(obs)
        obs2, _, _, _, _ = base_env.step(np.array([0.0], dtype=np.float32))
        assert int(obs2["step"]) == step_before + 1

    def test_v1_minus_v0_with_constant_zero_pi0(self, base_env):
        # pi0 = ConstantPolicy(0.0) → always lose after forced step
        # delta = 0
        obs, _ = base_env.reset(seed=0)
        est = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=1)
        est.set_env(base_env)
        assert est.estimate(obs) == pytest.approx(0.0, abs=1e-5)


class TestMLPValuationHead:
    N, D = 200, 8

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((self.N, self.D)).astype(np.float32)
        y = (2.0 * X.sum(axis=1)).astype(np.float32)
        return X, y

    @pytest.fixture
    def mlp(self):
        return MLPValuationHead(input_dim=self.D, hidden_dims=(16, 16))

    def test_input_dim_stored(self, mlp):
        assert mlp.input_dim == self.D

    def test_net_has_correct_output_shape(self, mlp):
        x = torch.zeros(3, self.D)
        out = mlp._net(x)
        assert out.shape == (3,)

    def test_fit_returns_self(self, mlp, data):
        X, y = data
        result = mlp.fit(X, y, n_epochs=5)
        assert result is mlp

    def test_fit_populates_train_history(self, mlp, data):
        X, y = data
        mlp.fit(X, y, n_epochs=10)
        assert len(mlp.history_["train"]) == 10

    def test_fit_with_val_populates_val_history(self, mlp, data):
        X, y = data
        split = int(0.8 * self.N)
        mlp.fit(X[:split], y[:split], X_val=X[split:], y_val=y[split:], n_epochs=10)
        assert len(mlp.history_["val"]) == 10

    def test_fit_without_val_gives_empty_val_history(self, mlp, data):
        X, y = data
        mlp.fit(X, y, n_epochs=5)
        assert mlp.history_["val"] == []

    def test_train_loss_decreases_over_many_epochs(self, data):
        X, y = data
        mlp = MLPValuationHead(input_dim=self.D, hidden_dims=(32, 32))
        mlp.fit(X, y, n_epochs=200, lr=1e-2)
        assert mlp.history_["train"][-1] < mlp.history_["train"][0]

    def test_fit_net_in_eval_mode_after_training(self, mlp, data):
        X, y = data
        mlp.fit(X, y, n_epochs=3)
        assert not mlp._net.training

    def test_predict_returns_float(self, base_env):
        obs, _ = base_env.reset(seed=0)
        from dynamicauc.baseline.utils import _obs_to_features
        dim = _obs_to_features(obs).shape[0]
        mlp = MLPValuationHead(input_dim=dim)
        result = mlp.predict(obs)
        assert isinstance(result, float)

    def test_predict_is_finite(self, base_env):
        obs, _ = base_env.reset(seed=0)
        from dynamicauc.baseline.utils import _obs_to_features
        dim = _obs_to_features(obs).shape[0]
        mlp = MLPValuationHead(input_dim=dim)
        assert np.isfinite(mlp.predict(obs))

    def test_predict_consistent_in_eval_mode(self, mlp, data):
        X, y = data
        mlp.fit(X, y, n_epochs=5)
        obs_dict = {
            "t":    np.array([0.0], dtype=np.float32),
            "ctx":  np.array([0.0], dtype=np.float32),
            "step": 0,
            "user_features":    np.zeros(3, dtype=np.float32),
            "display_features": np.zeros(3, dtype=np.float32),
            "y_now":            np.array([0.0], dtype=np.float32),
            "cumulative_reward_since_last_step":  np.array([0.0], dtype=np.float32),
            "conversion_times_since_last_step":   np.array([0.0], dtype=np.float32),
            "expected_reward":  np.array([0.5], dtype=np.float32),
        }
        pred1 = mlp.predict(obs_dict)
        pred2 = mlp.predict(obs_dict)
        assert pred1 == pytest.approx(pred2)

    def test_save_load_roundtrip_predictions(self, mlp, data, tmp_path):
        X, y = data
        mlp.fit(X, y, n_epochs=10)

        path = tmp_path / "mlp.pt"
        mlp.save(path)

        loaded = MLPValuationHead.load(path)
        x = torch.tensor(X[:5])
        with torch.no_grad():
            orig   = mlp._net(x).numpy()
            reloaded = loaded._net(x).numpy()
        np.testing.assert_array_almost_equal(orig, reloaded)

    def test_load_restores_input_dim(self, mlp, data, tmp_path):
        X, y = data
        mlp.fit(X, y, n_epochs=3)
        path = tmp_path / "mlp.pt"
        mlp.save(path)
        loaded = MLPValuationHead.load(path)
        assert loaded.input_dim == self.D

    def test_loaded_net_is_in_eval_mode(self, mlp, data, tmp_path):
        X, y = data
        mlp.fit(X, y, n_epochs=3)
        path = tmp_path / "mlp.pt"
        mlp.save(path)
        loaded = MLPValuationHead.load(path)
        assert not loaded._net.training

    def test_save_creates_file(self, mlp, tmp_path):
        path = tmp_path / "model.pt"
        mlp.save(path)
        assert path.exists()
