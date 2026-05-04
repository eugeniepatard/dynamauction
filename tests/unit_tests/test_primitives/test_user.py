"""Tests for dynamicauc/primitives/user.py: user feature generators."""
import numpy as np
import pytest

from dynamicauc.primitives.user import FixedUser, NormalUser, UniformUser


class TestUniformUser:

    def test_generate_shape(self):
        feat = UniformUser(full_dim=5, observable_dim=3).generate(np.random.default_rng(0))
        assert feat.shape == (5,)

    def test_generate_dtype_is_float32(self):
        feat = UniformUser(full_dim=4, observable_dim=4).generate(np.random.default_rng(0))
        assert feat.dtype == np.float32

    def test_generate_in_unit_range(self):
        rng = np.random.default_rng(0)
        u = UniformUser(full_dim=10, observable_dim=10)
        for _ in range(20):
            feat = u.generate(rng)
            assert np.all(feat >= 0.0) and np.all(feat <= 1.0)

    def test_get_observable_features_truncates(self):
        u = UniformUser(full_dim=5, observable_dim=3)
        full = u.generate(np.random.default_rng(0))
        obs = u.get_observable_features(full)
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs, full[:3])

    def test_get_observable_features_full_dim(self):
        u = UniformUser(full_dim=4, observable_dim=4)
        full = u.generate(np.random.default_rng(0))
        assert u.get_observable_features(full).shape == (4,)

    def test_invalid_observable_dim_raises(self):
        with pytest.raises(ValueError):
            UniformUser(full_dim=3, observable_dim=5)

    def test_reproducibility_with_same_seed(self):
        u = UniformUser(full_dim=6, observable_dim=3)
        feat1 = u.generate(np.random.default_rng(42))
        feat2 = u.generate(np.random.default_rng(42))
        np.testing.assert_array_equal(feat1, feat2)

    def test_different_seeds_yield_different_features(self):
        u = UniformUser(full_dim=6, observable_dim=3)
        feat1 = u.generate(np.random.default_rng(0))
        feat2 = u.generate(np.random.default_rng(1))
        assert not np.allclose(feat1, feat2)


class TestNormalUser:

    def test_generate_shape(self):
        feat = NormalUser(full_dim=4, observable_dim=2).generate(np.random.default_rng(0))
        assert feat.shape == (4,)

    def test_generate_dtype_is_float32(self):
        feat = NormalUser(full_dim=4, observable_dim=2).generate(np.random.default_rng(0))
        assert feat.dtype == np.float32

    def test_get_observable_features_truncates(self):
        u = NormalUser(full_dim=6, observable_dim=3)
        full = u.generate(np.random.default_rng(0))
        obs = u.get_observable_features(full)
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs, full[:3])

    def test_invalid_observable_dim_raises(self):
        with pytest.raises(ValueError):
            NormalUser(full_dim=2, observable_dim=5)

    def test_custom_mean_reflected_in_samples(self):
        # With a large dim and tight sigma, the empirical mean should be close to mu
        rng = np.random.default_rng(0)
        u = NormalUser(full_dim=1000, observable_dim=1000, mu=5.0, sigma=0.1)
        feat = u.generate(rng)
        assert abs(feat.mean() - 5.0) < 0.05

    def test_reproducibility_with_same_seed(self):
        u = NormalUser(full_dim=4, observable_dim=2)
        feat1 = u.generate(np.random.default_rng(7))
        feat2 = u.generate(np.random.default_rng(7))
        np.testing.assert_array_equal(feat1, feat2)


class TestFixedUser:

    @pytest.fixture
    def features(self):
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def test_generate_always_returns_stored_features(self, features):
        u = FixedUser(features=features, observable_dim=3)
        for seed in range(5):
            feat = u.generate(np.random.default_rng(seed))
            np.testing.assert_array_equal(feat, features)

    def test_generate_returns_copy(self, features):
        u = FixedUser(features=features, observable_dim=3)
        feat = u.generate(np.random.default_rng(0))
        feat[0] = 999.0
        feat2 = u.generate(np.random.default_rng(0))
        assert feat2[0] == pytest.approx(0.1)

    def test_get_observable_features_truncates(self, features):
        u = FixedUser(features=features, observable_dim=2)
        obs = u.get_observable_features(features)
        np.testing.assert_array_equal(obs, features[:2])

    def test_dtype_cast_to_float32(self):
        features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        u = FixedUser(features=features, observable_dim=2)
        feat = u.generate(np.random.default_rng(0))
        assert feat.dtype == np.float32

    def test_invalid_observable_dim_raises(self, features):
        with pytest.raises(ValueError):
            FixedUser(features=features, observable_dim=10)
