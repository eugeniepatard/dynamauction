import numpy as np
from dynamauction.primitives.user import UniformUser
from dynamauction.primitives.horizon import FixedHorizon, ExponentialHorizon

def test_uniform_user():
    rng = np.random.default_rng(42)
    user_gen = UniformUser(full_dim=5, observable_dim=3)
    u_full_samples = [user_gen.generate(rng) for _ in range(1000)]
    u_obs_samples = [user_gen.get_observable_features(u_full) for u_full in u_full_samples]
    mean_full = np.mean(u_full_samples, axis=0)
    std_full = np.std(u_full_samples, axis=0)
    assert np.all((0.3 <= mean_full) & (mean_full <= 0.7))
    assert np.all((0.25 <= std_full) & (std_full <= 0.35))
    for u_full, u_obs in zip(u_full_samples, u_obs_samples):
        assert u_full.shape == (5,)
        assert u_obs.shape == (3,)
        assert np.array_equal(u_full[:3], u_obs)

def test_fixed_horizon():
    rng = np.random.default_rng(42)
    horizon_gen = FixedHorizon(value=15.0)
    assert horizon_gen.sample(rng) == 15.0

def test_exponential_horizon():
    rng = np.random.default_rng(42)
    horizon_gen = ExponentialHorizon(eta=10.0)
    sample = horizon_gen.sample(rng)
    assert sample >= 0.0
