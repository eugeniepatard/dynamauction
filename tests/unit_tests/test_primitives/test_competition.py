import pytest
import numpy as np
from dynamicauc.primitives import LogNormalCompetition, UniformCompetition, ExponentialCompetition


# ── Helpers ────────────────────────────────────────────────────────────────────

N_SAMPLES = 2000
TOLERANCE = 0.1  # relative tolerance for mean/std checks


def draw_samples(dist, n=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    return np.array([dist.sample(rng) for _ in range(n)])


# ── Common properties (all distributions) ─────────────────────────────────────

@pytest.mark.parametrize("dist", [
    LogNormalCompetition(),
    UniformCompetition(),
    ExponentialCompetition(),
])
def test_sample_returns_float(dist):
    rng = np.random.default_rng(0)
    assert isinstance(dist.sample(rng), float)

@pytest.mark.parametrize("dist", [
    LogNormalCompetition(),
    UniformCompetition(),
    ExponentialCompetition(),
])
def test_sample_is_non_negative(dist):
    samples = draw_samples(dist)
    assert np.all(samples >= 0.0)

@pytest.mark.parametrize("dist", [
    LogNormalCompetition(),
    UniformCompetition(),
    ExponentialCompetition(),
])
def test_reproducibility_with_same_seed(dist):
    s1 = draw_samples(dist, seed=7)
    s2 = draw_samples(dist, seed=7)
    np.testing.assert_array_equal(s1, s2)

@pytest.mark.parametrize("dist", [
    LogNormalCompetition(),
    UniformCompetition(),
    ExponentialCompetition(),
])
def test_different_seeds_produce_different_samples(dist):
    s1 = draw_samples(dist, seed=1)
    s2 = draw_samples(dist, seed=2)
    assert not np.array_equal(s1, s2)


# ── LogNormalCompetition ───────────────────────────────────────────────────────

def test_lognormal_default_params():
    dist = LogNormalCompetition()
    assert dist.mu == 0.0
    assert dist.sigma == 0.5

def test_lognormal_expected_mean():
    """E[X] = exp(mu + sigma²/2)"""
    mu, sigma = 0.5, 0.3
    dist = LogNormalCompetition(mu=mu, sigma=sigma)
    expected_mean = np.exp(mu + sigma**2 / 2)
    samples = draw_samples(dist)
    assert abs(np.mean(samples) - expected_mean) / expected_mean < TOLERANCE

def test_lognormal_expected_variance():
    """Var[X] = (exp(sigma²) - 1) * exp(2*mu + sigma²)"""
    mu, sigma = 0.0, 0.5
    dist = LogNormalCompetition(mu=mu, sigma=sigma)
    expected_var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    samples = draw_samples(dist)
    assert abs(np.var(samples) - expected_var) / expected_var < TOLERANCE

def test_lognormal_samples_are_strictly_positive():
    dist = LogNormalCompetition()
    samples = draw_samples(dist)
    assert np.all(samples > 0.0)


# ── UniformCompetition ─────────────────────────────────────────────────────────

def test_uniform_default_params():
    dist = UniformCompetition()
    assert dist.low == 0.0
    assert dist.high == 2.0

def test_uniform_samples_within_bounds():
    low, high = 1.0, 5.0
    dist = UniformCompetition(low=low, high=high)
    samples = draw_samples(dist)
    assert np.all(samples >= low)
    assert np.all(samples <= high)

def test_uniform_expected_mean():
    """E[X] = (low + high) / 2"""
    low, high = 1.0, 5.0
    dist = UniformCompetition(low=low, high=high)
    expected_mean = (low + high) / 2
    samples = draw_samples(dist)
    assert abs(np.mean(samples) - expected_mean) / expected_mean < TOLERANCE

def test_uniform_expected_variance():
    """Var[X] = (high - low)² / 12"""
    low, high = 1.0, 5.0
    dist = UniformCompetition(low=low, high=high)
    expected_var = (high - low)**2 / 12
    samples = draw_samples(dist)
    assert abs(np.var(samples) - expected_var) / expected_var < TOLERANCE


# ── ExponentialCompetition ─────────────────────────────────────────────────────

def test_exponential_default_params():
    dist = ExponentialCompetition()
    assert dist.scale == 1.0

def test_exponential_expected_mean():
    """E[X] = scale"""
    scale = 2.5
    dist = ExponentialCompetition(scale=scale)
    samples = draw_samples(dist)
    assert abs(np.mean(samples) - scale) / scale < TOLERANCE

def test_exponential_expected_variance():
    """Var[X] = scale²"""
    scale = 2.5
    dist = ExponentialCompetition(scale=scale)
    expected_var = scale**2
    samples = draw_samples(dist)
    assert abs(np.var(samples) - expected_var) / expected_var < TOLERANCE

def test_exponential_memoryless_property():
    """
    P(X > s + t | X > s) = P(X > t).
    Verified empirically by checking survival probabilities.
    """
    scale = 1.0
    dist = ExponentialCompetition(scale=scale)
    samples = draw_samples(dist, n=5000)
    s, t = 1.0, 1.0
    p_gt_t = np.mean(samples > t)
    p_gt_s_plus_t_given_gt_s = np.mean(samples[samples > s] > s + t)
    assert abs(p_gt_s_plus_t_given_gt_s - p_gt_t) < 0.05