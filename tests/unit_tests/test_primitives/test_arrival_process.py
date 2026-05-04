import pytest
import numpy as np
from dynamauction.primitives import PiecewiseConstantPoisson


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_homogeneous(rate: float) -> PiecewiseConstantPoisson:
    """Single-interval process (no breakpoints) with constant rate."""
    return PiecewiseConstantPoisson(breakpoints=[], intensities=[rate])


# ── Basic properties ───────────────────────────────────────────────────────────

def test_times_are_sorted():
    proc = make_homogeneous(rate=5.0)
    times = proc.sample_times(horizon=20.0, rng=np.random.default_rng(0))
    assert list(times) == sorted(times)

def test_times_within_horizon():
    horizon = 10.0
    proc = make_homogeneous(rate=10.0)
    times = proc.sample_times(horizon=horizon, rng=np.random.default_rng(0))
    assert np.all(times >= 0.0)
    assert np.all(times <= horizon)

def test_zero_rate_produces_no_events():
    proc = make_homogeneous(rate=0.0)
    times = proc.sample_times(horizon=100.0, rng=np.random.default_rng(0))
    assert len(times) == 0

def test_zero_horizon_produces_no_events():
    proc = make_homogeneous(rate=10.0)
    times = proc.sample_times(horizon=0.0, rng=np.random.default_rng(0))
    assert len(times) == 0

def test_returns_numpy_array():
    proc = make_homogeneous(rate=1.0)
    times = proc.sample_times(horizon=10.0, rng=np.random.default_rng(0))
    assert isinstance(times, np.ndarray)


# ── Statistical properties ─────────────────────────────────────────────────────

def test_homogeneous_expected_count():
    """Mean event count should be close to rate * horizon."""
    rate, horizon = 4.0, 25.0
    proc = make_homogeneous(rate=rate)
    rng = np.random.default_rng(42)
    counts = [len(proc.sample_times(horizon, rng)) for _ in range(500)]
    assert abs(np.mean(counts) - rate * horizon) < 1.5

def test_piecewise_expected_count_per_interval():
    """
    With breakpoints, mean count in each interval should match
    intensity * interval_length independently.
    """
    breakpoints = [10.0]
    intensities = [2.0, 5.0]   # [0,10) → rate 2, [10,20) → rate 5
    horizon = 20.0
    proc = PiecewiseConstantPoisson(breakpoints=breakpoints, intensities=intensities)
    rng = np.random.default_rng(42)

    counts_first, counts_second = [], []
    for _ in range(500):
        times = proc.sample_times(horizon, rng)
        counts_first.append(np.sum(times < 10.0))
        counts_second.append(np.sum(times >= 10.0))

    assert abs(np.mean(counts_first) - 2.0 * 10.0) < 1.5   # expected 20
    assert abs(np.mean(counts_second) - 5.0 * 10.0) < 2.0  # expected 50

def test_piecewise_events_stay_in_correct_interval():
    """Events sampled in each interval must not bleed into other intervals."""
    breakpoints = [5.0, 15.0]
    intensities = [1.0, 1.0, 1.0]
    horizon = 20.0
    proc = PiecewiseConstantPoisson(breakpoints=breakpoints, intensities=intensities)
    rng = np.random.default_rng(0)

    for _ in range(50):
        times = proc.sample_times(horizon, rng)
        assert np.all(times >= 0.0)
        assert np.all(times <= horizon)


# ── Breakpoint edge cases ──────────────────────────────────────────────────────

def test_breakpoint_beyond_horizon_is_ignored():
    """A breakpoint beyond horizon should not cause errors or out-of-bound times."""
    proc = PiecewiseConstantPoisson(breakpoints=[50.0], intensities=[1.0, 1.0])
    horizon = 10.0
    times = proc.sample_times(horizon=horizon, rng=np.random.default_rng(0))
    assert np.all(times <= horizon)

def test_single_breakpoint_at_horizon_boundary():
    """Breakpoint exactly at horizon — second interval has zero length."""
    proc = PiecewiseConstantPoisson(breakpoints=[10.0], intensities=[2.0, 5.0])
    times = proc.sample_times(horizon=10.0, rng=np.random.default_rng(0))
    assert np.all(times <= 10.0)

def test_multiple_breakpoints():
    """Smoke test with many intervals — should not raise."""
    breakpoints = [10.0, 20.0, 30.0, 40.0]
    intensities = [1.0, 2.0, 0.5, 3.0, 1.5]
    proc = PiecewiseConstantPoisson(breakpoints=breakpoints, intensities=intensities)
    times = proc.sample_times(horizon=50.0, rng=np.random.default_rng(7))
    assert isinstance(times, np.ndarray)
    assert np.all(times >= 0.0)
    assert np.all(times <= 50.0)

def test_reproducibility_with_same_seed():
    """Same seed should produce identical results."""
    proc = make_homogeneous(rate=3.0)
    t1 = proc.sample_times(10.0, np.random.default_rng(99))
    t2 = proc.sample_times(10.0, np.random.default_rng(99))
    np.testing.assert_array_equal(t1, t2)