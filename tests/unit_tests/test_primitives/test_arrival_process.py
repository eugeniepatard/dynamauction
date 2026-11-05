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

# ── Step-decay Poisson: λ(t) = rate · factor^⌊t / period⌋ ─────────────────────

def test_step_decay_intensity_schedule():
    from dynamauction.primitives import StepDecayPoisson
    proc = StepDecayPoisson(rate=1.0, period=10.0, factor=0.5)
    assert [proc.intensity(t) for t in (0.0, 9.99, 10.0, 25.0, 35.0)] == [1.0, 1.0, 0.5, 0.25, 0.125]

def test_step_decay_expected_count_per_period():
    """Mean count in period k should be rate · period · factor^k."""
    from dynamauction.primitives import StepDecayPoisson
    proc = StepDecayPoisson(rate=2.0, period=10.0, factor=0.5)
    rng = np.random.default_rng(42)
    counts = np.array([np.histogram(proc.sample_times(40.0, rng), bins=[0, 10, 20, 30, 40])[0] for _ in range(1000)])
    np.testing.assert_allclose(counts.mean(axis=0), [20.0, 10.0, 5.0, 2.5], rtol=0.05)

def test_step_decay_times_sorted_within_horizon():
    from dynamauction.primitives import StepDecayPoisson
    proc = StepDecayPoisson(rate=3.0, period=4.0, factor=0.7)
    times = proc.sample_times(horizon=25.0, rng=np.random.default_rng(0))
    assert list(times) == sorted(times)
    assert np.all((times >= 0.0) & (times <= 25.0))

def test_step_decay_factor_one_is_homogeneous():
    from dynamauction.primitives import StepDecayPoisson
    proc = StepDecayPoisson(rate=4.0, period=3.0, factor=1.0)
    rng = np.random.default_rng(1)
    counts = [len(proc.sample_times(25.0, rng)) for _ in range(500)]
    assert abs(np.mean(counts) - 100.0) < 2.0

def test_step_decay_rejects_bad_parameters():
    import pytest
    from dynamauction.primitives import StepDecayPoisson
    with pytest.raises(ValueError):
        StepDecayPoisson(period=0.0)
    with pytest.raises(ValueError):
        StepDecayPoisson(factor=-0.5)


# ── sample_times_from: the planning rollouts use the rate at the absolute time ─

def test_step_decay_sample_times_from_uses_absolute_time():
    """A window starting at t = 30 has the rate of period 3 (0.125), not the initial rate."""
    from dynamauction.primitives import StepDecayPoisson
    proc = StepDecayPoisson(rate=1.0, period=10.0, factor=0.5)
    rng = np.random.default_rng(3)
    draws = [proc.sample_times_from(30.0, 10.0, rng) for _ in range(2000)]
    assert all(np.all((d >= 30.0) & (d <= 40.0)) for d in draws)
    assert abs(np.mean([len(d) for d in draws]) - 1.25) < 0.1

def test_constant_poisson_sample_times_from_is_the_shifted_draw():
    """Homogeneous process: exactly the previous rollout computation, sample_times(duration) + t_start."""
    from dynamauction.primitives import ConstantPoisson
    proc = ConstantPoisson(rate=2.0)
    np.testing.assert_array_equal(proc.sample_times_from(7.5, 12.0, np.random.default_rng(5)),
                                  proc.sample_times(12.0, np.random.default_rng(5)) + 7.5)

def test_piecewise_sample_times_from_uses_absolute_breakpoints():
    """Rate 5 on [0, 10) and 0 after: a window starting at 12 has no auction, one starting at 0 none after 10."""
    proc = PiecewiseConstantPoisson(breakpoints=[10.0], intensities=[5.0, 0.0])
    rng = np.random.default_rng(0)
    assert len(proc.sample_times_from(12.0, 20.0, rng)) == 0
    times = proc.sample_times_from(0.0, 20.0, rng)
    assert len(times) > 0 and np.all(times < 10.0)

def test_rollout_resampling_keeps_the_absolute_schedule():
    """_resample_remaining_horizon draws the future auctions with λ at the absolute time: with factor 0 there is
    no auction after the first period, whatever the remaining horizon drawn by the rollout."""
    import copy
    import dynamauction as dac
    from dynamauction import AuctionEnv
    from dynamauction.baseline.utils import _resample_remaining_horizon

    scenario = dac.make("survival-second-price-exponential", horizon_eta=50.0, arrival_name="decaypoisson",
                        arrival_rate=1.0, arrival_period=10.0, arrival_factor=0.0)
    env = AuctionEnv(scenario)
    rng = np.random.default_rng(0)
    for seed in range(20):
        env.reset(seed=seed)
        rollout = copy.deepcopy(env)
        _resample_remaining_horizon(rollout, rng)
        future = rollout.auction_times[rollout.state.step_idx + 1:]
        assert np.all(future < 10.0)
