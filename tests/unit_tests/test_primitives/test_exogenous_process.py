import pytest
import numpy as np
from unittest.mock import MagicMock
from dynamauction.primitives import TrivialProcess, CompetitorClickProcess


# ── TrivialProcess ─────────────────────────────────────────────────────────────

def test_trivial_process_y_is_always_zero():
    proc = TrivialProcess()
    proc.generate(horizon=100.0, rng=np.random.default_rng(42))
    for t in [0.0, 1.0, 50.0, 99.9, 100.0]:
        assert proc.y(t) == 0.0

def test_trivial_process_generate_is_noop():
    """generate() should not raise and state should remain trivial."""
    proc = TrivialProcess()
    proc.generate(horizon=0.0, rng=np.random.default_rng(0))
    proc.generate(horizon=1e9, rng=np.random.default_rng(0))
    assert proc.y(42.0) == 0.0


# ── CompetitorClickProcess ─────────────────────────────────────────────────────

def test_competitor_no_clicks_before_first_event():
    """Before any click, y(t) should be -inf."""
    proc = CompetitorClickProcess(rate=1.0)
    # Force zero clicks by using a rng that draws 0 from Poisson
    rng = MagicMock()
    rng.poisson.return_value = 0
    proc.generate(horizon=10.0, rng=rng)
    assert proc.y(0.0) == float("-inf")
    assert proc.y(5.0) == float("-inf")
    assert proc.y(10.0) == float("-inf")

def test_competitor_y_returns_most_recent_click_before_t():
    """y(t) should return the last click time strictly before t."""
    proc = CompetitorClickProcess(rate=1.0)
    rng = MagicMock()
    rng.poisson.return_value = 3
    rng.uniform.return_value = np.array([1.0, 3.0, 7.0])  # click times
    proc.generate(horizon=10.0, rng=rng)

    assert proc.y(0.5) == float("-inf")  # before first click
    assert proc.y(1.0) == float("-inf")  # strictly before: 1.0 is not < 1.0
    assert proc.y(2.0) == 1.0            # last click before 2.0 is 1.0
    assert proc.y(3.0) == 1.0            # 3.0 is not strictly before 3.0
    assert proc.y(5.0) == 3.0            # last click before 5.0 is 3.0
    assert proc.y(7.5) == 7.0            # last click before 7.5 is 7.0
    assert proc.y(10.0) == 7.0           # last click before 10.0 is 7.0

def test_competitor_click_times_are_sorted():
    """Internal click times should always be sorted after generate()."""
    proc = CompetitorClickProcess(rate=5.0)
    proc.generate(horizon=20.0, rng=np.random.default_rng(0))
    assert list(proc._click_times) == sorted(proc._click_times)

def test_competitor_click_times_within_horizon():
    """All click times should be within [0, horizon]."""
    horizon = 50.0
    proc = CompetitorClickProcess(rate=2.0)
    proc.generate(horizon=horizon, rng=np.random.default_rng(1))
    assert np.all(proc._click_times >= 0.0)
    assert np.all(proc._click_times <= horizon)

def test_competitor_generate_resets_state():
    """Calling generate() twice should reset click times, not accumulate."""
    proc = CompetitorClickProcess(rate=10.0)
    proc.generate(horizon=10.0, rng=np.random.default_rng(0))
    n_first = len(proc._click_times)
    proc.generate(horizon=10.0, rng=np.random.default_rng(0))
    n_second = len(proc._click_times)
    assert n_first == n_second
    
def test_competitor_expected_click_count():
    """Over many runs, mean click count should be close to rate * horizon."""
    rate, horizon = 3.0, 20.0
    rng = np.random.default_rng(42)
    counts = []
    for _ in range(500):
        proc = CompetitorClickProcess(rate=rate)
        proc.generate(horizon=horizon, rng=rng)
        counts.append(len(proc._click_times))
    assert abs(np.mean(counts) - rate * horizon) < 1.0

def test_competitor_zero_rate_produces_no_clicks():
    proc = CompetitorClickProcess(rate=0.0)
    proc.generate(horizon=100.0, rng=np.random.default_rng(42))
    assert len(proc._click_times) == 0
    assert proc.y(50.0) == float("-inf")