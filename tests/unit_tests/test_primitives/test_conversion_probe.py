"""Tests for dynamauction/primitives/conversion_probe.py: PoissonProbe, BernoulliProbe."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from dynamauction.primitives.conversion_probe import PoissonProbe, BernoulliProbe
from dynamauction.primitives.kernel import SurvivalKernel

_U = np.ones(2, dtype=np.float32)
_X = np.ones(2, dtype=np.float32)
_Y = lambda t: 0.0  # noqa: E731


def _poisson_kernel(lam_max=2.0, lam=1.0):
    """Mock kernel for PoissonProbe (no isinstance check involved)."""
    from dynamauction.primitives.kernel import RewardKernel
    k = MagicMock(spec=RewardKernel)
    k.lambda_upper_bound.return_value = lam_max
    k.lambda_.return_value = lam
    return k


class TestPoissonProbe:

    def test_sample_all_returns_list(self):
        probe = PoissonProbe(_poisson_kernel())
        result = probe.sample_all(_U, _X, 1.0, _Y, t=0.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert isinstance(result, list)

    def test_sample_one_returns_list(self):
        probe = PoissonProbe(_poisson_kernel())
        result = probe.sample_one(_U, _X, 1.0, _Y, t=0.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert isinstance(result, list)

    def test_sample_one_returns_at_most_one(self):
        probe = PoissonProbe(_poisson_kernel(lam_max=10.0, lam=10.0))
        rng = np.random.default_rng(0)
        for _ in range(20):
            result = probe.sample_one(_U, _X, 1.0, _Y, t=0.0, horizon=10.0, rng=rng)
            assert len(result) <= 1

    def test_delays_within_remaining_horizon(self):
        horizon = 5.0
        t = 1.0
        probe = PoissonProbe(_poisson_kernel(lam_max=5.0, lam=5.0))
        rng = np.random.default_rng(0)
        for _ in range(20):
            delays = probe.sample_all(_U, _X, 1.0, _Y, t=t, horizon=horizon, rng=rng)
            for d in delays:
                assert 0.0 <= d < (horizon - t)

    def test_zero_remaining_returns_empty(self):
        probe = PoissonProbe(_poisson_kernel())
        result = probe.sample_all(_U, _X, 1.0, _Y, t=5.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert result == []

    def test_zero_lam_max_returns_empty(self):
        probe = PoissonProbe(_poisson_kernel(lam_max=0.0))
        result = probe.sample_all(_U, _X, 1.0, _Y, t=0.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert result == []

    def test_negative_remaining_returns_empty(self):
        probe = PoissonProbe(_poisson_kernel())
        result = probe.sample_one(_U, _X, 1.0, _Y, t=6.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert result == []

    def test_sample_all_can_return_multiple_events(self):
        # With very high rate, it is very likely to get multiple events over a long horizon
        probe = PoissonProbe(_poisson_kernel(lam_max=100.0, lam=100.0))
        rng = np.random.default_rng(0)
        any_multiple = False
        for _ in range(50):
            result = probe.sample_all(_U, _X, 1.0, _Y, t=0.0, horizon=10.0, rng=rng)
            if len(result) > 1:
                any_multiple = True
                break
        assert any_multiple, "Expected sample_all to occasionally return >1 event"


class TestBernoulliProbe:

    @pytest.fixture
    def zero_prob_kernel(self):
        """SurvivalKernel with conversion probability = 0 (ctx=0 makes prob=0)."""
        return SurvivalKernel(w_u=np.ones(2), w_x=np.ones(2), delay_rate=1.0)

    @pytest.fixture
    def unit_prob_kernel(self):
        """SurvivalKernel with conversion probability = 1 (ctx=1, unit weights, unit features)."""
        return SurvivalKernel(w_u=np.ones(2), w_x=np.ones(2), delay_rate=100.0)

    def test_sample_one_returns_list(self, zero_prob_kernel):
        probe = BernoulliProbe(zero_prob_kernel)
        result = probe.sample_one(_U, _X, 0.0, _Y, t=0.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert isinstance(result, list)

    def test_sample_all_returns_list(self, zero_prob_kernel):
        probe = BernoulliProbe(zero_prob_kernel)
        result = probe.sample_all(_U, _X, 0.0, _Y, t=0.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert isinstance(result, list)

    def test_zero_prob_always_returns_empty(self, zero_prob_kernel):
        # ctx=0.0 → conversion_probability = 0 → no conversions
        probe = BernoulliProbe(zero_prob_kernel)
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert probe.sample_one(_U, _X, 0.0, _Y, t=0.0, horizon=10.0, rng=rng) == []
            assert probe.sample_all(_U, _X, 0.0, _Y, t=0.0, horizon=10.0, rng=rng) == []

    def test_sample_one_at_most_one_event(self, unit_prob_kernel):
        # At most one conversion per display for SurvivalKernel
        probe = BernoulliProbe(unit_prob_kernel)
        rng = np.random.default_rng(0)
        for _ in range(20):
            result = probe.sample_one(_U, _X, 1.0, _Y, t=0.0, horizon=100.0, rng=rng)
            assert len(result) <= 1

    def test_sample_all_at_most_one_event(self, unit_prob_kernel):
        probe = BernoulliProbe(unit_prob_kernel)
        rng = np.random.default_rng(0)
        for _ in range(20):
            result = probe.sample_all(_U, _X, 1.0, _Y, t=0.0, horizon=100.0, rng=rng)
            assert len(result) <= 1

    def test_sample_one_and_sample_all_are_equivalent(self, unit_prob_kernel):
        # BernoulliProbe.sample_one and sample_all call the same internal method
        probe = BernoulliProbe(unit_prob_kernel)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        r1 = probe.sample_one(_U, _X, 1.0, _Y, t=0.0, horizon=5.0, rng=rng1)
        r2 = probe.sample_all(_U, _X, 1.0, _Y, t=0.0, horizon=5.0, rng=rng2)
        assert r1 == r2

    def test_requires_survival_kernel(self):
        from dynamauction.primitives.kernel import RewardKernel
        wrong_kernel = MagicMock(spec=RewardKernel)
        probe = BernoulliProbe(wrong_kernel)
        with pytest.raises(AssertionError):
            probe.sample_one(_U, _X, 1.0, _Y, t=0.0, horizon=5.0,
                             rng=np.random.default_rng(0))

    def test_zero_remaining_returns_empty(self, unit_prob_kernel):
        probe = BernoulliProbe(unit_prob_kernel)
        result = probe.sample_one(_U, _X, 1.0, _Y, t=5.0, horizon=5.0,
                                  rng=np.random.default_rng(0))
        assert result == []
