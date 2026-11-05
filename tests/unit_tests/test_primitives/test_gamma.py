"""Tests for dynamauction/primitives/gamma.py: carry-over context functions."""
import numpy as np
import pytest

from dynamauction.primitives.gamma import (
    AdditiveGamma,
    ConstantGamma,
    LastWonTimeGamma,
    MultiplicativeGamma,
    TimeSinceLastWinGamma,
)

_X = np.array([0.1, 0.2, 0.3], dtype=np.float32)
_U = np.array([1.0, 0.5], dtype=np.float32)


class TestAdditiveGamma:

    def test_default_initial_context(self):
        assert AdditiveGamma().initiate_context() == pytest.approx(0.0)

    def test_custom_initial_context(self):
        assert AdditiveGamma(initial_value=3.0).initiate_context() == pytest.approx(3.0)

    def test_update_increments_by_one(self):
        g = AdditiveGamma()
        ctx = g.update_context(0.0, _X, _U, t_display=1.0, t_past_won_auctions=[0.5])
        assert ctx == pytest.approx(1.0)

    def test_update_is_additive_over_steps(self):
        g = AdditiveGamma()
        ctx = g.initiate_context()
        for _ in range(5):
            ctx = g.update_context(ctx, _X, _U, t_display=1.0, t_past_won_auctions=[])
        assert ctx == pytest.approx(5.0)

    def test_update_ignores_features_and_time(self):
        g = AdditiveGamma()
        ctx_a = g.update_context(0.0, _X, _U, t_display=1.0, t_past_won_auctions=[])
        ctx_b = g.update_context(0.0, np.zeros(5), np.zeros(2), t_display=99.0,
                                 t_past_won_auctions=[1.0, 2.0])
        assert ctx_a == pytest.approx(ctx_b)

    def test_update_with_empty_past_auctions(self):
        g = AdditiveGamma()
        assert g.update_context(0.0, _X, _U, t_display=1.0, t_past_won_auctions=[]) == pytest.approx(1.0)


class TestMultiplicativeGamma:

    def test_default_initial_context(self):
        assert MultiplicativeGamma(c=0.5).initiate_context() == pytest.approx(1.0)

    def test_custom_initial_context(self):
        assert MultiplicativeGamma(c=0.5, initial_value=2.0).initiate_context() == pytest.approx(2.0)

    def test_update_formula(self):
        # ctx=1.0, c=0.5, t_display=1.0, last_won=0.0 → factor = 1 - 0.5*exp(-1)
        g = MultiplicativeGamma(c=0.5)
        ctx = g.update_context(1.0, _X, _U, t_display=1.0, t_past_won_auctions=[0.0])
        assert ctx == pytest.approx(1.0 * (1.0 - 0.5 * np.exp(-1.0)))

    def test_update_uses_most_recent_won_auction(self):
        # Multiple past times: only the most recent one should be used
        g = MultiplicativeGamma(c=0.5)
        ctx1 = g.update_context(1.0, _X, _U, t_display=5.0, t_past_won_auctions=[3.0])
        ctx2 = g.update_context(1.0, _X, _U, t_display=5.0, t_past_won_auctions=[1.0, 2.0, 3.0])
        assert ctx1 == pytest.approx(ctx2)

    def test_update_c_zero_leaves_context_unchanged(self):
        g = MultiplicativeGamma(c=0.0)
        ctx = g.update_context(2.0, _X, _U, t_display=1.0, t_past_won_auctions=[0.5])
        assert ctx == pytest.approx(2.0)

    def test_update_large_gap_approaches_no_decay(self):
        # c=1.0, gap→∞ → exp(−gap)≈0 → factor≈1 → ctx essentially unchanged
        g = MultiplicativeGamma(c=1.0)
        ctx = g.update_context(3.0, _X, _U, t_display=1000.0, t_past_won_auctions=[0.0])
        assert ctx == pytest.approx(3.0, abs=1e-3)

    def test_first_win_leaves_context_unchanged(self):
        # No previous win → no factor, whatever the display time (S(0⁻) = initial value)
        g = MultiplicativeGamma(c=0.9)
        for t in [0.1, 0.5, 1.0, 7.0]:
            assert g.update_context(1.0, _X, _U, t_display=t, t_past_won_auctions=[]) == pytest.approx(1.0)

    def test_current_context_without_past_win(self):
        assert MultiplicativeGamma(c=0.9).current_context(1.0, t=0.2, t_past_won_auctions=[]) == pytest.approx(1.0)

    def test_current_context_uses_gap_to_last_win(self):
        g = MultiplicativeGamma(c=0.9)
        ctx = g.current_context(0.5, t=4.0, t_past_won_auctions=[1.0, 3.0])
        assert ctx == pytest.approx(0.5 * (1.0 - 0.9 * np.exp(-1.0)))

    def test_update_stores_current_context(self):
        g = MultiplicativeGamma(c=0.9)
        stored = g.update_context(0.5, _X, _U, t_display=4.0, t_past_won_auctions=[3.0])
        assert stored == pytest.approx(g.current_context(0.5, t=4.0, t_past_won_auctions=[3.0]))

    def test_default_beta_is_one(self):
        g = MultiplicativeGamma(c=0.9)
        assert g.beta == pytest.approx(1.0)
        assert g.current_context(0.5, t=4.0, t_past_won_auctions=[1.5]) == pytest.approx(
            MultiplicativeGamma(c=0.9, beta=1.0).current_context(0.5, t=4.0, t_past_won_auctions=[1.5]))

    def test_beta_scales_gap(self):
        g = MultiplicativeGamma(c=0.9, beta=0.5)
        ctx = g.current_context(0.5, t=4.0, t_past_won_auctions=[1.0, 2.0])
        assert ctx == pytest.approx(0.5 * (1.0 - 0.9 * np.exp(-0.5 * 2.0)))

    def test_beta_ignored_before_first_win(self):
        g = MultiplicativeGamma(c=0.9, beta=3.0)
        assert g.current_context(1.0, t=0.2, t_past_won_auctions=[]) == pytest.approx(1.0)


class TestCurrentContextDefault:

    @pytest.mark.parametrize("gamma", [AdditiveGamma(initial_value=2.0), LastWonTimeGamma(), ConstantGamma(value=7.5)])
    def test_identity(self, gamma):
        assert gamma.current_context(3.25, t=10.0, t_past_won_auctions=[1.0, 9.9]) == pytest.approx(3.25)


class TestLastWonTimeGamma:

    def test_default_initial_context_is_large_negative(self):
        assert LastWonTimeGamma().initiate_context() == pytest.approx(-1e6)

    def test_custom_initial_context(self):
        assert LastWonTimeGamma(initial_value=0.0).initiate_context() == pytest.approx(0.0)

    def test_update_returns_t_display(self):
        g = LastWonTimeGamma()
        ctx = g.update_context(float("-inf"), _X, _U, t_display=3.7, t_past_won_auctions=[])
        assert ctx == pytest.approx(3.7)

    def test_update_ignores_previous_context(self):
        g = LastWonTimeGamma()
        for prev in [0.0, 99.0, -1e6]:
            assert g.update_context(prev, _X, _U, t_display=5.0, t_past_won_auctions=[]) == pytest.approx(5.0)

    def test_update_ignores_past_auctions_list(self):
        g = LastWonTimeGamma()
        ctx1 = g.update_context(0.0, _X, _U, t_display=4.0, t_past_won_auctions=[])
        ctx2 = g.update_context(0.0, _X, _U, t_display=4.0, t_past_won_auctions=[1.0, 2.0, 3.0])
        assert ctx1 == pytest.approx(ctx2)


class TestTimeSinceLastWinGamma:
    """S(t⁻) = t − t_last (the min over the past wins of t − t_i), t − initial_value before the first win."""

    def test_before_the_first_win(self):
        g = TimeSinceLastWinGamma()
        assert g.initiate_context() == pytest.approx(-1000.0)
        assert g.current_context(g.initiate_context(), 3.0, []) == pytest.approx(1003.0)

    def test_time_since_the_last_win(self):
        g = TimeSinceLastWinGamma()
        ctx = g.update_context(g.initiate_context(), _X, _U, 2.0, [])
        ctx = g.update_context(ctx, _X, _U, 5.0, [2.0])
        assert ctx == pytest.approx(5.0)   # the stored context is the last win time
        assert g.current_context(ctx, 7.5, [2.0, 5.0]) == pytest.approx(2.5)
        assert g.current_context(ctx, 7.5, [2.0, 5.0]) == pytest.approx(min(7.5 - t for t in (2.0, 5.0)))

    def test_custom_initial_value(self):
        g = TimeSinceLastWinGamma(initial_value=-50.0)
        assert g.current_context(g.initiate_context(), 1.0, []) == pytest.approx(51.0)


class TestConstantGamma:

    def test_default_value_is_zero(self):
        g = ConstantGamma()
        assert g.initiate_context() == pytest.approx(0.0)
        assert g.update_context(99.0, _X, _U, t_display=5.0, t_past_won_auctions=[]) == pytest.approx(0.0)

    def test_custom_value(self):
        g = ConstantGamma(value=7.5)
        assert g.initiate_context() == pytest.approx(7.5)
        assert g.update_context(0.0, _X, _U, t_display=1.0, t_past_won_auctions=[]) == pytest.approx(7.5)

    def test_update_ignores_all_inputs(self):
        g = ConstantGamma(value=3.0)
        for ctx in [-1.0, 0.0, 100.0]:
            result = g.update_context(ctx, _X, _U, t_display=99.9, t_past_won_auctions=[1.0, 2.0])
            assert result == pytest.approx(3.0)

    def test_repeated_updates_stay_constant(self):
        g = ConstantGamma(value=2.5)
        ctx = g.initiate_context()
        for _ in range(10):
            ctx = g.update_context(ctx, _X, _U, t_display=1.0, t_past_won_auctions=[])
        assert ctx == pytest.approx(2.5)
