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
    """S(t⁻) = Π_{i: t_i < t} (1 − c·exp(−beta·(t − t_i))), 1 before the first win: Eq. (carry-over) with ([0, 1], ×, 1)."""

    def test_default_initial_context(self):
        assert MultiplicativeGamma(c=0.5).initiate_context() == pytest.approx(1.0)

    def test_current_context_without_past_win(self):
        assert MultiplicativeGamma(c=0.9).current_context(1.0, t=0.2, t_past_won_auctions=[]) == pytest.approx(1.0)

    def test_one_factor_per_past_win_evaluated_at_the_current_time(self):
        g = MultiplicativeGamma(c=0.9)
        ctx = g.current_context(1.0, t=4.0, t_past_won_auctions=[1.0, 3.0])
        assert ctx == pytest.approx((1.0 - 0.9 * np.exp(-3.0)) * (1.0 - 0.9 * np.exp(-1.0)))

    def test_every_factor_recovers(self):
        # the factor of each win tends to 1: no permanent fatigue, however close the wins were
        g = MultiplicativeGamma(c=0.9)
        assert g.current_context(1.0, t=10.0, t_past_won_auctions=[0.0, 1.0]) == pytest.approx(
            (1.0 - 0.9 * np.exp(-10.0)) * (1.0 - 0.9 * np.exp(-9.0)))
        assert g.current_context(1.0, t=1000.0, t_past_won_auctions=[0.0, 0.5, 1.0]) == pytest.approx(1.0)

    def test_factor_right_after_a_win_is_one_minus_c(self):
        g = MultiplicativeGamma(c=0.9)
        assert g.current_context(1.0, t=2.0, t_past_won_auctions=[2.0]) == pytest.approx(0.1)

    def test_order_of_the_wins_does_not_matter(self):
        g = MultiplicativeGamma(c=0.9)
        assert g.current_context(1.0, t=5.0, t_past_won_auctions=[1.0, 3.0]) == pytest.approx(
            g.current_context(1.0, t=5.0, t_past_won_auctions=[3.0, 1.0]))

    def test_stored_context_is_not_used(self):
        # the context only depends on the win times
        g = MultiplicativeGamma(c=0.9)
        assert g.current_context(2.0, t=4.0, t_past_won_auctions=[1.0, 3.0]) == pytest.approx(
            g.current_context(1.0, t=4.0, t_past_won_auctions=[1.0, 3.0]))
        assert g.current_context(0.3, t=4.0, t_past_won_auctions=[]) == pytest.approx(1.0)

    def test_c_zero_means_no_fatigue(self):
        assert MultiplicativeGamma(c=0.0).current_context(1.0, t=4.0, t_past_won_auctions=[1.0, 3.0]) == pytest.approx(1.0)

    def test_update_keeps_the_stored_context(self):
        # the current context is recomputed from the win times: a win does not change the stored value
        g = MultiplicativeGamma(c=0.9)
        for past in ([], [0.5], [0.5, 2.0]):
            assert g.update_context(1.0, _X, _U, t_display=3.0, t_past_won_auctions=past) == pytest.approx(1.0)

    def test_default_beta_is_one(self):
        g = MultiplicativeGamma(c=0.9)
        assert g.beta == pytest.approx(1.0)
        assert g.current_context(0.5, t=4.0, t_past_won_auctions=[1.5]) == pytest.approx(
            MultiplicativeGamma(c=0.9, beta=1.0).current_context(0.5, t=4.0, t_past_won_auctions=[1.5]))

    def test_beta_scales_every_gap(self):
        g = MultiplicativeGamma(c=0.9, beta=0.5)
        ctx = g.current_context(1.0, t=4.0, t_past_won_auctions=[1.0, 2.0])
        assert ctx == pytest.approx((1.0 - 0.9 * np.exp(-0.5 * 3.0)) * (1.0 - 0.9 * np.exp(-0.5 * 2.0)))

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
