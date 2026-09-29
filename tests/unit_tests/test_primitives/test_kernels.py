import pytest
import numpy as np
from typing import Callable
from dynamauction import (
    SaturationKernel, 
    RecencyDisplayKernel, 
    CompetitorExtinctionKernel, 
    MixedKernel,
    SurvivalKernel
)

@pytest.fixture
def user_vec():
    return np.array([1.0, 0.5])

@pytest.fixture
def x_vec():
    return np.array([0.0, 1.0])

@pytest.fixture
def constant_y():
    return lambda t: 0.0 

class TestSaturationKernel:
    @staticmethod
    def test_lambda_decay(user_vec, x_vec, constant_y):
        kernel = SaturationKernel(base_rate=1.0, delay_rate=0.5, saturation_scale=10.0)
        val = kernel.lambda_(user_vec, x_vec, ctx=0.0, y=0.0, t=0.0, s=0.0)
        assert pytest.approx(val) == 0.5
        
        val_saturated = kernel.lambda_(user_vec, x_vec, ctx=10.0, y=0.0, t=0.0, s=0.0)
        assert val_saturated < val
        assert pytest.approx(val_saturated, abs=0.001) == 0.183

    @staticmethod
    def test_statistical_expectation(user_vec, x_vec, constant_y):
        """
        Integral of delay_rate * exp(-delay_rate * s) from 0 to inf is 1.0.
        So expected reward should be base_rate * exp(-ctx/scale).
        """
        base = 2.0
        scale = 10.0
        ctx = 5.0
        kernel = SaturationKernel(base_rate=base, saturation_scale=scale, delay_rate=0.5)
        
        expected = base * np.exp(-ctx / scale)
        actual = kernel.expected_reward(user_vec, x_vec, ctx, constant_y, t=0, horizon=100)
        
        assert pytest.approx(actual, rel=1e-3) == expected

class TestRecencyDisplayKernel:

    @staticmethod
    def test_initial_recency():
        kernel = RecencyDisplayKernel(recency_coef=1.0)
        assert kernel._conversion_prob(float("-inf"), 10.0) == 1.0


    @staticmethod
    def test_recency_decay():
        kernel = RecencyDisplayKernel(recency_coef=1.0)
        low_recency = kernel._conversion_prob(ctx=9.9, t=10.0)
        high_recency = kernel._conversion_prob(ctx=1.0, t=10.0)
        assert low_recency < high_recency

    @staticmethod
    def test_expected_reward(user_vec, x_vec, constant_y):
        recency_coef = 0.5
        kernel = RecencyDisplayKernel(recency_coef=recency_coef, delay_rate=1.0)
        t, ctx = 10.0, 8.0

        expected = 1.0 - np.exp(-recency_coef * (t - ctx))
        actual = kernel.expected_reward(user_vec, x_vec, ctx, constant_y, t, horizon=100)
        
        assert pytest.approx(actual, rel=1e-3) == expected

class TestCompetitorExtinctionKernel:
    @staticmethod
    def test_extinction_logic(user_vec, x_vec):
        kernel = CompetitorExtinctionKernel(base_rate=1.0)
        val_extinguished = kernel.lambda_(user_vec, x_vec, 0.0, y=5.0, t=2.0, s=1.0)
        assert val_extinguished == 0.0
        
        val_active = kernel.lambda_(user_vec, x_vec, 0.0, y=1.0, t=2.0, s=1.0)
        assert val_active > 0.0

    @staticmethod
    def test_expected_reward_extinction(user_vec, x_vec):
        kernel = CompetitorExtinctionKernel(base_rate=1.0, delay_rate=0.5)
        
        y_func = lambda time: 10.0
        res = kernel.expected_reward(user_vec, x_vec, 0.0, y_func, t=5.0, horizon=20.0)
        assert res == 0.0

class TestMixedKernel:
    @staticmethod
    def test_kappa_additivity(user_vec, x_vec, constant_y):
        base = SaturationKernel(base_rate=1.0)
        kappa_val = 0.5
        mixed = MixedKernel(base, kappa_base_rate=kappa_val)
        
        assert mixed.kappa(user_vec, x_vec, 0.0, 0.0) == kappa_val
        
        base_expected = base.expected_reward(user_vec, x_vec, 0.0, constant_y, 0, 100)
        mixed_expected = mixed.expected_reward(user_vec, x_vec, 0.0, constant_y, 0, 100)
        
        assert pytest.approx(mixed_expected) == kappa_val + base_expected

class TestSurvivalKernel:
    @staticmethod
    def test_conversion_probability(user_vec, x_vec):
        kernel = SurvivalKernel(w_u=np.array([1.0, 1.0]), w_x=np.array([1.0, 1.0]), delay_rate=1.0)
        prob = kernel.conversion_probability(user_vec, x_vec, ctx=0.5)
        assert pytest.approx(prob) == 0.75

        prob_clip = kernel.conversion_probability(user_vec, x_vec, ctx=2.0)
        assert prob_clip == 1.0

    @staticmethod
    def test_expected_reward(user_vec, x_vec, constant_y):
        kernel = SurvivalKernel(w_u=np.array([1.0, 1.0]), w_x=np.array([1.0, 1.0]), delay_rate=1.0)

        expected = 0.75 * (1.0 - np.exp(-100.0))
        actual = kernel.expected_reward(user_vec, x_vec, ctx=0.5, y_func=constant_y, t=0.0, horizon=100.0)
        assert pytest.approx(actual, rel=1e-3) == expected

    # ── conversion_probability tests ───────────────────────────────────────────

    @staticmethod
    def test_clip_output_in_unit_interval(user_vec, x_vec):
        kernel = SurvivalKernel(w_u=np.array([1.0, 1.0]), w_x=np.array([1.0, 1.0]))
        for ctx in [-5.0, 0.0, 0.5, 1.0, 10.0]:
            p = kernel.conversion_probability(user_vec, x_vec, ctx=ctx)
            assert 0.0 <= p <= 1.0, f"clip output {p} not in [0,1] for ctx={ctx}"

    @staticmethod
    def test_clip_saturates_at_boundaries(user_vec, x_vec):
        kernel = SurvivalKernel(w_u=np.array([1.0, 1.0]), w_x=np.array([1.0, 1.0]))
        assert kernel.conversion_probability(user_vec, x_vec, ctx=100.0) == 1.0
        assert kernel.conversion_probability(user_vec, x_vec, ctx=-1.0) == 0.0

# --- General Statistical Property Test ---

@pytest.mark.parametrize("kernel_class", [SaturationKernel, RecencyDisplayKernel, CompetitorExtinctionKernel, MixedKernel, SurvivalKernel])
def test_lambda_upper_bound_validity(kernel_class, user_vec, x_vec, constant_y):
    """
    Property: lambda(s) must always be <= lambda_upper_bound for thinning to work.
    """
    if kernel_class == MixedKernel:
        kernel = MixedKernel(base_kernel=SaturationKernel())
    elif kernel_class == SurvivalKernel:
        kernel = SurvivalKernel(w_u=np.array([1.0, 1.0]), w_x=np.array([1.0, 1.0]))
    else:
        kernel = kernel_class()

    ctx = 0.5 if kernel_class == SurvivalKernel else (0.0 if kernel_class == SaturationKernel else 5.0)
    t = 10.0
    
    bound = kernel.lambda_upper_bound(user_vec, x_vec, ctx, t, remaining=5.0)
    
    for s in np.linspace(0, 5, 20):
        val = kernel.lambda_(user_vec, x_vec, ctx, t=t, s=s, y=constant_y(t))
        assert val <= bound + 1e-9