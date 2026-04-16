# TODO : Rajouter de meilleures tests
import pytest
import numpy as np
from typing import Callable
from dynamicauc import (
    SaturationKernel, 
    RecencyDisplayKernel, 
    CompetitorExtinctionKernel, 
    MixedKernel
)

# --- Mock Helpers ---

@pytest.fixture
def user_vec():
    return np.array([1.0, 0.5])

@pytest.fixture
def x_vec():
    return np.array([0.0, 1.0])

@pytest.fixture
def constant_y():
    # Constant exogenous process (e.g., no competitor click or fixed click time)
    return lambda t: 0.0 

# --- Tests for SaturationKernel ---

class TestSaturationKernel:
    @staticmethod
    def test_context_update():
        kernel = SaturationKernel()
        ctx = kernel.initial_context()
        assert ctx == 0.0
        ctx = kernel.update_context(ctx, np.array([]), 1.0)
        assert ctx == 1.0

    @staticmethod
    def test_lambda_decay(user_vec, x_vec, constant_y):
        kernel = SaturationKernel(base_rate=1.0, delay_rate=0.5, saturation_scale=10.0)
        # At s=0, lambda should be base_rate * exp(-ctx/scale) * delay_rate
        val = kernel.lambda_(user_vec, x_vec, ctx=0.0, y=0.0, t=0.0, s=0.0)
        assert pytest.approx(val) == 0.5
        
        # Test that higher context (saturation) reduces intensity
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

# --- Tests for RecencyDisplayKernel ---

class TestRecencyDisplayKernel:

    @staticmethod
    def test_initial_recency():
        kernel = RecencyDisplayKernel(alpha=1.0)
        # Should be 1.0 when ctx is -inf
        assert kernel._recency_factor(kernel.initial_context(), 10.0) == 1.0


    @staticmethod
    def test_recency_decay():
        kernel = RecencyDisplayKernel(alpha=1.0)
        # If we just showed an ad (t - ctx is small), intensity should be low
        low_recency = kernel._recency_factor(ctx=9.9, t=10.0)
        high_recency = kernel._recency_factor(ctx=1.0, t=10.0)
        assert low_recency < high_recency

    @staticmethod
    def test_expected_reward(user_vec, x_vec, constant_y):
        alpha = 0.5
        kernel = RecencyDisplayKernel(alpha=alpha, delay_rate=1.0)
        t, ctx = 10.0, 8.0
        
        # Integral of exponential delay (1.0) over infinite horizon is 1.0
        expected = 1.0 - np.exp(-alpha * (t - ctx))
        actual = kernel.expected_reward(user_vec, x_vec, ctx, constant_y, t, horizon=100)
        
        assert pytest.approx(actual, rel=1e-3) == expected

# --- Tests for CompetitorExtinctionKernel ---

class TestCompetitorExtinctionKernel:
    @staticmethod
    def test_extinction_logic(user_vec, x_vec):
        kernel = CompetitorExtinctionKernel(base_rate=1.0)
        # Case 1: Competitor clicked at y=5, display is at t=2.
        # At s=4 (absolute time 6), y < t+s is false? 
        # Actually the logic is: if y > t, return 0. 
        # This implies if a competitor click is scheduled in the future, 
        # the current display is "extinguished".
        val_extinguished = kernel.lambda_(user_vec, x_vec, 0.0, y=5.0, t=2.0, s=1.0)
        assert val_extinguished == 0.0
        
        val_active = kernel.lambda_(user_vec, x_vec, 0.0, y=1.0, t=2.0, s=1.0)
        assert val_active > 0.0

    @staticmethod
    def test_expected_reward_extinction(user_vec, x_vec):
        kernel = CompetitorExtinctionKernel(base_rate=1.0, delay_rate=0.5)
        
        # If competitor click is in the future (y > t), reward is 0
        y_func = lambda time: 10.0
        res = kernel.expected_reward(user_vec, x_vec, 0.0, y_func, t=5.0, horizon=20.0)
        assert res == 0.0

# --- Tests for MixedKernel ---

class TestMixedKernel:
    @staticmethod
    def test_kappa_additivity(user_vec, x_vec, constant_y):
        base = SaturationKernel(base_rate=1.0)
        kappa_val = 0.5
        mixed = MixedKernel(base, kappa_base_rate=kappa_val)
        
        # Kappa should be constant
        assert mixed.kappa(user_vec, x_vec, 0.0, 0.0) == kappa_val
        
        # Expected reward should be kappa + base_expected_reward
        base_expected = base.expected_reward(user_vec, x_vec, 0.0, constant_y, 0, 100)
        mixed_expected = mixed.expected_reward(user_vec, x_vec, 0.0, constant_y, 0, 100)
        
        assert pytest.approx(mixed_expected) == kappa_val + base_expected

    @staticmethod
    def test_context_delegation():
        base = RecencyDisplayKernel()
        mixed = MixedKernel(base)
        assert mixed.initial_context() == base.initial_context()
        assert mixed.update_context(0.0, np.array([]), 5.0) == 5.0

# --- General Statistical Property Test ---

@pytest.mark.parametrize("kernel_class", [SaturationKernel, RecencyDisplayKernel, CompetitorExtinctionKernel, MixedKernel])
def test_lambda_upper_bound_validity(kernel_class, user_vec, x_vec, constant_y):
    """
    Property: lambda(s) must always be <= lambda_upper_bound for thinning to work.
    """
    kernel = kernel_class(base_kernel=SaturationKernel()) if kernel_class == MixedKernel else kernel_class()
    ctx = 0.0 if kernel_class == SaturationKernel else 5.0
    t = 10.0
    
    bound = kernel.lambda_upper_bound(user_vec, x_vec, ctx, t, remaining=5.0)
    
    for s in np.linspace(0, 5, 20):
        val = kernel.lambda_(user_vec, x_vec, ctx, t=t, s=s, y=constant_y(t))
        assert val <= bound + 1e-9