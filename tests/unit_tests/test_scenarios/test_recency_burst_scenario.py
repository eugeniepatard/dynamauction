"""
Time-since-last-win context on the burst scenario: the survival-recovering-second-price-exponential-burst-arrivals preset
with its context replaced by S(t⁻) = time since the last won auction (TimeSinceLastWinGamma, t + 50 before the first win),
with its exponential horizon (eta = 50) or a fixed one (T = 50). Built through make() overrides.
"""
import numpy as np
import pytest

import dynamauction as dac
from dynamauction import AuctionEnv
from dynamauction.primitives import TimeSinceLastWinGamma
from dynamauction.primitives.horizon import ExponentialHorizon, FixedHorizon

BURST = "survival-recovering-second-price-exponential-burst-arrivals"
RECENCY = {"gamma_name": "time-since-last-win", "gamma_params": {"initial_value": -50.0}}
FIXED_T = {"horizon_name": "fixed", "horizon_params": {"value": 50.0}}


def make_recency_burst(fixed: bool):
    return dac.make(BURST, **RECENCY, **(FIXED_T if fixed else {}))


@pytest.mark.parametrize("fixed, horizon_cls", [(True, FixedHorizon), (False, ExponentialHorizon)])
def test_scenario_components(fixed, horizon_cls):
    sc = make_recency_burst(fixed)
    assert isinstance(sc.gamma, TimeSinceLastWinGamma)
    assert type(sc.kernel).__name__ == "SurvivalKernel"
    assert type(sc.arrival).__name__ == "PiecewiseConstantPoisson"
    assert (list(sc.arrival.breakpoints), list(sc.arrival.intensities)) == ([10.0], [10.0, 1.0])
    assert isinstance(sc.horizon_generator, horizon_cls)
    assert (sc.competition.mu, sc.competition.sigma) == (-1.0, 1.0)


@pytest.mark.parametrize("fixed", [True, False])
def test_context_is_the_time_since_the_last_win(fixed):
    """When winning every auction, the context of auction k is t_k − t_{k−1} (t_1 + 50 for the first one)."""
    env = AuctionEnv(make_recency_burst(fixed))
    env.reset(seed=1)
    times, ctx = [], []
    for _ in range(6):
        _, _, done, _, info = env.step(np.array([1e9], dtype=np.float32))
        times.append(float(info["auction"]["time"]))
        ctx.append(float(info["auction"]["context"]))
    expected = [times[0] + 50.0] + [b - a for a, b in zip(times, times[1:])]
    np.testing.assert_allclose(ctx, expected, rtol=1e-9, atol=1e-9)


def test_fixed_horizon_is_50():
    env = AuctionEnv(make_recency_burst(fixed=True))
    for s in range(5):
        env.reset(seed=s)
        assert env.horizon == pytest.approx(50.0)
