"""
Recency burst presets (survival-recency-second-price-{fixed,exponential}-burst-arrivals): the survival kernel with the
context S(t⁻) = time since the last won auction (TimeSinceLastWinGamma, t + 50 before the first win), a burst of auctions (rate 10 on [0, 10), then
1) and a fixed (T = 50) or exponential (eta = 50) horizon.
"""
import numpy as np
import pytest

import dynamauction as dac
from dynamauction import AuctionEnv
from dynamauction.primitives import TimeSinceLastWinGamma
from dynamauction.primitives.horizon import ExponentialHorizon, FixedHorizon

FIXED = "survival-recency-second-price-fixed-burst-arrivals"
EXPONENTIAL = "survival-recency-second-price-exponential-burst-arrivals"


@pytest.mark.parametrize("preset, horizon_cls", [(FIXED, FixedHorizon), (EXPONENTIAL, ExponentialHorizon)])
def test_preset_components(preset, horizon_cls):
    sc = dac.make(preset)
    assert isinstance(sc.gamma, TimeSinceLastWinGamma)
    assert type(sc.kernel).__name__ == "SurvivalKernel"
    assert type(sc.arrival).__name__ == "PiecewiseConstantPoisson"
    assert (list(sc.arrival.breakpoints), list(sc.arrival.intensities)) == ([10.0], [10.0, 1.0])
    assert isinstance(sc.horizon_generator, horizon_cls)
    assert (sc.competition.mu, sc.competition.sigma) == (-1.0, 1.0)


@pytest.mark.parametrize("preset", [FIXED, EXPONENTIAL])
def test_context_is_the_time_since_the_last_win(preset):
    """When winning every auction, the context of auction k is t_k − t_{k−1} (t_1 + 50 for the first one)."""
    env = AuctionEnv(dac.make(preset))
    env.reset(seed=1)
    times, ctx = [], []
    for _ in range(6):
        _, _, done, _, info = env.step(np.array([1e9], dtype=np.float32))
        times.append(float(info["auction"]["time"]))
        ctx.append(float(info["auction"]["context"]))
    expected = [times[0] + 50.0] + [b - a for a, b in zip(times, times[1:])]
    np.testing.assert_allclose(ctx, expected, rtol=1e-9, atol=1e-9)


def test_fixed_horizon_is_50():
    env = AuctionEnv(dac.make(FIXED))
    for s in range(5):
        env.reset(seed=s)
        assert env.horizon == pytest.approx(50.0)
