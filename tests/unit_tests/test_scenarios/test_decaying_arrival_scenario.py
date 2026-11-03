"""
Main survival scenario with a decaying arrival rate (``survival-second-price-exponential-decaying-arrivals`` and its
``-beta0.1`` variant): the same scenario as ``survival-second-price-exponential`` except the arrival process,
λ(t) = 1 · 0.5^⌊t / 10⌋ (and the recovery rate β = 0.1 for the variant).
"""
import numpy as np
import pytest

import dynamauction as dac
from dynamauction import AuctionEnv
from dynamauction.primitives import StepDecayPoisson

MAIN = "survival-second-price-exponential"
DECAY = "survival-second-price-exponential-decaying-arrivals"
DECAY_BETA01 = "survival-second-price-exponential-decaying-arrivals-beta0.1"


@pytest.mark.parametrize("preset, beta", [(DECAY, 1.0), (DECAY_BETA01, 0.1)])
def test_preset_matches_the_main_scenario_except_the_arrival(preset, beta):
    sc, main = dac.make(preset), dac.make(MAIN)
    assert isinstance(sc.arrival, StepDecayPoisson)
    assert (sc.arrival.rate, sc.arrival.period, sc.arrival.factor) == (1.0, 10.0, 0.5)
    assert type(sc.kernel) is type(main.kernel)
    assert (sc.competition.mu, sc.competition.sigma) == (main.competition.mu, main.competition.sigma) == (-1.0, 1.0)
    assert (sc.gamma.c, sc.gamma.beta) == (0.9, beta)
    assert sc.horizon_generator.eta == main.horizon_generator.eta == 50.0


@pytest.mark.parametrize("preset, beta", [(DECAY, 1.0), (DECAY_BETA01, 0.1)])
def test_preset_equals_the_main_scenario_with_overrides(preset, beta):
    """Same campaigns as dac.make(MAIN, arrival_name="decaypoisson", ...), the overrides the pilots were run with."""
    env_preset = AuctionEnv(dac.make(preset))
    env_overrides = AuctionEnv(dac.make(MAIN, gamma_beta=beta, arrival_name="decaypoisson", arrival_rate=1.0,
                                        arrival_period=10.0, arrival_factor=0.5))
    for seed in range(10):
        env_preset.reset(seed=seed)
        env_overrides.reset(seed=seed)
        np.testing.assert_array_equal(env_preset.auction_times, env_overrides.auction_times)
        assert env_preset.horizon == env_overrides.horizon


def test_auctions_follow_the_decaying_rate():
    """Auctions per full 10-unit period (count / time the campaigns run in it, × 10): 10, 5, 2.5."""
    env = AuctionEnv(dac.make(DECAY))
    counts, exposure = np.zeros(3), np.zeros(3)
    for seed in range(300):
        env.reset(seed=seed)
        t, T = np.asarray(env.auction_times), float(env.horizon)
        for k in range(3):
            counts[k] += ((t >= 10 * k) & (t < 10 * (k + 1))).sum()
            exposure[k] += max(0.0, min(T, 10.0 * (k + 1)) - 10.0 * k)
    np.testing.assert_allclose(10 * counts / exposure, [10.0, 5.0, 2.5], rtol=0.12)
