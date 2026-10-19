"""
Scenario 2 with immediate rewards (``survival-second-price-exponential-immediate``): same fatigue dynamics and
competition as ``survival-second-price-exponential``, but the conversion happens at bid time (no delay), so the
oracle value of a display is the conversion probability f itself.
"""
import numpy as np
import pytest

import dynamauction as dac
from dynamauction import AuctionEnv

IMM = "survival-second-price-exponential-immediate"
SCEN_2 = "survival-second-price-exponential"


def test_preset_matches_scenario_2_except_the_kernel():
    imm, s2 = dac.make(IMM), dac.make(SCEN_2)
    assert type(imm.kernel).__name__ == "ImmediateSurvivalKernel"
    assert type(s2.kernel).__name__ == "SurvivalKernel"
    assert (imm.competition.mu, imm.competition.sigma) == (s2.competition.mu, s2.competition.sigma) == (-1.0, 1.0)
    assert (imm.gamma.c, imm.gamma.beta) == (s2.gamma.c, s2.gamma.beta) == (0.9, 1.0)
    assert imm.kappa_distribution == "bernoulli"


def test_reward_is_realized_at_bid_time():
    """Winning every auction: each step's reward is (number of conversions at that bid) − payment, with at most one
    conversion, and no reward ever arrives on a lost auction."""
    env = AuctionEnv(dac.make(IMM, horizon_eta=50))
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            _, reward, done, _, info = env.step(np.array([1e9], dtype=np.float32))
            conversions = reward + info["auction"]["payment"]
            assert conversions == pytest.approx(round(conversions), abs=1e-9)
            assert round(conversions) in (0, 1)


def test_oracle_value_is_the_conversion_probability():
    """expected_reward = f = clip(S·(1ᵀu)(1ᵀx), 0, 1), with no horizon factor (it can be exactly 1)."""
    env = AuctionEnv(dac.make(IMM, horizon_eta=50))
    values = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            values.append(float(obs["expected_reward"].ravel()[0]))
            obs, _, done, _, _ = env.step(np.array([0.0], dtype=np.float32))
    values = np.array(values)
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert np.isclose(values, 1.0).any()      # clipped probabilities reach 1 (the delayed kernel stays below 1)
