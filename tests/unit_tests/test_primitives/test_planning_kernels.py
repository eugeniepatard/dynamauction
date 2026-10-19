"""
EffectiveFrequencyKernel (wear-in then wear-out of repeated exposures) and ReminderKernel (value peaks some time after
the previous exposure), and their presets.
"""
import numpy as np
import pytest

import dynamauction as dac
from dynamauction import AuctionEnv, EffectiveFrequencyKernel, ReminderKernel
from dynamauction.primitives import AdditiveGamma, LastWonTimeGamma

U = np.zeros(3)
X = np.zeros(3)


class TestEffectiveFrequencyKernel:

    def test_conversion_probability(self):
        k = EffectiveFrequencyKernel(base_rate=1.0, wear_in_mid=2.0, wear_in_slope=0.7, saturation_scale=10.0)
        for n in (0, 1, 2, 5, 20):
            expected = 1.0 / (1.0 + np.exp(-(n - 2.0) / 0.7)) * np.exp(-n / 10.0)
            assert k._conversion_prob(n) == pytest.approx(expected)

    def test_wear_in_then_wear_out(self):
        k = EffectiveFrequencyKernel()
        p = [k._conversion_prob(n) for n in range(30)]
        peak = int(np.argmax(p))
        assert 2 <= peak <= 6
        assert all(np.diff(p[:peak + 1]) > 0)           # wear-in
        assert all(np.diff(p[peak + 1:]) < 0)           # wear-out
        assert p[0] < 0.1 * max(p) + 0.05               # the first exposure rarely converts

    def test_rates_and_expected_reward(self):
        k = EffectiveFrequencyKernel(delay_rate=0.5)
        p = k._conversion_prob(3)
        assert k.kappa(U, X, 3, 0.0) == 0.0
        assert k.lambda_(U, X, 3, 0.0, 1.0, 0.0) == pytest.approx(p * 0.5)
        assert k.lambda_upper_bound(U, X, 3, 1.0, 10.0) == pytest.approx(p * 0.5)
        assert k.expected_reward(U, X, 3, lambda s: 0.0, 1.0, 5.0) == pytest.approx(p * (1 - np.exp(-0.5 * 4.0)))
        assert k.expected_reward(U, X, 3, lambda s: 0.0, 5.0, 5.0) == 0.0


class TestReminderKernel:

    def test_conversion_probability(self):
        k = ReminderKernel(p_cold=0.1, p_peak=0.8, peak_time=2.0)
        for tau in (0.5, 2.0, 5.0):
            expected = 0.1 + 0.7 * (tau / 2.0) * np.exp(1 - tau / 2.0)
            assert k._conversion_prob(10.0, 10.0 + tau) == pytest.approx(expected)

    def test_window_shape(self):
        k = ReminderKernel(p_cold=0.1, p_peak=0.8, peak_time=2.0)
        assert k._conversion_prob(0.0, 0.0) == pytest.approx(0.1)          # right after a win: no reminder effect
        assert k._conversion_prob(0.0, 2.0) == pytest.approx(0.8)          # peak at peak_time
        assert k._conversion_prob(-1000.0, 0.0) == pytest.approx(0.1)      # cold contact (before any win)
        assert k._conversion_prob(0.0, 30.0) == pytest.approx(0.1, abs=1e-4)   # forgotten

    def test_expected_reward(self):
        k = ReminderKernel(delay_rate=0.5)
        p = k._conversion_prob(1.0, 3.0)
        assert k.expected_reward(U, X, 1.0, lambda s: 0.0, 3.0, 7.0) == pytest.approx(p * (1 - np.exp(-0.5 * 4.0)))


@pytest.mark.parametrize("preset, kernel_cls, gamma_cls", [
    ("effective-frequency-second-price-lognormal", EffectiveFrequencyKernel, AdditiveGamma),
    ("reminder-second-price-lognormal", ReminderKernel, LastWonTimeGamma),
])
def test_preset_runs(preset, kernel_cls, gamma_cls):
    sc = dac.make(preset, horizon_eta=50)
    assert isinstance(sc.kernel, kernel_cls) and isinstance(sc.gamma, gamma_cls)
    assert (sc.competition.mu, sc.competition.sigma) == (-1.0, 1.0)
    env = AuctionEnv(sc)
    obs, _ = env.reset(seed=0)
    done, total = False, 0.0
    while not done:                                     # FiPla: bids the oracle value
        obs, r, done, _, _ = env.step(np.array([float(obs["expected_reward"][0])], dtype=np.float32))
        total += r
    assert np.isfinite(total)


def test_reminder_preset_initial_context():
    env = AuctionEnv(dac.make("reminder-second-price-lognormal", horizon_eta=50))
    obs, _ = env.reset(seed=0)
    assert float(obs["ctx"][0]) == pytest.approx(-1000.0)


@pytest.mark.parametrize("kernel", [EffectiveFrequencyKernel(w_u=[1, 1, 1], w_x=[1, 1, 1]),
                                    ReminderKernel(w_u=[1, 1, 1], w_x=[1, 1, 1])])
def test_relevance_multiplies_the_exposure_effect(kernel):
    u, x = np.array([0.2, 0.1, 0.1]), np.array([0.5, 0.3, 0.2])          # (1ᵀu)(1ᵀx) = 0.4
    effect = kernel._conversion_prob(3.0) if isinstance(kernel, EffectiveFrequencyKernel) else kernel._conversion_prob(1.0, 3.0)
    args = (u, x, 3.0) if isinstance(kernel, EffectiveFrequencyKernel) else (u, x, 1.0, 3.0)
    assert kernel.relevance(u, x) == pytest.approx(0.4)
    assert kernel.conversion_probability(*args) == pytest.approx(0.4 * effect)
    assert kernel.relevance(u, -x) == 0.0                               # negative product → irrelevant display
    assert kernel.relevance(3 * u, 3 * x) == 1.0                         # clipped at 1
    ctx, t = (3.0, 5.0) if isinstance(kernel, EffectiveFrequencyKernel) else (1.0, 3.0)
    assert kernel.expected_reward(u, x, ctx, lambda s: 0.0, t, t + 2.0) == pytest.approx(
        0.4 * effect * (1 - np.exp(-kernel.delay_rate * 2.0)))


def test_without_weights_the_relevance_is_one():
    assert EffectiveFrequencyKernel().relevance(np.ones(3), -np.ones(3)) == 1.0
    assert ReminderKernel().relevance(np.ones(3), -np.ones(3)) == 1.0


@pytest.mark.parametrize("preset", ["effective-frequency-second-price-lognormal", "reminder-second-price-lognormal"])
def test_presets_use_the_survival_relevance(preset):
    k = dac.make(preset).kernel
    assert k.w_u == (1.0, 1.0, 1.0) and k.w_x == (1.0, 1.0, 1.0)
