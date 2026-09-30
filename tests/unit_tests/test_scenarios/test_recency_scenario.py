"""
Integration tests for the recency-second-price-lognormal scenario.

These tests guard against the -inf ctx bug: LastWonTimeGamma initialises ctx
to -1e6 (finite sentinel for "never won").  Before the fix, ctx was float("-inf"),
which propagated NaN through MLP features and corrupted VecNormalize stats.
"""
import math

import gymnasium as gym
import numpy as np
import pytest

import dynamauction as dac
from dynamauction.baseline.policies import ConstantPolicy, FiPlaPolicy
from dynamauction.baseline.utils import _obs_to_features


SCENARIO_ID = "recency-second-price-lognormal"
N_EPISODES  = 10
SEEDS       = list(range(N_EPISODES))


@pytest.fixture(scope="module")
def recency_env():
    scenario = dac.make(SCENARIO_ID)
    env = gym.make("DynamAuction-v0", scenario=scenario)
    yield env
    env.close()


def _run_episode(env, policy, seed: int) -> tuple[float, list]:
    """Run one episode; return (total_reward, list_of_obs)."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    total = 0.0
    obs_list = [obs]
    done = False
    while not done:
        bid = policy.bid(obs)
        obs, reward, done, _, _ = env.step(np.array([bid], dtype=np.float32))
        total += float(reward)
        obs_list.append(obs)
    return total, obs_list


# ── ctx sentinel ───────────────────────────────────────────────────────────────

class TestCtxSentinel:

    def test_initial_ctx_is_finite(self, recency_env):
        obs, _ = recency_env.reset(seed=0)
        ctx = float(obs["ctx"].ravel()[0])
        assert math.isfinite(ctx), f"ctx at episode start is not finite: {ctx}"

    def test_initial_ctx_is_large_negative(self, recency_env):
        obs, _ = recency_env.reset(seed=0)
        ctx = float(obs["ctx"].ravel()[0])
        assert ctx == pytest.approx(-1e6)

    def test_ctx_stays_finite_throughout_episode(self, recency_env):
        obs, _ = recency_env.reset(seed=0)
        done = False
        while not done:
            bid = FiPlaPolicy().bid(obs)
            obs, _, done, _, _ = recency_env.step(np.array([bid], dtype=np.float32))
            ctx = float(obs["ctx"].ravel()[0])
            assert math.isfinite(ctx), f"ctx became non-finite mid-episode: {ctx}"

    def test_ctx_increases_after_win(self, recency_env):
        """After a win ctx should jump to (approximately) the current auction time."""
        obs, _ = recency_env.reset(seed=7)
        initial_ctx = float(obs["ctx"].ravel()[0])
        done = False
        found_update = False
        while not done:
            bid = 999.0   # always win
            obs, _, done, _, _ = recency_env.step(np.array([bid], dtype=np.float32))
            ctx = float(obs["ctx"].ravel()[0])
            if ctx != initial_ctx and ctx > -1e6 + 1:
                found_update = True
                t_now = float(obs["t"].ravel()[0])
                assert ctx <= t_now + 1e-3
                break
        assert found_update, "ctx never updated despite always-winning bids"


# ── observation finite-value guarantee ────────────────────────────────────────

class TestObsFinite:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_obs_no_nan(self, recency_env, seed):
        _, obs_list = _run_episode(recency_env, FiPlaPolicy(), seed)
        for obs in obs_list:
            for key, arr in obs.items():
                a = np.asarray(arr, dtype=float)
                assert not np.any(np.isnan(a)), f"NaN in obs['{key}'] at seed {seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_obs_no_inf(self, recency_env, seed):
        _, obs_list = _run_episode(recency_env, FiPlaPolicy(), seed)
        for obs in obs_list:
            for key, arr in obs.items():
                a = np.asarray(arr, dtype=float)
                assert not np.any(np.isinf(a)), f"inf in obs['{key}'] at seed {seed}"


# ── MLP feature vector ────────────────────────────────────────────────────────

class TestMlpFeatures:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_features_no_nan(self, recency_env, seed):
        _, obs_list = _run_episode(recency_env, FiPlaPolicy(), seed)
        for obs in obs_list[:-1]:   # last obs may not have display features
            feat = _obs_to_features(obs)
            assert not np.any(np.isnan(feat)), f"NaN in features at seed {seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_features_no_inf(self, recency_env, seed):
        _, obs_list = _run_episode(recency_env, FiPlaPolicy(), seed)
        for obs in obs_list[:-1]:
            feat = _obs_to_features(obs)
            assert not np.any(np.isinf(feat)), f"inf in features at seed {seed}"


# ── expected_reward is finite ─────────────────────────────────────────────────

class TestExpectedReward:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_expected_reward_finite(self, recency_env, seed):
        _, obs_list = _run_episode(recency_env, FiPlaPolicy(), seed)
        for obs in obs_list[:-1]:
            er = float(obs["expected_reward"].ravel()[0])
            assert math.isfinite(er), f"expected_reward is not finite at seed {seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_expected_reward_non_negative(self, recency_env, seed):
        _, obs_list = _run_episode(recency_env, FiPlaPolicy(), seed)
        for obs in obs_list[:-1]:
            er = float(obs["expected_reward"].ravel()[0])
            assert er >= -1e-9, f"expected_reward < 0 at seed {seed}: {er}"


# ── episode reward with FiPla and ConstantPolicy ──────────────────────────────

class TestEpisodeReward:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fipla_reward_finite(self, recency_env, seed):
        reward, _ = _run_episode(recency_env, FiPlaPolicy(), seed)
        assert math.isfinite(reward), f"FiPla total reward not finite at seed {seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_constant_policy_reward_finite(self, recency_env, seed):
        reward, _ = _run_episode(recency_env, ConstantPolicy(0.5), seed)
        assert math.isfinite(reward), f"ConstantPolicy reward not finite at seed {seed}"

    def test_fipla_beats_zero_bid(self, recency_env):
        """FiPla should earn more than always bidding 0 on average."""
        fipla_rewards  = [_run_episode(recency_env, FiPlaPolicy(),      s)[0] for s in SEEDS]
        zero_rewards   = [_run_episode(recency_env, ConstantPolicy(0.0), s)[0] for s in SEEDS]
        assert np.mean(fipla_rewards) > np.mean(zero_rewards)


# ── flat RL obs pipeline ──────────────────────────────────────────────────────

class TestFlatObsPipeline:
    """Checks the RL wrapper chain (MaskExpectedReward + FlatAuction) is also clean."""

    @pytest.fixture(scope="class")
    @classmethod
    def flat_env(cls):
        scenario = dac.make(SCENARIO_ID)
        env = gym.make("DynamAuction-v0", scenario=scenario)
        env = dac.MaskExpectedRewardWrapper(env)
        env = dac.FlatAuctionObsWrapper(env)
        yield env
        env.close()

    def test_initial_obs_no_inf(self, flat_env):
        obs, _ = flat_env.reset(seed=0)
        assert not np.any(np.isinf(obs)), "inf in initial flat obs"

    def test_initial_obs_no_nan(self, flat_env):
        obs, _ = flat_env.reset(seed=0)
        assert not np.any(np.isnan(obs)), "NaN in initial flat obs"

    @pytest.mark.parametrize("seed", SEEDS[:3])
    def test_full_episode_obs_finite(self, flat_env, seed):
        obs, _ = flat_env.reset(seed=seed)
        done = False
        while not done:
            action = np.array([0.5], dtype=np.float32)
            obs, _, done, _, _ = flat_env.step(action)
            assert not np.any(np.isinf(obs)), f"inf in flat obs at seed {seed}"
            assert not np.any(np.isnan(obs)), f"NaN in flat obs at seed {seed}"
