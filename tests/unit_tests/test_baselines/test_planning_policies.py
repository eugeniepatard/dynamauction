"""
Tests for dynamauction/baseline/baseline_algo.py:
  - GreedyMyopicPolicy
  - ShadingPolicy
  - MPCPolicy
  - MPCOraclePolicy
  - MPCTimeBinnedPolicy
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from dynamauction.baseline.baseline_algo import (
    GreedyMyopicPolicy,
    MPCOraclePolicy,
    MPCPolicy,
    MPCTimeBinnedPolicy,
    ShadingPolicy,
)
from dynamauction.baseline.estimators import SimulationEstimator, MLPValuationHead
from dynamauction.baseline.policies import ConstantPolicy

@pytest.fixture
def mock_estimator():
    est = MagicMock(spec=SimulationEstimator)
    est.estimate.return_value = 0.6
    return est


@pytest.fixture
def live_env(base_env):
    obs, _ = base_env.reset(seed=0)
    return base_env, obs


@pytest.fixture
def mock_mlp():
    mlp = MagicMock(spec=MLPValuationHead)
    mlp.predict.return_value = 0.5
    return mlp

class TestGreedyMyopicPolicy:

    def test_bid_equals_delta(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.6
        policy = GreedyMyopicPolicy(mock_estimator)
        assert policy.bid({}) == pytest.approx(0.6)

    def test_bid_clips_negative_delta_to_zero(self, mock_estimator):
        mock_estimator.estimate.return_value = -0.4
        policy = GreedyMyopicPolicy(mock_estimator)
        assert policy.bid({}) == pytest.approx(0.0)

    def test_bid_returns_float(self, mock_estimator):
        policy = GreedyMyopicPolicy(mock_estimator)
        assert isinstance(policy.bid({}), float)

    def test_set_env_delegates_to_estimator(self, mock_estimator, base_env):
        policy = GreedyMyopicPolicy(mock_estimator)
        policy.set_env(base_env)
        mock_estimator.set_env.assert_called_once_with(base_env)

    def test_estimate_called_with_obs(self, mock_estimator):
        obs = {"key": 1}
        policy = GreedyMyopicPolicy(mock_estimator)
        policy.bid(obs)
        mock_estimator.estimate.assert_called_once_with(obs)

    def test_end_to_end_with_real_env(self, live_env):
        env, obs = live_env
        est = SimulationEstimator(pi0=ConstantPolicy(0.0), n_rollouts=1)
        policy = GreedyMyopicPolicy(est)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert isinstance(bid, float)
        assert bid >= 0.0

class TestShadingPolicy:

    def test_bid_is_alpha_times_delta(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.6
        policy = ShadingPolicy(mock_estimator, alpha=0.5)
        assert policy.bid({}) == pytest.approx(0.3)

    def test_bid_clips_negative_to_zero(self, mock_estimator):
        mock_estimator.estimate.return_value = -1.0
        policy = ShadingPolicy(mock_estimator, alpha=0.8)
        assert policy.bid({}) == pytest.approx(0.0)

    def test_alpha_one_equals_greedy(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.6
        greedy = GreedyMyopicPolicy(mock_estimator)
        shading = ShadingPolicy(mock_estimator, alpha=1.0)
        assert shading.bid({}) == pytest.approx(greedy.bid({}))

    def test_alpha_gt_one_amplifies_bid(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.5
        assert ShadingPolicy(mock_estimator, alpha=2.0).bid({}) == pytest.approx(1.0)

    def test_invalid_alpha_raises(self, mock_estimator):
        with pytest.raises(ValueError, match="alpha must be positive"):
            ShadingPolicy(mock_estimator, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            ShadingPolicy(mock_estimator, alpha=-1.0)

    def test_default_alpha_is_one(self, mock_estimator):
        mock_estimator.estimate.return_value = 0.4
        policy = ShadingPolicy(mock_estimator)
        assert policy.alpha == pytest.approx(1.0)
        assert policy.bid({}) == pytest.approx(0.4)

    def test_set_env_delegates_to_estimator(self, mock_estimator, base_env):
        policy = ShadingPolicy(mock_estimator, alpha=0.8)
        policy.set_env(base_env)
        mock_estimator.set_env.assert_called_once_with(base_env)

    def test_bid_returns_float(self, mock_estimator):
        assert isinstance(ShadingPolicy(mock_estimator, alpha=0.8).bid({}), float)

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0, 1.2])
    def test_bid_always_nonnegative(self, mock_estimator, alpha):
        mock_estimator.estimate.return_value = 0.5
        assert ShadingPolicy(mock_estimator, alpha=alpha).bid({}) >= 0.0


class TestMPCPolicy:

    def test_bid_raises_without_set_env(self, mock_mlp):
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1)
        with pytest.raises(RuntimeError, match="set_env"):
            policy.bid({})

    def test_set_env_stores_env(self, mock_mlp, base_env):
        policy = MPCPolicy(v0=mock_mlp)
        policy.set_env(base_env)
        assert policy._env is base_env

    def test_bid_returns_float(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        assert isinstance(policy.bid(obs), float)

    def test_bid_is_nonnegative(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        assert policy.bid(obs) >= 0.0

    def test_alpha_log_appended_on_each_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        assert policy.alpha_log == []
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        policy.bid(obs)
        assert len(policy.alpha_log) == 2

    def test_chosen_alpha_is_from_candidates(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0, 1.5]
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=candidates, n_rollouts=2,
                           planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        assert policy.alpha_log[-1] in candidates

    def test_bid_equals_alpha_star_times_v0(self, mock_mlp, live_env):
        env, obs = live_env
        mock_mlp.predict.return_value = 0.4
        # Single candidate: alpha* must be that candidate
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[2.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert bid == pytest.approx(2.0 * 0.4, abs=1e-5)

    def test_alpha_candidates_stored(self, mock_mlp):
        candidates = [0.6, 0.8, 1.0]
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=candidates)
        assert policy.alpha_candidates == candidates

    def test_default_alpha_candidates(self, mock_mlp):
        policy = MPCPolicy(v0=mock_mlp)
        assert len(policy.alpha_candidates) > 0

    def test_v0_predict_called_during_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        mock_mlp.predict.assert_called()

    def test_planning_horizon_zero_stops_after_first_step(self, mock_mlp, live_env):
        # With H=0, rollout terminates after 1 step, so MPC only sees immediate reward.
        # This is tested indirectly: bid should still return a valid float and not hang.
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert np.isfinite(bid)

    def test_end_to_end_real_mlp(self, live_env):
        from dynamauction.baseline.utils import _obs_to_features
        env, obs = live_env
        dim = _obs_to_features(obs).shape[0]
        mlp = MLPValuationHead(input_dim=dim, hidden_dims=(8,))
        policy = MPCPolicy(v0=mlp, alpha_candidates=[0.8, 1.0], n_rollouts=1,
                           planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert isinstance(bid, float)
        assert bid >= 0.0

    def test_invalid_crn_raises(self, mock_mlp):
        with pytest.raises(ValueError, match="crn"):
            MPCPolicy(v0=mock_mlp, crn="independent")

    def test_per_component_crn_bid(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0]
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=candidates, n_rollouts=2,
                           planning_horizon=5.0, crn="per_component")
        policy.set_env(env)
        bid = policy.bid(obs)
        assert policy.alpha_log[-1] in candidates
        assert bid == pytest.approx(policy.alpha_log[-1] * 0.5, abs=1e-5)
        # only the rollout copies get their own streams: the live env is untouched
        from dynamauction.baseline.utils import _OwnStream
        assert isinstance(env.unwrapped.rng, np.random.Generator)
        assert not isinstance(env.unwrapped.scenario.competition, _OwnStream)
        assert not isinstance(env.unwrapped.scenario.display_sampler, _OwnStream)


class TestCampaignAttribution:
    """MPCPolicy(attribution="campaign"): a rollout stopped at the end of its window counts, on top of the window's
    rewards, exactly the conversions that the environment credits afterwards (when no later auction is won)."""

    def test_invalid_attribution_raises(self, mock_mlp):
        with pytest.raises(ValueError, match="attribution"):
            MPCPolicy(v0=mock_mlp, attribution="horizon")

    @pytest.mark.parametrize("seed", [12, 14, 15])   # campaigns whose first display converts after a 1-unit window
    def test_counts_the_conversions_credited_after_the_window(self, mock_mlp, seed):
        import copy
        import gymnasium as gym
        import dynamauction as dac
        from dynamauction.baseline.utils import _per_component_rngs

        env = gym.make("DynamAuction-v0", scenario=dac.make("survival-second-price-exponential"))
        obs, _ = env.reset(seed=seed)
        t_start = float(obs["t"].ravel()[0])
        copies = []
        for _ in range(2):
            e = copy.deepcopy(env)
            e.unwrapped.rng = np.random.default_rng(11)
            _per_component_rngs(e, 11)
            copies.append(e)
        kw = dict(v0=mock_mlp, alpha_cont=1.0, planning_horizon=1.0)
        with_after = MPCPolicy(attribution="campaign", **kw)._mpc_rollout(copies[0], 1.0, 0.5, t_start)
        window_only = MPCPolicy(attribution="window", **kw)._mpc_rollout(copies[1], 1.0, 0.5, t_start)
        e, done, credited_after = copies[1], False, 0.0
        while not done and e.unwrapped.state.step_idx < len(e.unwrapped.auction_times):   # bid 0 until the end
            _, reward, done, _, _ = e.step(np.array([0.0], dtype=np.float32))
            credited_after += float(reward)
        assert credited_after > 0   # the case is not trivial: some conversions are credited after the window
        assert with_after == pytest.approx(window_only + credited_after, abs=1e-12)


class TestV0Reward:
    """MPCPolicy(reward="v0"): a simulated win is worth V₀ − payment (the paper's g), the simulated conversions are
    not counted."""

    def test_invalid_reward_raises(self, mock_mlp):
        with pytest.raises(ValueError, match="reward"):
            MPCPolicy(v0=mock_mlp, reward="expected")
        with pytest.raises(ValueError, match="attribution"):
            MPCPolicy(v0=mock_mlp, reward="v0", attribution="campaign")

    def test_without_lookahead_bids_truthfully(self, mock_mlp, live_env):
        """H = 0: Q̂(α) = mean of W(α·V₀, B)·(V₀ − B), maximised by bidding V₀ (α = 1) in a second-price auction."""
        env, obs = live_env
        policy = MPCPolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_rollouts=50, planning_horizon=0.0,
                           reward="v0", crn="per_component")
        policy.set_env(env)
        policy.bid(obs)
        assert policy.alpha_log[-1] == 1.0

    @pytest.mark.parametrize("seed", [12, 14, 15])
    def test_return_is_v0_minus_payment_of_the_wins(self, mock_mlp, seed):
        import copy
        import gymnasium as gym
        import dynamauction as dac
        from dynamauction.baseline.utils import _per_component_rngs

        env = gym.make("DynamAuction-v0", scenario=dac.make("survival-second-price-exponential"))
        obs, _ = env.reset(seed=seed)
        t_start, horizon = float(obs["t"].ravel()[0]), 5.0
        copies = []
        for _ in range(2):
            e = copy.deepcopy(env)
            e.unwrapped.rng = np.random.default_rng(11)
            _per_component_rngs(e, 11)
            copies.append(e)
        got = MPCPolicy(v0=mock_mlp, alpha_cont=0.8, planning_horizon=horizon, reward="v0")._mpc_rollout(
            copies[0], 1.0, 0.5, t_start)
        e, expected, wins = copies[1], 0.0, 0   # replay: same bids, value = Σ over the wins of 0.5 − payment
        o, _, done, _, info = e.step(np.array([1.0 * 0.5], dtype=np.float32))
        while True:
            if info["auction"]["won"]:
                expected, wins = expected + 0.5 - info["auction"]["payment"], wins + 1
            if done or float(o["t"].ravel()[0]) - t_start > horizon:
                break
            o, _, done, _, info = e.step(np.array([0.8 * 0.5], dtype=np.float32))
        assert wins > 0
        assert got == pytest.approx(expected, abs=1e-9)


class TestPerComponentCRN:
    """_per_component_rngs (MPCPolicy(crn="per_component")): two rollouts with the same seed share the competing bids
    and the displays of every auction, even after a forced win in one and a forced loss in the other."""

    @staticmethod
    def _branches(per_component):
        import copy
        import gymnasium as gym
        import dynamauction as dac
        from dynamauction.baseline.utils import _per_component_rngs

        env = gym.make("DynamAuction-v0", scenario=dac.make("survival-second-price-exponential"))
        env.reset(seed=5)
        out = []
        for first_bid in (1e12, 0.0):   # forced win, forced loss
            e = copy.deepcopy(env)
            e.unwrapped.rng = np.random.default_rng(7)
            if per_component:
                _per_component_rngs(e, 7)
            obs, _, done, _, _ = e.step(np.array([first_bid], dtype=np.float32))
            bids, displays = [], []
            while not done and len(bids) < 5:
                displays.append(obs["display_features"].copy())
                obs, _, done, _, info = e.step(np.array([0.0], dtype=np.float32))
                bids.append(info["auction"]["competing_bid"])
            out.append((bids, displays))
        return out

    def test_draws_shared_after_different_outcomes(self):
        (bids_win, x_win), (bids_loss, x_loss) = self._branches(per_component=True)
        assert len(bids_win) == 5
        np.testing.assert_allclose(bids_win, bids_loss)
        np.testing.assert_allclose(x_win, x_loss)

    def test_shared_stream_draws_shift_after_different_outcomes(self):
        (bids_win, _), (bids_loss, _) = self._branches(per_component=False)
        assert not np.allclose(bids_win, bids_loss)


class TestMPCOraclePolicy:

    def test_bid_raises_without_set_env(self, mock_mlp):
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1)
        with pytest.raises(RuntimeError, match="set_env"):
            policy.bid({})

    def test_set_env_stores_env(self, mock_mlp, base_env):
        policy = MPCOraclePolicy(v0=mock_mlp)
        policy.set_env(base_env)
        assert policy._env is base_env

    def test_bid_returns_float(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        assert isinstance(policy.bid(obs), float)

    def test_bid_is_nonnegative(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        assert policy.bid(obs) >= 0.0

    def test_alpha_log_grows_with_each_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        assert policy.alpha_log == []
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        policy.bid(obs)
        assert len(policy.alpha_log) == 2

    def test_chosen_alpha_is_from_candidates(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0, 1.5]
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=candidates, n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        assert policy.alpha_log[-1] in candidates

    def test_bid_equals_alpha_star_times_v0(self, mock_mlp, live_env):
        env, obs = live_env
        mock_mlp.predict.return_value = 0.4
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[2.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        bid = policy.bid(obs)
        assert bid == pytest.approx(2.0 * 0.4, abs=1e-5)

    def test_alpha_candidates_stored(self, mock_mlp):
        candidates = [0.6, 0.8, 1.0]
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=candidates)
        assert policy.alpha_candidates == candidates

    def test_v0_predict_called_during_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCOraclePolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1,
                                 planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        mock_mlp.predict.assert_called()


class TestMPCTimeBinnedPolicy:

    def test_bid_raises_without_set_env(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=1)
        with pytest.raises(RuntimeError):
            policy.bid({})

    def test_set_env_stores_env(self, mock_mlp, base_env):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp)
        policy.set_env(base_env)
        assert policy._env is base_env

    def test_bid_returns_float(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_bins=2,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        assert isinstance(policy.bid(obs), float)

    def test_bid_is_nonnegative(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_bins=2,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        assert policy.bid(obs) >= 0.0

    def test_alpha_log_contains_tuples(self, mock_mlp, live_env):
        env, obs = live_env
        n_bins = 2
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_bins=n_bins,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        assert isinstance(policy.alpha_log[0], tuple)
        assert len(policy.alpha_log[0]) == n_bins

    def test_chosen_alpha_vec_from_candidates(self, mock_mlp, live_env):
        env, obs = live_env
        candidates = [0.5, 1.0]
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=candidates, n_bins=2,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        policy.bid(obs)
        alpha_vec = policy.alpha_log[-1]
        assert all(a in candidates for a in alpha_vec)

    def test_n_bins_stored(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=3)
        assert policy.n_bins == 3

    def test_alpha_log_grows_with_each_bid(self, mock_mlp, live_env):
        env, obs = live_env
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, alpha_candidates=[1.0], n_bins=1,
                                     n_rollouts=1, planning_horizon=0.0)
        policy.set_env(env)
        assert policy.alpha_log == []
        policy.bid(obs)
        assert len(policy.alpha_log) == 1
        policy.bid(obs)
        assert len(policy.alpha_log) == 2

    # ── non-uniform bins (bin_edges) ──────────────────────────────────────────

    def test_default_bins_are_equal_width(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=2, planning_horizon=10.0)
        assert policy.bin_edges is None
        assert [policy._bin_index(e, 5.0) for e in (0.0, 4.9, 5.0, 9.9, 12.0)] == [0, 0, 1, 1, 1]

    def test_bin_edges_assign_bins(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=2, planning_horizon=25.0, bin_edges=(1.5,))
        assert [policy._bin_index(e, 12.5) for e in (0.1, 1.49, 1.5, 7.0, 24.9)] == [0, 0, 1, 1, 1]
        three = MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=3, planning_horizon=25.0, bin_edges=(1.0, 5.0))
        assert [three._bin_index(e, 0.0) for e in (0.5, 1.0, 4.0, 5.0, 20.0)] == [0, 1, 1, 2, 2]

    def test_bin_edges_zero_puts_only_the_current_auction_in_bin_0(self, mock_mlp):
        policy = MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=2, planning_horizon=25.0, bin_edges=(0.0,))
        assert all(policy._bin_index(e, 12.5) == 1 for e in (1e-9, 0.3, 10.0))

    @pytest.mark.parametrize("edges, n_bins", [((1.5,), 3), ((2.0, 1.0), 3), ((-1.0,), 2)])
    def test_bin_edges_validation(self, mock_mlp, edges, n_bins):
        with pytest.raises(ValueError):
            MPCTimeBinnedPolicy(v0=mock_mlp, n_bins=n_bins, bin_edges=edges)

    def test_explicit_uniform_edges_match_default(self, mock_mlp, live_env):
        """bin_edges=(H/2,) with K=2 gives exactly the default (equal-width) decisions."""
        env, obs = live_env
        kw = dict(v0=mock_mlp, alpha_candidates=[0.5, 1.0], n_bins=2, n_rollouts=3, planning_horizon=6.0, seed=0)
        default, explicit = MPCTimeBinnedPolicy(**kw), MPCTimeBinnedPolicy(**kw, bin_edges=(3.0,))
        for p in (default, explicit):
            p.set_env(env)
        assert default.bid(obs) == explicit.bid(obs)
        assert default.alpha_log == explicit.alpha_log


class TestNoOracleAccess:
    """
    MPCPolicy and MPCTimeBinnedPolicy must not see the true future auction schedule.
    They must call _resample_remaining_horizon on every rollout copy to replace the
    pre-baked horizon and arrival times with a fresh independent draw.
    MPCOraclePolicy is the deliberate exception: it skips resampling (oracle access).
    """

    # ── mechanism tests: count _resample_remaining_horizon invocations ────────

    def test_mpc_calls_resample_per_rollout(self, mock_mlp, live_env):
        """_resample_remaining_horizon is called exactly n_candidates × n_rollouts times."""
        env, obs = live_env
        n_candidates, n_rollouts = 3, 4
        policy = MPCPolicy(
            v0=mock_mlp,
            alpha_candidates=[0.5, 1.0, 1.5],
            n_rollouts=n_rollouts,
            planning_horizon=0.0,
        )
        policy.set_env(env)

        with patch("dynamauction.baseline.baseline_algo._resample_remaining_horizon") as mock_resample:
            policy.bid(obs)

        assert mock_resample.call_count == n_candidates * n_rollouts

    def test_mpc_timebinned_calls_resample_per_rollout(self, mock_mlp, live_env):
        """MPCTimeBinnedPolicy calls _resample_remaining_horizon once per (candidate_vec × rollout)."""
        env, obs = live_env
        n_candidates, n_bins, n_rollouts = 2, 2, 3
        policy = MPCTimeBinnedPolicy(
            v0=mock_mlp,
            alpha_candidates=[0.8, 1.2],
            n_bins=n_bins,
            n_rollouts=n_rollouts,
            planning_horizon=0.0,
        )
        policy.set_env(env)

        with patch("dynamauction.baseline.baseline_algo._resample_remaining_horizon") as mock_resample:
            policy.bid(obs)

        assert mock_resample.call_count == (n_candidates ** n_bins) * n_rollouts

    def test_mpc_oracle_never_calls_resample(self, mock_mlp, live_env):
        """MPCOraclePolicy never calls _resample_remaining_horizon (oracle access by design)."""
        env, obs = live_env
        policy = MPCOraclePolicy(
            v0=mock_mlp,
            alpha_candidates=[0.5, 1.0, 1.5],
            n_rollouts=4,
            planning_horizon=0.0,
        )
        policy.set_env(env)

        with patch("dynamauction.baseline.baseline_algo._resample_remaining_horizon") as mock_resample:
            policy.bid(obs)

        assert mock_resample.call_count == 0

    # ── behavioral tests: verify actual auction_times state ───────────────────

    def test_mpc_rollout_copies_have_resampled_auction_times(self, mock_mlp, live_env):
        """After resampling, rollout copies have auction_times different from the live env."""
        from dynamauction.baseline.utils import _resample_remaining_horizon as real_fn

        env, obs = live_env
        original_times = env.unwrapped.auction_times.copy()

        captured_after: list = []

        def spy(env_copy, rng):
            real_fn(env_copy, rng)
            captured_after.append(env_copy.unwrapped.auction_times.copy())

        policy = MPCPolicy(
            v0=mock_mlp,
            alpha_candidates=[1.0],
            n_rollouts=2,
            planning_horizon=5.0,
        )
        policy.set_env(env)

        with patch("dynamauction.baseline.baseline_algo._resample_remaining_horizon", side_effect=spy):
            policy.bid(obs)

        assert len(captured_after) > 0
        for times in captured_after:
            assert not np.array_equal(times, original_times), (
                "Rollout copy retains original auction_times — _resample_remaining_horizon had no effect."
            )

    def test_live_env_auction_times_unchanged_after_mpc_bid(self, mock_mlp, live_env):
        """MPCPolicy rollout deepcopies never mutate the live env's auction_times."""
        env, obs = live_env
        original_times = env.unwrapped.auction_times.copy()

        policy = MPCPolicy(
            v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=3, planning_horizon=5.0
        )
        policy.set_env(env)
        policy.bid(obs)

        assert np.array_equal(env.unwrapped.auction_times, original_times)

    def test_live_env_auction_times_unchanged_after_oracle_bid(self, mock_mlp, live_env):
        """MPCOraclePolicy rollout deepcopies never mutate the live env's auction_times."""
        env, obs = live_env
        original_times = env.unwrapped.auction_times.copy()

        policy = MPCOraclePolicy(
            v0=mock_mlp, alpha_candidates=[1.0], n_rollouts=3, planning_horizon=5.0
        )
        policy.set_env(env)
        policy.bid(obs)

        assert np.array_equal(env.unwrapped.auction_times, original_times)
