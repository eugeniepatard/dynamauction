import numpy as np
import pytest

from dynamicauc.baseline.policies import BiddingPolicy, ConstantPolicy, FiPlaPolicy, RandomPolicy


class TestBiddingPolicyInterface:

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BiddingPolicy()

    def test_concrete_subclass_must_implement_bid(self):
        class NoBid(BiddingPolicy):
            pass
        with pytest.raises(TypeError):
            NoBid()


class TestConstantPolicy:

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 10.0])
    def test_bid_returns_configured_value(self, value):
        assert ConstantPolicy(value).bid({}) == pytest.approx(value)

    def test_bid_ignores_observations(self):
        policy = ConstantPolicy(2.0)
        assert policy.bid({"some_key": 999}) == pytest.approx(2.0)

    def test_bid_is_deterministic(self):
        policy = ConstantPolicy(1.5)
        bids = [policy.bid({}) for _ in range(20)]
        assert all(b == pytest.approx(1.5) for b in bids)

    def test_default_bid_value(self):
        assert ConstantPolicy().bid({}) == pytest.approx(0.5)


class TestFiPlaPolicy:

    def _obs(self, value):
        return {"expected_reward": np.array([value], dtype=np.float32)}

    @pytest.mark.parametrize("value", [0.0, 0.25, 1.0, 3.14])
    def test_bid_equals_expected_reward(self, value):
        assert FiPlaPolicy().bid(self._obs(value)) == pytest.approx(value)

    def test_bid_works_with_2d_array(self):
        obs = {"expected_reward": np.array([[0.7]], dtype=np.float32)}
        assert FiPlaPolicy().bid(obs) == pytest.approx(0.7)

    def test_bid_returns_float(self):
        assert isinstance(FiPlaPolicy().bid(self._obs(1.0)), float)


class TestRandomPolicy:

    @pytest.mark.parametrize("max_bid", [1.0, 2.0, 5.0])
    def test_bid_in_range(self, max_bid):
        policy = RandomPolicy(max_bid=max_bid)
        for _ in range(50):
            b = policy.bid({})
            assert 0.0 <= b <= max_bid

    def test_bid_is_nondeterministic(self):
        policy = RandomPolicy(max_bid=10.0)
        bids = {policy.bid({}) for _ in range(30)}
        assert len(bids) > 1

    def test_default_max_bid(self):
        policy = RandomPolicy()
        for _ in range(20):
            assert 0.0 <= policy.bid({}) <= 2.0
