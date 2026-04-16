import numpy as np
import pytest
from dynamicauc import ConstantPolicy, RandomPolicy, FiPlaPolicy

def test_constant_policy():
    policy = ConstantPolicy(bid_value=1.23)
    assert policy.bid({}) == 1.23

def test_random_policy():
    policy = RandomPolicy(max_bid=2.0)
    bid = policy.bid({})
    assert 0.0 <= bid <= 2.0

def test_fipla_policy():
    policy = FiPlaPolicy()
    obs = {
        "expected_reward": np.array([3.14], dtype=np.float32)
    }
    assert pytest.approx(policy.bid(obs)) == 3.14