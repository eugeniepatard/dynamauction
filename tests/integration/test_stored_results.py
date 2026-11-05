"""The current code reproduces the paper's results stored in benchmark/benchmark_results/main/.

Marked `paper`: not part of the default test run (`pytest -m paper` runs it; about 20 s). In CI it runs in its own
non-blocking job: a failure only means that the code now produces results that no longer align with the paper's
results, which is expected after an intentional change of the model or of the planning code; the paper's numbers are
reproduced by the version of the code they were computed with.

Each test recomputes a few test campaigns step by step and compares every column with the stored per-step results
(stored with 6 significant digits), so it fails as soon as the environment, the survival kernel, the carry-over context
(MultiplicativeGamma), the arrival / horizon sampling or the planning code no longer give the published numbers:
  - FiPla, every step of 20 campaigns at eta = 50 and eta = 100 (environment, kernel and context along FiPla's wins);
  - the stored bids of MPC(H=25, M=60) and of PPO replayed open-loop (the same, along other win patterns);
  - Shading(α=0.4, M=20) on two short campaigns and MPC(H=10, M=60) with the shipped valuation head on one, every
    decision, and the first decisions of MPC(H=25, M=60, α_cont=0.4) (the rollouts, CRN seeds and α selection).
Test campaigns: env seed s, policy seed s + 1000, as in the stored results.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import gymnasium as gym

import dynamauction as dac
from dynamauction.baseline.baseline_algo import MPCPolicy, ShadingPolicy
from dynamauction.baseline.estimators import MLPValuationHead, SimulationEstimator
from dynamauction.baseline.policies import ConstantPolicy

pytestmark = pytest.mark.paper
REPO = Path(__file__).resolve().parents[2]
WARNING = ("WARNING: the code being changed produces results that no longer align with the paper's results "
           "(benchmark/benchmark_results/main/)")
STORED = {eta: REPO / f"benchmark/benchmark_results/main/survival-second-price-exponential_eta{eta}_results_final.csv.gz"
          for eta in (50, 100)}
HEAD = (REPO / "benchmark/models/mlp_valuation_head"
        / "survival-second-price-exponential_h32-32_actrelu_ep1000_e500_lr0.001_s10000.pt")
SCENARIO = "survival-second-price-exponential"
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
COLUMNS = ["t", "reward", "cum_reward", "bid", "payment", "ctx", "expected_reward", "competing_bid"]
SHORT_CAMPAIGNS = [4, 12]   # 11 and 9 auctions at eta = 50: the planning policies stay cheap


@pytest.fixture(scope="module")
def stored():
    return {eta: pd.read_csv(path) for eta, path in STORED.items()}


def play(eta: int, seed: int, bidder, max_steps: int | None = None) -> pd.DataFrame:
    """One campaign, logged as the stored results: t, ctx and expected_reward of the next observation, the rest of the
    auction just played. bidder(env, obs, k) returns the bid of the k-th auction."""
    env = gym.make("DynamAuction-v0", scenario=dac.make(SCENARIO, horizon_eta=eta))
    obs, _ = env.reset(seed=seed)
    rows, cum_reward, done = [], 0.0, False
    while not done and (max_steps is None or len(rows) < max_steps):
        bid = bidder(env, obs, len(rows))
        obs, reward, done, _, info = env.step(np.array([bid], dtype=np.float32))
        cum_reward += float(reward)
        rows.append({"step": obs["step"], "t": float(obs["t"].ravel()[0]), "reward": float(reward), "cum_reward": cum_reward,
                     "won": bool(info["auction"]["won"]), "bid": float(info["auction"]["bid"]),
                     "payment": float(info["auction"]["payment"]), "ctx": float(obs["ctx"].ravel()[0]),
                     "expected_reward": float(obs["expected_reward"].ravel()[0]),
                     "competing_bid": float(info["auction"]["competing_bid"])})
    env.close()
    return pd.DataFrame(rows)


def assert_matches(stored_df: pd.DataFrame, policy: str, seed: int, got: pd.DataFrame, full: bool = True) -> None:
    expected = stored_df[(stored_df["policy"] == policy) & (stored_df["seed"] == seed)].sort_values("step")
    if full:
        assert len(got) == len(expected), (f"{WARNING}. {policy}, seed {seed}: {len(got)} auctions instead of "
                                           f"{len(expected)}")
    expected = expected.head(len(got))
    np.testing.assert_array_equal(got["step"].to_numpy(), expected["step"].to_numpy(),
                                  err_msg=f"{WARNING}. {policy}, seed {seed}")
    np.testing.assert_array_equal(got["won"].to_numpy(), expected["won"].astype(bool).to_numpy(),
                                  err_msg=f"{WARNING}. {policy}, seed {seed}: won")
    for col in COLUMNS:
        np.testing.assert_allclose(got[col].to_numpy(), expected[col].to_numpy(dtype=float), rtol=1e-5, atol=1e-6,
                                   err_msg=f"{WARNING}. {policy}, seed {seed}: {col}")


def stored_bids(stored_df: pd.DataFrame, policy: str, seed: int) -> np.ndarray:
    rows = stored_df[(stored_df["policy"] == policy) & (stored_df["seed"] == seed)].sort_values("step")
    return rows["bid"].to_numpy(dtype=float)


@pytest.mark.parametrize("eta", [50, 100])
def test_fipla_reproduces_the_stored_trajectories(stored, eta):
    for seed in range(20):
        got = play(eta, seed, lambda env, obs, k: max(0.0, float(obs["expected_reward"].ravel()[0])))
        assert_matches(stored[eta], "FiPla", seed, got)


@pytest.mark.parametrize("policy", ["MPC(H=25, M=60)", "PPO (training 1)"])
def test_replayed_bids_reproduce_the_stored_outcomes(stored, policy):
    """The environment alone (no planning, no network): the stored bids give back the stored contexts and rewards."""
    for seed in range(20):
        bids = stored_bids(stored[50], policy, seed)
        got = play(50, seed, lambda env, obs, k: bids[k])
        assert_matches(stored[50], policy, seed, got)


@pytest.mark.parametrize("seed", SHORT_CAMPAIGNS)
def test_shading_reproduces_the_stored_decisions(stored, seed):
    policy = ShadingPolicy(SimulationEstimator(ConstantPolicy(0.0), n_rollouts=20, seed=seed + 1000), alpha=0.4)

    def bidder(env, obs, k):
        if k == 0:
            policy.set_env(env)
        return policy.bid(obs)

    assert_matches(stored[50], "Shading(α=0.4, M=20)", seed, play(50, seed, bidder))


@pytest.mark.parametrize("name, horizon, alpha_cont, seed, max_steps", [
    ("MPC(H=10, M=60)", 10.0, 1.0, SHORT_CAMPAIGNS[1], None),
    ("MPC(H=25, M=60, α_cont=0.4)", 25.0, 0.4, SHORT_CAMPAIGNS[1], 3),
])
def test_mpc_reproduces_the_stored_decisions(stored, name, horizon, alpha_cont, seed, max_steps):
    policy = MPCPolicy(v0=MLPValuationHead.load(HEAD), alpha_candidates=ALPHAS, alpha_cont=alpha_cont, n_rollouts=60,
                       planning_horizon=horizon, seed=seed + 1000)

    def bidder(env, obs, k):
        if k == 0:
            policy.set_env(env)
        return policy.bid(obs)

    got = play(50, seed, bidder, max_steps)
    assert_matches(stored[50], name, seed, got, full=max_steps is None)
    expected_alpha = stored[50][(stored[50]["policy"] == name) & (stored[50]["seed"] == seed)].sort_values("step")
    np.testing.assert_allclose(policy.alpha_log, expected_alpha["alpha_chosen"].to_numpy(dtype=float)[:len(got)],
                               err_msg=f"{WARNING}. {name}, seed {seed}: alpha_chosen")
