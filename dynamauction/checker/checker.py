"""
check_env: a Gymnasium-style sanity checker for AuctionEnv.

Verifies scenario component types, numeric invariants (non-negativity,
shape consistency, upper-bound relationships), and runs a live rollout
to confirm obs / reward / done / info contracts.

Usage
-----
    from dynamauction.checker import check_env
    check_env(env)               # raises AssertionError on any error
    check_env(env, warn_only=True)   # only warns, never raises
"""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from dynamauction.env import AuctionEnv, OBSERVATION_KEYS
from dynamauction.types import Scenario
from dynamauction.primitives import (
    AuctionRule,
    ArrivalProcess,
    CompetitionDistribution,
    DisplaySampler,
    ExogenousProcess,
    RewardKernel,
    HorizonGenerator,
    UserFeatureGenerator,
    Gamma,
    ConversionProbe,
    DelaySampler,
)

class _Issues:
    def __init__(self) -> None:
        self._errors: List[str] = []
        self._warnings: List[str] = []

    def error(self, msg: str) -> None:
        self._errors.append(msg)

    def warn(self, msg: str) -> None:
        self._warnings.append(msg)

    def report(self, warn_only: bool) -> None:
        for w in self._warnings:
            warnings.warn(f"[check_env] {w}", stacklevel=4)
        if self._errors:
            bullet = "\n  - "
            msg = f"[check_env] {len(self._errors)} error(s) found:" + bullet + bullet.join(self._errors)
            if warn_only:
                warnings.warn(msg, stacklevel=4)
            else:
                raise AssertionError(msg)


def _try(issues: _Issues, label: str, fn, *args, **kwargs):
    """Call fn(*args, **kwargs); catch and record any exception as an error."""
    try:
        return fn(*args, **kwargs), True
    except Exception as exc:
        issues.error(f"{label} raised {type(exc).__name__}: {exc}")
        return None, False

def _check_scenario_structure(sc: Scenario, issues: _Issues) -> None:
    if not isinstance(sc, Scenario):
        issues.error(f"env.scenario must be a Scenario, got {type(sc)}")
        return

    if not isinstance(sc.conversion_value, (int, float)) or sc.conversion_value <= 0:
        issues.error(f"scenario.conversion_value must be > 0, got {sc.conversion_value}")
    if not isinstance(sc.max_history, int) or sc.max_history < 1:
        issues.error(f"scenario.max_history must be a positive int, got {sc.max_history}")

    has_probe = sc.probe is not None
    has_delay = sc.delay_sampler is not None
    if has_probe != has_delay:
        issues.error(
            "scenario.probe and scenario.delay_sampler must both be set or both be None "
            f"(probe={type(sc.probe).__name__}, delay_sampler={type(sc.delay_sampler).__name__})"
        )
    if has_probe and not isinstance(sc.probe, ConversionProbe):
        issues.error(f"scenario.probe must be a ConversionProbe, got {type(sc.probe)}")
    if has_delay and not isinstance(sc.delay_sampler, DelaySampler):
        issues.error(f"scenario.delay_sampler must be a DelaySampler, got {type(sc.delay_sampler)}")

    for attr, cls in [
        ("auction",               AuctionRule),
        ("competition",           CompetitionDistribution),
        ("arrival",               ArrivalProcess),
        ("kernel",                RewardKernel),
        ("display_sampler",       DisplaySampler),
        ("exogenous",             ExogenousProcess),
        ("horizon_generator",     HorizonGenerator),
        ("user_feature_generator", UserFeatureGenerator),
        ("gamma",                 Gamma),
    ]:
        obj = getattr(sc, attr, None)
        if not isinstance(obj, cls):
            issues.error(
                f"scenario.{attr} must be an instance of {cls.__name__}, got {type(obj).__name__}"
            )


def _check_auction_rule(auction: AuctionRule, issues: _Issues) -> None:
    bid, B_lo, B_hi = 1.0, 0.5, 2.0

    won, ok = _try(issues, "AuctionRule.win(bid > B)", auction.win, bid, B_lo)
    if ok:
        if not isinstance(won, (bool, np.bool_)):
            issues.error(f"AuctionRule.win must return bool, got {type(won).__name__}")
        elif not won:
            issues.warn("AuctionRule.win(1.0, 0.5) returned False — expected True for bid > competing_bid")

    won_lo, ok = _try(issues, "AuctionRule.win(bid < B)", auction.win, bid, B_hi)
    if ok and won_lo:
        issues.warn("AuctionRule.win(1.0, 2.0) returned True — expected False for bid < competing_bid")

    pay, ok = _try(issues, "AuctionRule.payment", auction.payment, bid, B_lo)
    if ok:
        if not isinstance(pay, (int, float, np.floating)):
            issues.error(f"AuctionRule.payment must return a numeric scalar, got {type(pay).__name__}")
        elif pay < 0:
            issues.error(f"AuctionRule.payment(1.0, 0.5) = {pay:.4g} < 0")
        elif pay > bid + 1e-9:
            issues.warn(f"AuctionRule.payment(1.0, 0.5) = {pay:.4g} > bid — unusual for standard auctions")


def _check_competition(comp: CompetitionDistribution, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(30):
        s, ok = _try(issues, "CompetitionDistribution.sample", comp.sample, rng)
        if ok:
            samples.append(s)
        if not ok:
            break
    if samples:
        if not all(isinstance(s, (int, float, np.floating)) for s in samples):
            issues.error("CompetitionDistribution.sample must return a scalar float")
        if any(s < 0 for s in samples):
            issues.warn(
                f"CompetitionDistribution.sample returned negative values (min={min(samples):.3g}) — "
                "bids are typically non-negative"
            )


def _check_arrival(arrival: ArrivalProcess, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    horizon = 10.0
    times, ok = _try(issues, "ArrivalProcess.sample_times", arrival.sample_times, horizon, rng)
    if not ok:
        return
    if not isinstance(times, np.ndarray):
        issues.error(f"ArrivalProcess.sample_times must return np.ndarray, got {type(times).__name__}")
        return
    if len(times) > 1 and not np.all(np.diff(times) >= 0):
        issues.error("ArrivalProcess.sample_times must return sorted times")
    if len(times) > 0 and (np.any(times < 0) or np.any(times > horizon + 1e-9)):
        issues.error(f"ArrivalProcess.sample_times returned times outside [0, {horizon}]")


def _check_horizon(hgen: HorizonGenerator, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        h, ok = _try(issues, "HorizonGenerator.sample", hgen.sample, rng)
        if not ok:
            break
        if not isinstance(h, (int, float, np.floating)):
            issues.error(f"HorizonGenerator.sample must return a numeric scalar, got {type(h).__name__}")
            break
        if h <= 0:
            issues.error(f"HorizonGenerator.sample must return a positive value, got {h:.4g}")


def _check_exogenous(exog: ExogenousProcess, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    _, ok = _try(issues, "ExogenousProcess.generate", exog.generate, 10.0, rng)
    if not ok:
        return
    y, ok = _try(issues, "ExogenousProcess.y", exog.y, 5.0)
    if ok and not isinstance(y, (int, float, np.floating)):
        issues.error(f"ExogenousProcess.y must return a scalar float, got {type(y).__name__}")


def _check_display_sampler(ds: DisplaySampler, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    if not hasattr(ds, "dim"):
        issues.error(f"{type(ds).__name__} must expose a .dim property")
        return
    dim = ds.dim
    if not isinstance(dim, int) or dim < 1:
        issues.error(f"DisplaySampler.dim must be a positive int, got {dim!r}")
        return
    x, ok = _try(issues, "DisplaySampler.sample", ds.sample, rng)
    if ok:
        if not isinstance(x, np.ndarray):
            issues.error(f"DisplaySampler.sample must return np.ndarray, got {type(x).__name__}")
        elif x.shape != (dim,):
            issues.error(f"DisplaySampler.sample shape mismatch: expected ({dim},), got {x.shape}")


def _check_user(ufg: UserFeatureGenerator, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    if not hasattr(ufg, "observable_dim"):
        issues.error(f"{type(ufg).__name__} must have an .observable_dim attribute")
        return
    obs_dim = ufg.observable_dim
    if not isinstance(obs_dim, int) or obs_dim < 1:
        issues.error(f"UserFeatureGenerator.observable_dim must be a positive int, got {obs_dim!r}")

    u, ok = _try(issues, "UserFeatureGenerator.generate", ufg.generate, rng)
    if not ok:
        return
    if not isinstance(u, np.ndarray):
        issues.error(f"UserFeatureGenerator.generate must return np.ndarray, got {type(u).__name__}")
        return
    obs, ok = _try(issues, "UserFeatureGenerator.get_observable_features", ufg.get_observable_features, u)
    if ok:
        if not isinstance(obs, np.ndarray):
            issues.error(
                f"UserFeatureGenerator.get_observable_features must return np.ndarray, got {type(obs).__name__}"
            )
        elif obs.shape != (obs_dim,):
            issues.error(
                f"UserFeatureGenerator.get_observable_features shape mismatch: "
                f"expected ({obs_dim},), got {obs.shape}"
            )


def _check_gamma(gamma: Gamma, sc: Scenario, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    ctx, ok = _try(issues, "Gamma.initiate_context", gamma.initiate_context)
    if not ok:
        return
    u = sc.user_feature_generator.generate(rng)
    x = sc.display_sampler.sample(rng)
    ctx2, ok = _try(issues, "Gamma.update_context", gamma.update_context, ctx, x, u, 1.0, [0.5])
    if ok and not isinstance(ctx2, (int, float, np.floating)):
        issues.error(f"Gamma.update_context must return a scalar float, got {type(ctx2).__name__}")
    ctx3, ok = _try(issues, "Gamma.current_context", gamma.current_context, ctx, 1.0, [0.5])
    if ok and not isinstance(ctx3, (int, float, np.floating)):
        issues.error(f"Gamma.current_context must return a scalar float, got {type(ctx3).__name__}")


def _check_kernel(kernel: RewardKernel, sc: Scenario, issues: _Issues) -> None:
    rng = np.random.default_rng(0)
    sc.exogenous.generate(10.0, rng)
    u = sc.user_feature_generator.generate(rng)
    x = sc.display_sampler.sample(rng)
    ctx = sc.gamma.initiate_context()
    y_func = sc.exogenous.y
    t, s, horizon = 0.0, 1.0, 10.0

    kap, ok = _try(issues, "RewardKernel.kappa", kernel.kappa, u, x, ctx, y_func(t))
    if ok and kap < 0:
        issues.error(f"RewardKernel.kappa returned {kap:.4g} < 0")

    lam, ok_lam = _try(issues, "RewardKernel.lambda_", kernel.lambda_, u, x, ctx, y_func(t + s), t, s)
    if ok_lam and lam < 0:
        issues.error(f"RewardKernel.lambda_ returned {lam:.4g} < 0")

    ub, ok_ub = _try(issues, "RewardKernel.lambda_upper_bound", kernel.lambda_upper_bound, u, x, ctx, t, horizon - t)
    if ok_ub and ub < 0:
        issues.error(f"RewardKernel.lambda_upper_bound returned {ub:.4g} < 0")
    if ok_lam and ok_ub and lam is not None and ub is not None and lam > ub + 1e-9:
        issues.error(
            f"RewardKernel.lambda_upper_bound ({ub:.4g}) < lambda_ ({lam:.4g}) — "
            "upper bound must dominate lambda_ for Ogata thinning to be correct"
        )

    er, ok = _try(issues, "RewardKernel.expected_reward", kernel.expected_reward, u, x, ctx, y_func, t, horizon)
    if ok and er < 0:
        issues.warn(f"RewardKernel.expected_reward returned {er:.4g} < 0 — unexpected for a conversion count")


def _check_probe_sampler(sc: Scenario, issues: _Issues) -> None:
    if sc.probe is None:
        return
    rng = np.random.default_rng(0)
    sc.exogenous.generate(10.0, rng)
    u = sc.user_feature_generator.generate(rng)
    x = sc.display_sampler.sample(rng)
    ctx = sc.gamma.initiate_context()
    t, horizon = 0.0, 10.0

    delays, ok = _try(
        issues, "DelaySampler.sample",
        sc.delay_sampler.sample,
        sc.probe, u, x, ctx, sc.exogenous.y, t, horizon, rng,
    )
    if not ok:
        return
    if not isinstance(delays, list):
        issues.error(f"DelaySampler.sample must return a list, got {type(delays).__name__}")
        return
    if any(d < 0 for d in delays):
        issues.error(f"DelaySampler.sample returned negative delays: {delays}")
    if any(t + d > horizon + 1e-9 for d in delays):
        issues.warn("DelaySampler.sample returned delays that exceed the remaining horizon")


# ── rollout integration check ───────────────────────────────────────────────────

def _check_obs(obs, env: AuctionEnv, issues: _Issues, context: str) -> None:
    if not isinstance(obs, dict):
        issues.error(f"Observation must be a dict, got {type(obs).__name__} [{context}]")
        return

    missing = set(OBSERVATION_KEYS) - set(obs.keys())
    extra = set(obs.keys()) - set(OBSERVATION_KEYS)
    if missing:
        issues.error(f"Observation missing keys {sorted(missing)} [{context}]")
    if extra:
        issues.warn(f"Observation has unexpected keys {sorted(extra)} [{context}]")

    if not env.observation_space.contains(obs):
        issues.warn(f"Observation not contained in observation_space [{context}]")


def _check_rollout(env: AuctionEnv, n_episodes: int, issues: _Issues) -> None:
    rng = np.random.default_rng(42)
    for ep in range(n_episodes):
        seed = int(rng.integers(1_000_000))
        try:
            obs, info = env.reset(seed=seed)
        except Exception as exc:
            issues.error(f"env.reset() raised {type(exc).__name__} at ep={ep}: {exc}")
            continue

        _check_obs(obs, env, issues, context=f"reset() ep={ep}")

        if not isinstance(info, dict):
            issues.error(f"reset() must return a dict as second element, got {type(info).__name__}")
        else:
            for key in ("horizon", "auction_times"):
                if key not in info:
                    issues.warn(f"reset() info missing expected key '{key}'")
            horizon = info.get("horizon")
            if horizon is not None and not (isinstance(horizon, (int, float)) and horizon > 0):
                issues.error(f"reset() info['horizon'] must be a positive number, got {horizon!r}")

        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            try:
                result = env.step(action)
            except Exception as exc:
                issues.error(f"env.step() raised {type(exc).__name__} at ep={ep} step={step}: {exc}")
                break

            if len(result) != 5:
                issues.error(
                    f"env.step() must return 5 values (obs, reward, done, truncated, info), "
                    f"got {len(result)} [{ep=} {step=}]"
                )
                break

            obs, reward, done, truncated, info = result

            _check_obs(obs, env, issues, context=f"step() ep={ep} step={step}")

            if not isinstance(reward, (int, float, np.floating)):
                issues.error(f"reward must be a scalar, got {type(reward).__name__} [{ep=} {step=}]")
            if not isinstance(done, (bool, np.bool_)):
                issues.error(f"done must be bool, got {type(done).__name__} [{ep=} {step=}]")
            if not isinstance(truncated, (bool, np.bool_)):
                issues.error(f"truncated must be bool, got {type(truncated).__name__} [{ep=} {step=}]")
            if not isinstance(info, dict):
                issues.error(f"info must be a dict, got {type(info).__name__} [{ep=} {step=}]")
            else:
                for key in ("auction", "display", "actual_conversions"):
                    if key not in info:
                        issues.warn(f"step() info missing expected key '{key}' [{ep=} {step=}]")
                auction = info.get("auction")
                if isinstance(auction, dict):
                    for akey in ("time", "bid", "won", "payment"):
                        if akey not in auction:
                            issues.warn(f"step() info['auction'] missing key '{akey}' [{ep=} {step=}]")

            step += 1


# ── public API ──────────────────────────────────────────────────────────────────

def check_env(env: AuctionEnv, *, n_episodes: int = 3, warn_only: bool = False) -> None:
    """
    Sanity-check an AuctionEnv instance.

    Runs structural checks on every Scenario component (types, shapes,
    numeric invariants) and a live rollout to verify the obs / reward /
    done / info contracts.

    Parameters
    ----------
    env        : AuctionEnv instance to check.
    n_episodes : Number of rollout episodes to simulate (default: 3).
    warn_only  : If True, emit warnings instead of raising on errors.

    Raises
    ------
    TypeError        if `env` is not an AuctionEnv.
    AssertionError   if errors are found and warn_only=False.
    """
    if not isinstance(env, AuctionEnv):
        raise TypeError(f"check_env expects an AuctionEnv, got {type(env).__name__}")

    issues = _Issues()
    sc = env.scenario

    _check_scenario_structure(sc, issues)

    if not issues._errors:
        _check_auction_rule(sc.auction, issues)
        _check_competition(sc.competition, issues)
        _check_arrival(sc.arrival, issues)
        _check_horizon(sc.horizon_generator, issues)
        _check_exogenous(sc.exogenous, issues)
        _check_display_sampler(sc.display_sampler, issues)
        _check_user(sc.user_feature_generator, issues)
        _check_gamma(sc.gamma, sc, issues)
        _check_kernel(sc.kernel, sc, issues)
        _check_probe_sampler(sc, issues)

    _check_rollout(env, n_episodes, issues)
    issues.report(warn_only)

    n_err = len(issues._errors)
    n_warn = len(issues._warnings)
    status = "OK" if n_err == 0 else f"FAILED ({n_err} error(s))"
    print(f"[check_env] {status} — {n_episodes} episode(s), {n_warn} warning(s).")
