"""
Baseline bidding policies for the repeated auction environment.

Simple non-learning baselines (``FiPlaPolicy``, ``ConstantPolicy``,
``RandomPolicy``) are defined in ``policies.py``.
Simulation-based planning policies (``GreedyMyopicPolicy``, ``ShadingPolicy``,
``MPCPolicy``, ``MPCOraclePolicy``, ``MPCTimeBinnedPolicy``) and the
``SimulationEstimator`` / ``MLPValuationHead`` are in ``baseline_algo.py``.
They are imported lazily to avoid pulling in ``torch`` on every
``import dynamauction``.
"""
from dynamauction.baseline.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy

__all__ = [
    "BiddingPolicy",
    "FiPlaPolicy",
    "ConstantPolicy",
    "RandomPolicy",
    "SimulationEstimator",
    "MLPValuationHead",
    "GreedyMyopicPolicy",
    "ShadingPolicy",
    "MPCOraclePolicy",
    "MPCPolicy",
    "MPCTimeBinnedPolicy",
]

_PLANNING = {
    "SimulationEstimator",
    "MLPValuationHead",
    "GreedyMyopicPolicy",
    "ShadingPolicy",
    "MPCOraclePolicy",
    "MPCPolicy",
    "MPCTimeBinnedPolicy",
}


def __getattr__(name: str):
    if name in _PLANNING:
        from dynamauction.baseline import baseline_algo as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'dynamauction.baseline' has no attribute {name!r}")
