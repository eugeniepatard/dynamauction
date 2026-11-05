"""
Baseline bidding policies for the repeated auction environment.

Simple non-learning baselines (``FiPlaPolicy``, ``ConstantPolicy``,
``RandomPolicy``) are defined in ``policies.py``.
Simulation-based planning policies (``GreedyMyopicPolicy``, ``ShadingPolicy``,
``MPCPolicy``, ``MPCOraclePolicy``, ``MPCTimeBinnedPolicy``) and the
``SimulationEstimator`` / ``MLPValuationHead`` are in ``baseline_algo.py`` (``OracleV0`` in ``estimators.py``).
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
    "OracleV0",
    "GreedyMyopicPolicy",
    "ShadingPolicy",
    "MPCOraclePolicy",
    "MPCPolicy",
    "MPCTimeBinnedPolicy",
]

def __getattr__(name: str):
    # Called only for names not already in this module's namespace (i.e. not
    # the eagerly imported policies above). Loads baseline_algo — and torch —
    # on demand for planning classes listed in __all__.
    if name == "OracleV0":
        from dynamauction.baseline.estimators import OracleV0
        return OracleV0
    if name in __all__:
        from dynamauction.baseline import baseline_algo as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'dynamauction.baseline' has no attribute {name!r}")
