"""
Benchmark package for the repeated-auction-dynamic-value library.

Typical workflow
----------------
1. Train::

    from benchmark.train_rl import train_all, TrainingConfig
    train_all(["ppo", "sac", "td3"], TrainingConfig(total_timesteps=500_000))

2. Evaluate::

    from benchmark.evaluate import run_full_benchmark, EvalConfig, score_matrix
    df     = run_full_benchmark(cfg=EvalConfig(seeds=list(range(30))))
    scores = score_matrix(df)

3. Plot::

    from benchmark import plots
    plots.plot_full_benchmark(df, scores)
    plots.plot_training_curves()
"""

from .train_rl import (
    train,
    train_all,
    load_model,
    TrainingConfig,
    SCENARIO_ID,
    MODELS_DIR,
    LOGS_DIR,
)

from .evaluate import (
    benchmark,
    run_full_benchmark,
    evaluate_policy,
    get_rl_policies,
    get_planning_policies,
    episode_summary,
    score_matrix,
    SB3PolicyWrapper,
    EvalConfig,
    RESULTS_DIR,
)

from dynamicauc import FlatAuctionObsWrapper, BidScaleWrapper
from . import plots

__all__ = [
    # Training
    "train", "train_all", "load_model",
    "TrainingConfig", "FlatAuctionObsWrapper", "BidScaleWrapper",
    "SCENARIO_ID", "MODELS_DIR", "LOGS_DIR",
    # Evaluation
    "benchmark", "run_full_benchmark", "evaluate_policy",
    "get_rl_policies", "get_planning_policies",
    "episode_summary", "score_matrix",
    "SB3PolicyWrapper", "EvalConfig", "RESULTS_DIR",
    # Wrappers (live in dynamicauc, re-exported here for convenience)
    "FlatAuctionObsWrapper",
    "BidScaleWrapper",
    # Plots module
    "plots",
]
