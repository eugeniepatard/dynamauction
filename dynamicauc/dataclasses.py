"""
Core dataclasses shared across the package.

``Scenario`` bundles all environment primitives (auction rule, competition
distribution, kernel, gamma, horizon generator, …) into a single object
passed to ``AuctionEnv`` and ``Simulator``.
``AuctionState`` holds the mutable per-episode state updated at each step.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from dynamicauc.primitives import (
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

@dataclass
class Scenario:
    auction: AuctionRule
    competition: CompetitionDistribution
    arrival: ArrivalProcess
    kernel: RewardKernel
    display_sampler: DisplaySampler
    exogenous: ExogenousProcess
    horizon_generator: HorizonGenerator
    user_feature_generator: UserFeatureGenerator
    gamma: Gamma
    max_history: int
    probe: ConversionProbe = None
    delay_sampler: DelaySampler = None
    conversion_value: float = 1.0

@dataclass
class AuctionState:
    t: float
    ctx: float
    step_idx: int
    y_func: Callable[[float], float]
    display_features : np.ndarray
