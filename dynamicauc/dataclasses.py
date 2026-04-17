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
    Gamma
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
    conversion_value: float = 1.0

@dataclass
class AuctionState:
    t: float
    ctx: float
    step_idx: int
    y_func: Callable[[float], float]
    display_features : np.ndarray
