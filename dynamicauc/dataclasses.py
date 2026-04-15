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
    UserFeatureGenerator,
)

# TODO : horizon => classe / horizon = nombre aléatoire loi exp / ou constant

@dataclass
class Scenario:
    auction: AuctionRule
    competition: CompetitionDistribution
    arrival: ArrivalProcess
    kernel: RewardKernel
    display_sampler: DisplaySampler
    exogenous: ExogenousProcess
    horizon: float
    user_feature_generator: UserFeatureGenerator

@dataclass
class AuctionState:
    t: float
    ctx: float
    step_idx: int
    y_func: Callable[[float], float]
    display_features : np.ndarray
