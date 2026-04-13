from dataclasses import dataclass
from typing import Callable

import numpy as np
from dynamicauc.policy import BiddingPolicy
from dynamicauc.primitives import (
    AuctionRule,
    ArrivalProcess,
    CompetitionDistribution,
    DisplaySampler,
    ExogenousProcess,
    RewardKernel,
)

@dataclass
class Scenario:
    auction: AuctionRule
    competition: CompetitionDistribution
    arrival: ArrivalProcess
    kernel: RewardKernel
    display_sampler: DisplaySampler
    exogenous: ExogenousProcess
    policy: BiddingPolicy
    horizon: float
    user_features: np.ndarray

@dataclass
class AuctionState:
    t: float
    ctx: float
    step_idx: int
    y_func: Callable[[float], float]

