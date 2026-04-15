from importlib.metadata import version

from dynamicauc.registry import (
    make,
    ScenarioBuilder,
    list_scenarios,
    get_scenario_info,
    describe_scenarios,
)
from dynamicauc.dataclasses import Scenario
from dynamicauc.simulator import Simulator
from dynamicauc.env import AuctionEnv
from dynamicauc.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy
from dynamicauc.primitives import (
    # Auction rules
    FirstPriceAuction,
    SecondPriceAuction,
    # Competition distributions
    LogNormalCompetition,
    UniformCompetition,
    ExponentialCompetition,
    # Arrival processes
    PiecewiseConstantPoisson,
    # Reward kernels
    SaturationKernel,
    RecencyDisplayKernel,
    CompetitorExtinctionKernel,
    MixedKernel,
    # Display samplers
    IIDUniformSampler,
    CatalogueSampler,
    FixedSampler,
    # Exogenous processes
    TrivialProcess,
    # User
    UserFeatureGenerator,
    UniformUser,
)

__version__ = version("dynamicauc")

__all__ = [
    # Main factory API
    "make",
    "ScenarioBuilder",
    "list_scenarios",
    "get_scenario_info",
    "describe_scenarios",
    # Core classes
    "Scenario",
    "Simulator",
    "BiddingPolicy",
    "FiPlaPolicy",
    # Primitives
    "FirstPriceAuction",
    "SecondPriceAuction",
    "LogNormalCompetition",
    "UniformCompetition",
    "ExponentialCompetition",
    "PiecewiseConstantPoisson",
    "SaturationKernel",
    "RecencyDisplayKernel",
    "CompetitorExtinctionKernel",
    "MixedKernel",
    "IIDUniformSampler",
    "CatalogueSampler",
    "FixedSampler",
    "TrivialProcess",
    "UserFeatureGenerator",
    "UniformUser",
]

from gymnasium.envs.registration import register

register(
    id="DynamicAuction-v0",
    entry_point="dynamicauc.env:AuctionEnv",
    max_episode_steps=None, 
)