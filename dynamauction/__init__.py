"""
dynamauction — Repeated auction environment with dynamic values.

Entry point: use ``dac.make(scenario_id, **overrides)`` to instantiate a
pre-built scenario, then ``gym.make("DynamAuction-v0", scenario=scenario)``
to get a Gymnasium environment.

    import dynamauction as dac
    scenario = dac.make("survival-second-price-exponential", horizon_eta=50)

The package also exposes all primitives (kernels, gamma models, auction rules,
…) for building custom scenarios via ``ScenarioBuilder``, as well as planning
baseline policies (FiPla, GreedyMyopic, Shading, MPC, …) and Gymnasium
wrappers for RL training.
"""
from importlib.metadata import version

from dynamauction.scenarios import (
    make,
    ScenarioBuilder,
    list_scenarios,
    get_scenario_info,
    describe_scenarios,
)
from dynamauction.wrappers import FlatAuctionObsWrapper, BidScaleWrapper, SB3PolicyWrapper, MaskExpectedRewardWrapper
from dynamauction.types import Scenario
from dynamauction.simulator import Simulator
from dynamauction.env import AuctionEnv, OBSERVATION_KEYS
from dynamauction.checker import check_env
from dynamauction.baseline.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy
from dynamauction.primitives import (
    # Auction rules
    FirstPriceAuction,
    SecondPriceAuction,
    # Competition distributions
    LogNormalCompetition,
    UniformCompetition,
    ExponentialCompetition,
    ConstantCompetition,
    # Arrival processes
    PiecewiseConstantPoisson,
    StepDecayPoisson,
    # Reward kernels
    SaturationKernel,
    RecencyDisplayKernel,
    CompetitorExtinctionKernel,
    MixedKernel,
    SurvivalKernel,
    EffectiveFrequencyKernel,
    ReminderKernel,
    # Gammas
    Gamma,
    AdditiveGamma,
    MultiplicativeGamma,
    LastWonTimeGamma,
    TimeSinceLastWinGamma,
    ConstantGamma,
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

__version__ = version("dynamauction")

__all__ = [
    # Checker
    "check_env",
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
    "ConstantCompetition",
    "PiecewiseConstantPoisson",
    "StepDecayPoisson",
    "SaturationKernel",
    "RecencyDisplayKernel",
    "CompetitorExtinctionKernel",
    "MixedKernel",
    "SurvivalKernel",
    "EffectiveFrequencyKernel",
    "ReminderKernel",
    "Gamma",
    "AdditiveGamma",
    "MultiplicativeGamma",
    "LastWonTimeGamma",
    "TimeSinceLastWinGamma",
    "ConstantGamma",
    "IIDUniformSampler",
    "CatalogueSampler",
    "FixedSampler",
    "TrivialProcess",
    "UserFeatureGenerator",
    "UniformUser",
    "AuctionEnv",
    "ConstantPolicy",
    "RandomPolicy",
    "FlatAuctionObsWrapper",
    "MaskExpectedRewardWrapper",
    "BidScaleWrapper",
    "SB3PolicyWrapper",
    "OBSERVATION_KEYS",
]

from gymnasium.envs.registration import register

register(
    id="DynamAuction-v0",
    entry_point="dynamauction.env:AuctionEnv",
    max_episode_steps=None, 
)