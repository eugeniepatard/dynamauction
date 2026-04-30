"""
dynamicauc — Repeated auction environment with dynamic values.

Entry point: use ``dac.make(scenario_id, **overrides)`` to instantiate a
pre-built scenario, then ``gym.make("DynamicAuction-v0", scenario=scenario)``
to get a Gymnasium environment.

    import dynamicauc as dac
    scenario = dac.make("survival-second-price-exponential", horizon_eta=50)

The package also exposes all primitives (kernels, gamma models, auction rules,
…) for building custom scenarios via ``ScenarioBuilder``, as well as planning
baseline policies (FiPla, GreedyMyopic, Shading, MPC, …) and Gymnasium
wrappers for RL training.
"""
from importlib.metadata import version

from dynamicauc.scenarios import (
    make,
    ScenarioBuilder,
    list_scenarios,
    get_scenario_info,
    describe_scenarios,
)
from dynamicauc.wrappers import FlatAuctionObsWrapper, BidScaleWrapper, SB3PolicyWrapper, MaskExpectedRewardWrapper
from dynamicauc.dataclasses import Scenario
from dynamicauc.simulator import Simulator
from dynamicauc.env import AuctionEnv, OBSERVATION_KEYS
from dynamicauc.checker import check_env
from dynamicauc.baseline.policies import BiddingPolicy, FiPlaPolicy, ConstantPolicy, RandomPolicy
from dynamicauc.baseline.baseline_algo import (
    SimulationEstimator,
    MLPValuationHead,
    GreedyMyopicPolicy,
    ShadingPolicy,
    MPCOraclePolicy,
    MPCPolicy,
)
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
    SurvivalKernel,
    # Gammas
    Gamma,
    AdditiveGamma,
    MultiplicativeGamma,
    LastWonTimeGamma,
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

__version__ = version("dynamicauc")

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
    "PiecewiseConstantPoisson",
    "SaturationKernel",
    "RecencyDisplayKernel",
    "CompetitorExtinctionKernel",
    "MixedKernel",
    "SurvivalKernel",
    "Gamma",
    "AdditiveGamma",
    "MultiplicativeGamma",
    "LastWonTimeGamma",
    "ConstantGamma",
    "IIDUniformSampler",
    "CatalogueSampler",
    "FixedSampler",
    "TrivialProcess",
    "UserFeatureGenerator",
    "UniformUser",
    AuctionEnv,
    ConstantPolicy,
    RandomPolicy,
    FlatAuctionObsWrapper,
    MaskExpectedRewardWrapper,
    BidScaleWrapper,
    SB3PolicyWrapper,
    OBSERVATION_KEYS,
    "SimulationEstimator",
    "MLPValuationHead",
    "GreedyMyopicPolicy",
    "ShadingPolicy",
    "MPCOraclePolicy",
    "MPCPolicy",
]

from gymnasium.envs.registration import register

register(
    id="DynamicAuction-v0",
    entry_point="dynamicauc.env:AuctionEnv",
    max_episode_steps=None, 
)