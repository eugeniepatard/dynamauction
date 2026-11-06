"""
Scenario registry and factory for professional, discoverable scenario construction.

Examples
--------
    # Quick start with presets
    scenario = make("saturation-first-price-uniform")

    # Or use builder pattern — call order does not matter
    scenario = (
        ScenarioBuilder()
        .with_kernel("saturation", base_rate=0.5)
        .with_auction("first-price")
        .with_competition("uniform", low=0.0, high=20.0)
        .with_arrival("pwpoisson", breakpoints=[5.0], intensities=[0.1, 1.2])
        .with_display("catalogue", creatives=[[1.0, 0.0], [0.0, 1.0]])
        .with_exogenous("trivial")
        .with_horizon("fixed", value=10.0)
        .with_user_features("uniform", full_dim=3, observable_dim=3)
        .with_probe("poisson")
        .with_delay_sampler("single")
        .build()
    )

    # List available configurations
    list_scenarios()
    get_scenario_info("saturation-first-price-uniform")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from dynamauction.types import Scenario
from dynamauction.scenarios.presets import SCENARIO_PRESETS
from dynamauction.baseline.policies import BiddingPolicy, FiPlaPolicy
from dynamauction.primitives import (
    AuctionRule,
    ArrivalProcess,
    UserFeatureGenerator,
    UniformUser,
    NormalUser,
    FixedUser,
    HorizonGenerator,
    FixedHorizon,
    ExponentialHorizon,
    CompetitionDistribution,
    DisplaySampler,
    ExogenousProcess,
    RewardKernel,
    # Auction rules
    FirstPriceAuction,
    SecondPriceAuction,
    # Competition
    LogNormalCompetition,
    UniformCompetition,
    ExponentialCompetition,
    ConstantCompetition,
    # Arrival
    PiecewiseConstantPoisson,
    ConstantPoisson,
    StepDecayPoisson,
    # Kernels
    SaturationKernel,
    RecencyDisplayKernel,
    CompetitorExtinctionKernel,
    MixedKernel,
    SurvivalKernel,
    ImmediateSurvivalKernel,
    EffectiveFrequencyKernel,
    ReminderKernel,
    # Gammas
    Gamma,
    AdditiveGamma,
    MultiplicativeGamma,
    RecoveringMultiplicativeGamma,
    LastWonTimeGamma,
    TimeSinceLastWinGamma,
    ConstantGamma,
    # Display samplers
    IIDUniformSampler,
    CatalogueSampler,
    FixedSampler,
    NormalSampler,
    # Exogenous
    TrivialProcess,
    CompetitorClickProcess,
    # Conversion probes
    ConversionProbe,
    PoissonProbe,
    BernoulliProbe,
    NullProbe,
    # Delay samplers
    DelaySampler,
    SingleConversionSampler,
    MultiConversionSampler,
)

# ── Registry creators (primitive factories) ────────────────────────────────────

def create_gamma(name: str, **kwargs) -> Gamma:
    """Create a Gamma by name and parameters."""
    if name == "additive":
        return AdditiveGamma(**kwargs)
    elif name == "multiplicative":
        return MultiplicativeGamma(**kwargs)
    elif name == "recovering-multiplicative":
        return RecoveringMultiplicativeGamma(**kwargs)
    elif name == "last-won-time":
        return LastWonTimeGamma(**kwargs)
    elif name == "time-since-last-win":
        return TimeSinceLastWinGamma(**kwargs)
    elif name == "constant":
        return ConstantGamma(**kwargs)
    else:
        raise ValueError(f"Unknown gamma: {name!r}")


def create_kernel(name: str, **kwargs) -> RewardKernel:
    """Create a RewardKernel by name and parameters."""
    if name == "saturation":
        return SaturationKernel(**kwargs)
    elif name == "recency":
        return RecencyDisplayKernel(**kwargs)
    elif name == "competitor-extinction":
        return CompetitorExtinctionKernel(**kwargs)
    elif name == "survival":
        return SurvivalKernel(**kwargs)
    elif name == "immediate-survival":
        return ImmediateSurvivalKernel(**kwargs)
    elif name == "effective-frequency":
        return EffectiveFrequencyKernel(**kwargs)
    elif name == "reminder":
        return ReminderKernel(**kwargs)
    elif name == "mixed":
        base_kernel_cfg = kwargs.pop("base_kernel")
        base_kernel = create_kernel(base_kernel_cfg["name"], **base_kernel_cfg.get("params", {}))
        return MixedKernel(base_kernel=base_kernel, **kwargs)
    else:
        raise ValueError(f"Unknown kernel: {name!r}")


def create_auction(name: str, **kwargs) -> AuctionRule:
    """Create an AuctionRule by name and parameters."""
    if name == "first-price":
        return FirstPriceAuction(**kwargs)
    elif name == "second-price":
        return SecondPriceAuction(**kwargs)
    else:
        raise ValueError(f"Unknown auction: {name!r}")


def create_competition(name: str, **kwargs) -> CompetitionDistribution:
    """Create a CompetitionDistribution by name and parameters."""
    if name == "uniform":
        return UniformCompetition(**kwargs)
    elif name == "lognormal":
        return LogNormalCompetition(**kwargs)
    elif name == "exponential":
        return ExponentialCompetition(**kwargs)
    elif name == "constant":
        return ConstantCompetition(**kwargs)
    else:
        raise ValueError(f"Unknown competition: {name!r}")


def create_arrival(name: str, **kwargs) -> ArrivalProcess:
    """Create an ArrivalProcess by name and parameters."""
    if name == "pwpoisson":
        return PiecewiseConstantPoisson(**kwargs)
    elif name == "cstpoisson":
        return ConstantPoisson(**kwargs)
    elif name == "decaypoisson":
        return StepDecayPoisson(**kwargs)
    else:
        raise ValueError(f"Unknown arrival: {name!r}")


def create_display_sampler(name: str, **kwargs) -> DisplaySampler:
    """Create a DisplaySampler by name and parameters."""
    if name == "iid":
        return IIDUniformSampler(**kwargs)
    elif name == "catalogue":
        return CatalogueSampler(**kwargs)
    elif name == "fixed":
        return FixedSampler(**kwargs)
    elif name == "normal":
        return NormalSampler(**kwargs)
    else:
        raise ValueError(f"Unknown display sampler: {name!r}")


def create_exogenous(name: str, **kwargs) -> ExogenousProcess:
    """Create an ExogenousProcess by name and parameters."""
    if name == "trivial":
        return TrivialProcess(**kwargs)
    elif name == "competitor-click":
        return CompetitorClickProcess(**kwargs)
    else:
        raise ValueError(f"Unknown exogenous: {name!r}")


def create_policy(name: str, **kwargs) -> BiddingPolicy:
    """Create a BiddingPolicy by name and parameters."""
    if name == "fipla":
        return FiPlaPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown policy: {name!r}")


def create_user_feature_generator(name: str, **kwargs) -> UserFeatureGenerator:
    """Create a UserFeatureGenerator by name and parameters."""
    if name == "uniform":
        return UniformUser(**kwargs)
    elif name == "normal":
        return NormalUser(**kwargs)
    elif name == "fixed":
        return FixedUser(**kwargs)
    else:
        raise ValueError(f"Unknown user feature generator: {name!r}")


def create_horizon_generator(name: str, **kwargs) -> HorizonGenerator:
    """Create a HorizonGenerator by name and parameters."""
    if name == "fixed":
        return FixedHorizon(**kwargs)
    elif name == "exponential":
        return ExponentialHorizon(**kwargs)
    else:
        raise ValueError(f"Unknown horizon generator: {name!r}")


def create_probe(name: str, kernel: RewardKernel, **kwargs) -> ConversionProbe:
    """
    Create a ConversionProbe by name, wrapping a fully-constructed kernel.

    Always call this *after* the kernel has been finalised (i.e. inside build()).
    """
    if name == "poisson":
        return PoissonProbe(kernel=kernel, **kwargs)
    elif name == "bernoulli":
        return BernoulliProbe(kernel=kernel, **kwargs)
    elif name == "null":
        return NullProbe(kernel=kernel, **kwargs)
    else:
        raise ValueError(f"Unknown probe: {name!r}")


def create_delay_sampler(name: str, **kwargs) -> DelaySampler:
    """
    Create a DelaySampler by name.

    Note: SingleConversionSampler and MultiConversionSampler receive the probe
    at runtime via sample(), not at construction time, so no probe is passed here.
    """
    if name == "single":
        return SingleConversionSampler(**kwargs)
    elif name == "multi":
        return MultiConversionSampler(**kwargs)
    else:
        raise ValueError(f"Unknown delay sampler: {name!r}")


# ── Scenario builder (fluent API) ──────────────────────────────────────────────

class ScenarioBuilder:
    """
    Fluent builder for constructing Scenario objects programmatically.

    Call order is irrelevant: probe and delay_sampler configurations are stored
    as (name, params) tuples and resolved inside build() after all components
    (especially the kernel) have been finalised and defaults applied.
    """

    def __init__(self):
        self._kernel: Optional[RewardKernel] = None
        self._auction: Optional[AuctionRule] = None
        self._competition: Optional[CompetitionDistribution] = None
        self._arrival: Optional[ArrivalProcess] = None
        self._display_sampler: Optional[DisplaySampler] = None
        self._exogenous: Optional[ExogenousProcess] = None
        self._horizon_generator: Optional[HorizonGenerator] = None
        self._user_feature_generator: Optional[UserFeatureGenerator] = None
        self._gamma: Optional[Gamma] = None
        self._max_history: int = 10
        self._kappa_distribution: str = 'poisson'
        # Deferred: resolved in build() so call order never matters
        self._probe_cfg: Optional[Tuple[str, dict]] = None
        self._delay_sampler_cfg: Optional[Tuple[str, dict]] = None

    # ── Component setters ──────────────────────────────────────────────────────

    def with_kernel(self, name: str, **params) -> ScenarioBuilder:
        """Set the reward kernel."""
        self._kernel = create_kernel(name, **params)
        return self

    def with_gamma(self, name: str, **params) -> ScenarioBuilder:
        """Set the gamma function."""
        self._gamma = create_gamma(name, **params)
        return self

    def with_auction(self, name: str, **params) -> ScenarioBuilder:
        """Set the auction rule."""
        self._auction = create_auction(name, **params)
        return self

    def with_competition(self, name: str, **params) -> ScenarioBuilder:
        """Set the competition distribution."""
        self._competition = create_competition(name, **params)
        return self

    def with_arrival(self, name: str, **params) -> ScenarioBuilder:
        """Set the arrival process."""
        self._arrival = create_arrival(name, **params)
        return self

    def with_display(self, name: str, **params) -> ScenarioBuilder:
        """Set the display sampler."""
        self._display_sampler = create_display_sampler(name, **params)
        return self

    def with_exogenous(self, name: str, **params) -> ScenarioBuilder:
        """Set the exogenous process."""
        self._exogenous = create_exogenous(name, **params)
        return self

    def with_horizon(self, name: str, **params) -> ScenarioBuilder:
        """Set the campaign horizon generator."""
        self._horizon_generator = create_horizon_generator(name, **params)
        return self

    def with_user_features(self, name: str, **params) -> ScenarioBuilder:
        """Set the user feature generator."""
        self._user_feature_generator = create_user_feature_generator(name, **params)
        return self

    def with_max_history(self, max_history: int) -> ScenarioBuilder:
        """Set the maximum conversion history capacity for observations."""
        self._max_history = max_history
        return self

    def with_kappa_distribution(self, distribution: str) -> ScenarioBuilder:
        """Set how immediate conversions (kappa) are sampled: 'poisson' or 'bernoulli'."""
        if distribution not in ('poisson', 'bernoulli'):
            raise ValueError(f"kappa_distribution must be 'poisson' or 'bernoulli', got {distribution!r}")
        self._kappa_distribution = distribution
        return self

    def with_probe(self, name: str, **params) -> ScenarioBuilder:
        """
        Set the conversion probe by name.
        """
        self._probe_cfg = (name, params)
        return self

    def with_delay_sampler(self, name: str, **params) -> ScenarioBuilder:
        """
        Set the delay sampler by name ('single' or 'multi').
        """
        self._delay_sampler_cfg = (name, params)
        return self

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self) -> Scenario:
        """
        Finalise and return the Scenario.

        Resolution order inside build():
          1. All primitive components (kernel first, then the rest).
          2. Probe  — needs the finalised kernel.
          3. DelaySampler — independent of probe at construction time.
        """
        if self._kernel is None:
            self._kernel = SaturationKernel()
        if self._auction is None:
            self._auction = SecondPriceAuction()
        if self._competition is None:
            self._competition = UniformCompetition()
        if self._arrival is None:
            self._arrival = PiecewiseConstantPoisson(breakpoints=[], intensities=[1.0])
        if self._display_sampler is None:
            self._display_sampler = IIDUniformSampler(dim=3)
        if self._exogenous is None:
            self._exogenous = TrivialProcess()
        if self._horizon_generator is None:
            self._horizon_generator = FixedHorizon(value=10.0)
        if self._user_feature_generator is None:
            self._user_feature_generator = UniformUser(full_dim=3, observable_dim=3)
        if self._gamma is None:
            self._gamma = AdditiveGamma()
        if self._max_history is None:
            self._max_history = 10

        if self._probe_cfg is not None:
            probe_name, probe_params = self._probe_cfg
            probe: ConversionProbe = create_probe(probe_name, self._kernel, **probe_params)
        else:
            probe = PoissonProbe(kernel=self._kernel)

        if self._delay_sampler_cfg is not None:
            ds_name, ds_params = self._delay_sampler_cfg
            delay_sampler: DelaySampler = create_delay_sampler(ds_name, **ds_params)
        else:
            delay_sampler = SingleConversionSampler()

        return Scenario(
            auction=self._auction,
            competition=self._competition,
            arrival=self._arrival,
            kernel=self._kernel,
            display_sampler=self._display_sampler,
            exogenous=self._exogenous,
            horizon_generator=self._horizon_generator,
            user_feature_generator=self._user_feature_generator,
            gamma=self._gamma,
            max_history=self._max_history,
            probe=probe,
            delay_sampler=delay_sampler,
            kappa_distribution=self._kappa_distribution,
        )


# ── Factory functions ──────────────────────────────────────────────────────────

def make(scenario_id: str, **overrides) -> Scenario:
    """
    Make a scenario from a preset ID.

    Parameters
    ----------
    scenario_id : str
        Preset scenario identifier (e.g., "saturation-first-price-uniform").
    **overrides :
        Optional parameter overrides using underscore-separated paths,
        e.g. ``kernel_base_rate=0.5``, ``horizon_value=20.0``, ``max_history=5``.

    Returns
    -------
    Scenario

    Examples
    --------
    >>> scenario = make("saturation-first-price-uniform")
    >>> scenario = make("saturation-first-price-uniform", horizon_value=20.0)
    """
    if scenario_id not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario: {scenario_id!r}")

    preset = copy.deepcopy(SCENARIO_PRESETS[scenario_id])
    preset.pop("description", None)

    # ── Nested override helper ─────────────────────────────────────────────────

    def set_nested_value(d: dict, path_parts: list, value: Any) -> None:
        """
        Surgically inject a value into a nested preset dict.

        Tries to match the longest prefix of path_parts to an existing key
        (at top level or inside 'params'), then recurses into sub-dicts.
        Falls back to writing an unknown key into 'params'.
        """
        for i in range(len(path_parts), 0, -1):
            candidate_key = "_".join(path_parts[:i])
            remaining_path = path_parts[i:]

            target = None
            in_params = False

            if candidate_key in d:
                target = d[candidate_key]
            elif "params" in d and candidate_key in d["params"]:
                target = d["params"][candidate_key]
                in_params = True

            if target is not None:
                if not remaining_path:
                    # Base case: set the value
                    if "params" not in d:
                        d["params"] = {}
                    if in_params:
                        d["params"][candidate_key] = value
                    else:
                        d[candidate_key] = value
                    return
                # Recursive case: descend into a sub-dict
                if isinstance(target, dict):
                    set_nested_value(target, remaining_path, value)
                    return

        # No match: add as a new param
        if "params" not in d:
            d["params"] = {}
        d["params"]["_".join(path_parts)] = value

    # ── Apply overrides ────────────────────────────────────────────────────────

    for key, value in list(overrides.items()):
        if "_" in key:
            parts = key.split("_")
            prefix = parts[0]
            if prefix in preset:
                set_nested_value(preset[prefix], parts[1:], value)
                del overrides[key]

    if "max_history" in overrides:
        preset["max_history"] = overrides.pop("max_history")

    if "kappa_distribution" in overrides:
        preset["kappa_distribution"] = overrides.pop("kappa_distribution")

    # ── Assemble via builder ───────────────────────────────────────────────────

    builder = ScenarioBuilder()

    k_cfg = preset.pop("kernel")
    builder.with_kernel(k_cfg["name"], **k_cfg.get("params", {}))

    a_cfg = preset.pop("auction")
    builder.with_auction(a_cfg["name"], **a_cfg.get("params", {}))

    c_cfg = preset.pop("competition")
    builder.with_competition(c_cfg["name"], **c_cfg.get("params", {}))

    arr_cfg = preset.pop("arrival")
    builder.with_arrival(arr_cfg["name"], **arr_cfg.get("params", {}))

    disp_cfg = preset.pop("display")
    builder.with_display(disp_cfg["name"], **disp_cfg.get("params", {}))

    exo_cfg = preset.pop("exogenous")
    builder.with_exogenous(exo_cfg["name"], **exo_cfg.get("params", {}))

    g_cfg = preset.pop("gamma", {"name": "additive"})
    builder.with_gamma(g_cfg["name"], **g_cfg.get("params", {}))

    user_cfg = preset.pop("user")
    builder.with_user_features(user_cfg["name"], **user_cfg.get("params", {}))

    h_cfg = preset.pop("horizon")
    builder.with_horizon(h_cfg["name"], **h_cfg.get("params", {}))

    builder.with_max_history(preset.get("max_history", 10))
    builder.with_kappa_distribution(preset.get("kappa_distribution", "poisson"))

    # Probe and delay sampler are optional in presets
    if "probe" in preset:
        p_cfg = preset.pop("probe")
        builder.with_probe(p_cfg["name"], **p_cfg.get("params", {}))

    if "delay_sampler" in preset:
        ds_cfg = preset.pop("delay_sampler")
        builder.with_delay_sampler(ds_cfg["name"], **ds_cfg.get("params", {}))

    return builder.build()


# ── Inspection helpers ─────────────────────────────────────────────────────────

def list_scenarios() -> List[str]:
    """Return list of available scenario preset IDs."""
    return list(SCENARIO_PRESETS.keys())


def get_scenario_info(scenario_id: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Get the full specification of a scenario preset.

    Parameters
    ----------
    scenario_id : str
    verbose : bool
        If True, prints a formatted parameter tree to stdout.
    """
    if scenario_id not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario: {scenario_id!r}")

    preset = SCENARIO_PRESETS[scenario_id]

    if verbose:
        print(f"\nTECHNICAL SPECIFICATIONS: {scenario_id}")
        print("=" * 60)
        print(f"Description: {preset.get('description', 'N/A')}")
        print("-" * 60)

        def print_nested(d: dict, indent: int = 0) -> None:
            for key, value in d.items():
                if key == "description":
                    continue
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_nested(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")

        print_nested({k: v for k, v in preset.items() if k != "description"})
        print("=" * 60 + "\n")

    return preset


def describe_scenarios() -> str:
    """Return a concise table of IDs and descriptions."""
    lines = ["Available Scenarios", "=" * 60]
    for scenario_id in list_scenarios():
        info = get_scenario_info(scenario_id, verbose=False)
        lines.append(f"{scenario_id:<35} | {info.get('description', 'N/A')}")
    lines.append("\n* Use get_scenario_info(scenario_id) for full technical specs.")
    return "\n".join(lines)