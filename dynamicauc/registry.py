"""
Scenario registry and factory for professional, discoverable scenario construction.

Examples
--------
    # Quick start with presets
    scenario = make("saturation-first-price-uniform")
    
    # Or use builder pattern
    scenario = (
        ScenarioBuilder()
        .with_kernel("saturation", base_rate=0.5)
        .with_auction("first-price")
        .with_competition("uniform", low=0.0, high=20.0)
        .with_arrival("poisson", breakpoints=[5.0], intensities=[0.1, 1.2])
        .with_display("catalogue", creatives=[[1.0, 0.0], [0.0, 1.0]])
        .with_exogenous("trivial")
        .with_policy("fipla")
        .with_horizon(10.0)
        .with_user_features([0.5, 0.8, 0.02])
        .build()
    )
    
    # List available configurations
    list_scenarios()
    get_scenario_info("saturation-first-price-uniform")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dynamicauc.dataclasses import Scenario
from dynamicauc.policy import BiddingPolicy, FiPlaPolicy
from dynamicauc.primitives import (
    AuctionRule,
    ArrivalProcess,
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
    # Arrival
    PiecewiseConstantPoisson,
    # Kernels
    SaturationKernel,
    RecencyDisplayKernel,
    CompetitorExtinctionKernel,
    MixedKernel,
    # Display samplers
    IIDUniformSampler,
    CatalogueSampler,
    FixedSampler,
    # Exogenous
    TrivialProcess,
)

# TODO : Créer un builder utilisant un fichier yaml

# ── Registry of presets ────────────────────────────────────────────────────────

SCENARIO_PRESETS = {
    "saturation-first-price-uniform": {
        "description": "Saturation kernel with first-price auction and uniform competition",
        "kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.1}},
        "auction": {"name": "first-price"},
        "competition": {"name": "uniform", "params": {"low": 0.0, "high": 1.0}},
        "arrival": {"name": "poisson", "params": {"breakpoints": [5.0], "intensities": [0.1, 1.2]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": 10.0,
        "user_features": [0.5, 0.8, 0.02],
    },
    "recency-second-price-lognormal": {
        "description": "Recency kernel with second-price auction and lognormal competition",
        "kernel": {"name": "recency", "params": {"alpha": 0.5, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}},
        "arrival": {"name": "poisson", "params": {"breakpoints": [], "intensities": [1.0]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": 30.0,
        "user_features": [0.5, 0.5, 0.5],
    },
    "mixed-first-price-exponential": {
        "description": "Mixed kernel (immediate + delayed) with first-price and exponential competition",
        "kernel": {
            "name": "mixed",
            "params": {
                "base_kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.1}},
                "kappa_base_rate": 0.2,
            },
        },
        "auction": {"name": "second-price"},
        "competition": {"name": "exponential", "params": {"scale": 1.0}},
        "arrival": {"name": "poisson", "params": {"breakpoints": [], "intensities": [0.5]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": 20.0,
        "user_features": [0.5, 0.3, 0.2],
    },
}


# ── Registry creators (primitive factories) ────────────────────────────────────

def create_kernel(name: str, **kwargs) -> RewardKernel:
    """Create a RewardKernel by name and parameters."""
    if name == "saturation":
        return SaturationKernel(**kwargs)
    elif name == "recency":
        return RecencyDisplayKernel(**kwargs)
    elif name == "competitor-extinction":
        return CompetitorExtinctionKernel(**kwargs)
    elif name == "mixed":
        base_kernel_cfg = kwargs.pop("base_kernel")
        base_kernel = create_kernel(base_kernel_cfg["name"], **base_kernel_cfg.get("params", {}))
        return MixedKernel(base_kernel=base_kernel, **kwargs)
    else:
        raise ValueError(f"Unknown kernel: {name}")


def create_auction(name: str, **kwargs) -> AuctionRule:
    """Create an AuctionRule by name and parameters."""
    if name == "first-price":
        return FirstPriceAuction(**kwargs)
    elif name == "second-price":
        return SecondPriceAuction(**kwargs)
    else:
        raise ValueError(f"Unknown auction: {name}")


def create_competition(name: str, **kwargs) -> CompetitionDistribution:
    """Create a CompetitionDistribution by name and parameters."""
    if name == "uniform":
        return UniformCompetition(**kwargs)
    elif name == "lognormal":
        return LogNormalCompetition(**kwargs)
    elif name == "exponential":
        return ExponentialCompetition(**kwargs)
    else:
        raise ValueError(f"Unknown competition: {name}")


def create_arrival(name: str, **kwargs) -> ArrivalProcess:
    """Create an ArrivalProcess by name and parameters."""
    if name == "poisson":
        return PiecewiseConstantPoisson(**kwargs)
    else:
        raise ValueError(f"Unknown arrival: {name}")


def create_display_sampler(name: str, **kwargs) -> DisplaySampler:
    """Create a DisplaySampler by name and parameters."""
    if name == "iid":
        return IIDUniformSampler(**kwargs)
    elif name == "catalogue":
        return CatalogueSampler(**kwargs)
    elif name == "fixed":
        return FixedSampler(**kwargs)
    else:
        raise ValueError(f"Unknown display sampler: {name}")


def create_exogenous(name: str, **kwargs) -> ExogenousProcess:
    """Create an ExogenousProcess by name and parameters."""
    if name == "trivial":
        return TrivialProcess(**kwargs)
    else:
        raise ValueError(f"Unknown exogenous: {name}")


def create_policy(name: str, **kwargs) -> BiddingPolicy:
    """Create a BiddingPolicy by name and parameters."""
    if name == "fipla":
        return FiPlaPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown policy: {name}")


# ── Scenario builder (fluent API) ──────────────────────────────────────────────

class ScenarioBuilder:
    """Fluent builder for constructing scenarios programmatically."""

    def __init__(self):
        self._kernel: Optional[RewardKernel] = None
        self._auction: Optional[AuctionRule] = None
        self._competition: Optional[CompetitionDistribution] = None
        self._arrival: Optional[ArrivalProcess] = None
        self._display_sampler: Optional[DisplaySampler] = None
        self._exogenous: Optional[ExogenousProcess] = None
        self._horizon: float = 10.0
        self._user_features: Optional[np.ndarray] = None

    def with_kernel(self, name: str, **params) -> ScenarioBuilder:
        """Set the reward kernel."""
        self._kernel = create_kernel(name, **params)
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

    def with_horizon(self, horizon: float) -> ScenarioBuilder:
        """Set the campaign horizon."""
        self._horizon = horizon
        return self

    def with_user_features(self, features: List[float]) -> ScenarioBuilder:
        """Set the user features."""
        self._user_features = np.array(features, dtype=np.float32)
        return self

    def build(self) -> Scenario:
        """Build the scenario."""
        # Apply defaults if not set
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
        if self._user_features is None:
            self._user_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        if self._horizon is None:
            self._horizon = 10.0

        return Scenario(
            auction=self._auction,
            competition=self._competition,
            arrival=self._arrival,
            kernel=self._kernel,
            display_sampler=self._display_sampler,
            exogenous=self._exogenous,
            horizon=self._horizon,
            user_features=self._user_features,
        )


# ── Factory functions ──────────────────────────────────────────────────────────

def make(scenario_id: str, **overrides) -> Scenario:
    """
    Make a scenario from a preset ID.

    Parameters
    ----------
    scenario_id : str
        Preset scenario identifier (e.g., "saturation-first-price-uniform")
    **overrides : dict
        Optional parameter overrides (e.g., horizon=20.0, user_features=[0.1, 0.2])

    Returns
    -------
    scenario : Scenario

    Examples
    --------
    >>> scenario = make("saturation-first-price-uniform")
    >>> scenario = make("saturation-first-price-uniform", horizon=20.0)
    """
    if scenario_id not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    # 1. Deep copy the preset to avoid side effects
    import copy
    preset = copy.deepcopy(SCENARIO_PRESETS[scenario_id])
    preset.pop("description", None)

    def set_nested_value(d: dict, path_parts: list, value: Any):
        """
        Surgically injects values into nested components.
        """
        for i in range(len(path_parts), 0, -1):
            candidate_key = "_".join(path_parts[:i])
            remaining_path = path_parts[i:]

            # Check if candidate_key exists in the current dictionary or its 'params'
            target = None
            in_params = False
            
            if candidate_key in d:
                target = d[candidate_key]
            elif "params" in d and candidate_key in d["params"]:
                target = d["params"][candidate_key]
                in_params = True

            if target is not None:
                if not remaining_path:
                    # BASE CASE: We found the actual parameter to set
                    if "params" not in d: d["params"] = {}
                    # If it was already in params, set it there, else set it in d
                    if in_params:
                        d["params"][candidate_key] = value
                    else:
                        # If it's a top level attr of the dict (not a sub-component)
                        d[candidate_key] = value
                    return
                
                # RECURSIVE STEP: We found a sub-dictionary (like base_kernel)
                if isinstance(target, dict):
                    return set_nested_value(target, remaining_path, value)

        # If no match found at all, it's a new parameter at the current level
        if "params" not in d: d["params"] = {}
        d["params"]["_".join(path_parts)] = value
    # -----------------------

    # 2. Process underscore overrides (e.g., kernel_base_kernel_saturation_scale)
    for key, value in list(overrides.items()):
        if "_" in key:
            parts = key.split("_")
            prefix = parts[0]
            if prefix in preset:
                # We pass everything after the prefix (e.g., ["base", "kernel", "saturation", "scale"])
                set_nested_value(preset[prefix], parts[1:], value)
                overrides.pop(key)

    # 3. Standard top-level overrides (horizon, user_features)
    if "horizon" in overrides:
        preset["horizon"] = overrides.pop("horizon")
    if "user_features" in overrides:
        preset["user_features"] = overrides.pop("user_features")

    # 4. Build the scenario using the updated preset dict
    builder = ScenarioBuilder()
    
    # Kernel (Mixed or otherwise) now receives the correctly nested 'base_kernel' dict
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
    
    builder.with_horizon(preset.get("horizon", 10.0))
    builder.with_user_features(preset.get("user_features", [0.5]*3))

    return builder.build()

def list_scenarios() -> List[str]:
    """Return list of available scenario preset IDs."""
    return list(SCENARIO_PRESETS.keys())


def get_scenario_info(scenario_id: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Get the full specification of a scenario preset.
    If verbose=True, prints a formatted tree of all parameters.
    """
    if scenario_id not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    
    preset = SCENARIO_PRESETS[scenario_id]

    if verbose:
        print(f"\nTECHNICAL SPECIFICATIONS: {scenario_id}")
        print("=" * 60)
        print(f"Description: {preset.get('description', 'N/A')}")
        print("-" * 60)

        def print_nested(d, indent=0):
            for key, value in d.items():
                if key == "description": continue
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_nested(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")

        # Start printing from the top level (excluding description)
        content = {k: v for k, v in preset.items() if k != 'description'}
        print_nested(content)
        print("=" * 60 + "\n")

    return preset


# ── Pretty printing ────────────────────────────────────────────────────────────

def describe_scenarios() -> str:
    """Return a concise list of IDs and descriptions."""
    lines = ["Available Scenarios", "=" * 60]
    for scenario_id in list_scenarios():
        # verbose=False prevents the full parameter tree from printing
        info = get_scenario_info(scenario_id, verbose=False)
        lines.append(f"{scenario_id:<35} | {info.get('description', 'N/A')}")
    
    lines.append("\n* Use get_scenario_info(scenario_id) for full technical specs.")
    return "\n".join(lines)