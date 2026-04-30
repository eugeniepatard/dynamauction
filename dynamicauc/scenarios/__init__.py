"""
Scenario registry and factory.

Use ``make(scenario_id, **overrides)`` to instantiate a built-in scenario,
``list_scenarios()`` to enumerate available presets, and ``ScenarioBuilder``
to compose a fully custom scenario from primitives.
"""
from .presets import SCENARIO_PRESETS
from .builder import (
    make,
    ScenarioBuilder,
    list_scenarios,
    get_scenario_info,
    describe_scenarios,
)

__all__ = [
    "SCENARIO_PRESETS",
    "make",
    "ScenarioBuilder",
    "list_scenarios",
    "get_scenario_info",
    "describe_scenarios",
]