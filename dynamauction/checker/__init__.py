"""
Environment sanity checker.

Exposes ``check_env(scenario)`` which verifies component types, numeric
invariants, and runs a short live rollout to catch configuration errors
before a full training or evaluation run.
"""
from dynamauction.checker.checker import check_env

__all__ = ["check_env"]
