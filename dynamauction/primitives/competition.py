"""
Competing bid distributions modelling the auction market.

``CompetitionDistribution`` (abstract): samples the highest competing bid B.
Concrete implementations:
- ``UniformCompetition``: B ~ Uniform(low, high).
- ``LogNormalCompetition``: B ~ LogNormal(μ, σ).
- ``ExponentialCompetition``: B ~ Exponential(λ).
- ``ConstantCompetition``: B = value (deterministic).
"""
from abc import ABC, abstractmethod
import numpy as np


class CompetitionDistribution(ABC):
    """Distribution F of the highest competing bid B at each auction."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float: ...


class LogNormalCompetition(CompetitionDistribution):
    def __init__(self, mu: float = 0.0, sigma: float = 0.5):
        self.mu, self.sigma = mu, sigma

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))


class UniformCompetition(CompetitionDistribution):
    def __init__(self, low: float = 0.0, high: float = 2.0):
        self.low, self.high = low, high

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


class ExponentialCompetition(CompetitionDistribution):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.scale))


class ConstantCompetition(CompetitionDistribution):
    """Deterministic competing bid: B = value at every auction.

    Useful as a controlled baseline or for unit-testing auction mechanisms
    where the competing bid should be fixed and known in advance.
    """

    def __init__(self, value: float = 1.0):
        if value < 0:
            raise ValueError(f"Competing bid value must be non-negative, got {value}.")
        self.value = value

    def sample(self, _rng: np.random.Generator) -> float:
        return self.value