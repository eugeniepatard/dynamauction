"""
Competing bid distributions modelling the auction market.

``CompetitionDistribution`` (abstract): samples the highest competing bid B.
Concrete implementations:
- ``UniformCompetition``: B ~ Uniform(low, high).
- ``LogNormalCompetition``: B ~ LogNormal(μ, σ).
- ``ExponentialCompetition``: B ~ Exponential(λ).
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
# TODO : rajouter constant 