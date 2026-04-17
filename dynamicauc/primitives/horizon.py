from abc import ABC, abstractmethod
import numpy as np


class HorizonGenerator(ABC):
    """Abstract horizon T generator."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float: ...


class FixedHorizon(HorizonGenerator):
    """Fixed horizon T."""

    def __init__(self, value: float):
        self.value = value

    def sample(self, rng: np.random.Generator) -> float:
        return self.value


class ExponentialHorizon(HorizonGenerator):
    """Horizon T sampled from an exponential distribution.
    Parameterized by mean eta, so that E[T] = eta."""

    def __init__(self, eta: float):
        self.eta = eta

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.eta))