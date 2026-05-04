"""
User feature generators sampling the fixed user vector u once per episode.

``UserFeatureGenerator`` (abstract): samples u ∈ Rᵈ at episode start.
Only the first d_obs ≤ d components are included in the agent's observation,
modelling partial observability of user attributes.
``UniformUser``: each component drawn i.i.d. from Uniform(0, 1).
"""
from abc import ABC, abstractmethod
import numpy as np


class UserFeatureGenerator(ABC):
    """Abstract user feature generator."""
    @abstractmethod
    def generate(self, rng: np.random.Generator) -> np.ndarray: ...

    @property
    @abstractmethod
    def get_observable_features(self, full_features: np.ndarray) -> np.ndarray:
        """Extracts the observable part of the user features."""
        ...


class UniformUser(UserFeatureGenerator):
    """
    Generates user features from a uniform distribution in [0, 1]^d.
    The first `observable_dim` features are observable, the rest are hidden.
    """
    def __init__(self, full_dim: int, observable_dim: int):
        if observable_dim > full_dim:
            raise ValueError("observable_dim cannot be greater than full_dim")
        self.full_dim = full_dim
        self.observable_dim = observable_dim

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0, 1, self.full_dim).astype(np.float32)

    def get_observable_features(self, full_features: np.ndarray) -> np.ndarray:
        return full_features[:self.observable_dim]

class NormalUser(UserFeatureGenerator):
    """
    Generates user features from a normal distribution.
    The first `observable_dim` features are observable, the rest are hidden.
    """
    def __init__(self, full_dim: int, observable_dim: int, mu: float = 0.0, sigma: float = 1.0):
        if observable_dim > full_dim:
            raise ValueError("observable_dim cannot be greater than full_dim")
        self.full_dim = full_dim
        self.observable_dim = observable_dim
        self.mu = mu
        self.sigma = sigma

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, self.full_dim).astype(np.float32)

    def get_observable_features(self, full_features: np.ndarray) -> np.ndarray:
        return full_features[:self.observable_dim]

class FixedUser(UserFeatureGenerator):
    """Always return the same user feature vector."""
    def __init__(self, features: np.ndarray, observable_dim: int):
        if observable_dim > len(features):
            raise ValueError("observable_dim cannot be greater than the length of features")
        self.features = features.astype(np.float32)
        self.observable_dim = observable_dim

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        return self.features.copy()

    def get_observable_features(self, full_features: np.ndarray) -> np.ndarray:
        return full_features[:self.observable_dim]