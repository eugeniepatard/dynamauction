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