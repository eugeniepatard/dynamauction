# ── Display feature samplers ───────────────────────────────────────────────────

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class DisplaySampler(ABC):
    """Generates the display feature vector x_i at each auction opportunity."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray: ...


class IIDUniformSampler(DisplaySampler):
    """x_i ~ Uniform([0, 1]^dim) i.i.d. at each auction."""

    def __init__(self, dim: int = 3):
        self.dim = dim

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, self.dim)


class CatalogueSampler(DisplaySampler):
    """Pick uniformly at random from a fixed catalogue of creatives."""

    def __init__(self, creatives: List[List[float]]):
        self._catalogue = np.array(creatives, dtype=float)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self._catalogue[rng.integers(len(self._catalogue))].copy()


class FixedSampler(DisplaySampler):
    """Always return the same feature vector."""

    def __init__(self, x: List[float]):
        self._x = np.array(x, dtype=float)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self._x.copy()
