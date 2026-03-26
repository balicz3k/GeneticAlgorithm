from abc import ABC, abstractmethod
import numpy as np
from core.real_chromosome import RealChromosome


class RealMutationStrategy(ABC):
    @abstractmethod
    def mutate(self, chromosome: RealChromosome) -> None:
        pass


class UniformMutation(RealMutationStrategy):
    """Mutacja równomierna — losowy gen zastępowany wartością z uniform(a, b)."""

    def mutate(self, chromosome: RealChromosome) -> None:
        idx = np.random.randint(0, chromosome.num_variables)
        a, b = chromosome.bounds[idx]
        chromosome.genes[idx] = np.random.uniform(a, b)


class GaussianMutation(RealMutationStrategy):
    """Mutacja Gaussa — losowy gen przesuwany o N(0, sigma), obcięcie do bounds."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def mutate(self, chromosome: RealChromosome) -> None:
        idx = np.random.randint(0, chromosome.num_variables)
        chromosome.genes[idx] += np.random.normal(0, self.sigma)
        chromosome.clip_to_bounds()
