from abc import ABC, abstractmethod
import numpy as np
from core.chromosome import Chromosome

class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> None:
        pass

class MarginalMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome) -> None:
        edge_index = np.random.choice([0, -1])
        chromosome.invert_bit(edge_index)

class OnePointMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome) -> None:
        random_index = np.random.randint(0, len(chromosome.bits))
        chromosome.invert_bit(random_index)

class TwoPointMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome) -> None:
        if len(chromosome.bits) < 2:
            return
        random_indices = np.random.choice(len(chromosome.bits), size=2, replace=False)
        for idx in random_indices:
            chromosome.invert_bit(idx)
