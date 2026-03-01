from abc import ABC, abstractmethod
from core.chromosome import Chromosome

class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> None:
        pass

class MarginalMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome) -> None:
        pass

class OnePointMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome) -> None:
        pass

class TwoPointMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome) -> None:
        pass
