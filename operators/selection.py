from abc import ABC, abstractmethod
from typing import List
from core.population import Population
from core.chromosome import Chromosome

class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        pass

class BestSelection(SelectionStrategy):
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        pass

class RouletteSelection(SelectionStrategy):
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        pass

class TournamentSelection(SelectionStrategy):
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        pass
    