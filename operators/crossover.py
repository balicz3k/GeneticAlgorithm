from abc import ABC, abstractmethod
from typing import Tuple
from core.chromosome import Chromosome

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass
    
class OnePointCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass    

class TwoPointCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass    

class UniformCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass    

class DiscreteCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass
