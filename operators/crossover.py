from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from core.chromosome import Chromosome

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass
    
class OnePointCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        random_index = np.random.randint(1, len(parent1.bits))
        
        child1.bits = np.concatenate([parent1.bits[:random_index], parent2.bits[random_index:]])
        child2.bits = np.concatenate([parent2.bits[:random_index], parent1.bits[random_index:]])
        return child1, child2

class TwoPointCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()

        random_indices = np.random.choice(len(parent1.bits), size=2, replace=False)
        
        cp1 = min(random_indices)
        cp2 = max(random_indices)

        # (parent1, parent2, parent1)
        child1.bits = np.concatenate([
            parent1.bits[:cp1], 
            parent2.bits[cp1:cp2], 
            parent1.bits[cp2:]
        ])
        # (parent2, parent1, parent2)
        child2.bits = np.concatenate([
            parent2.bits[:cp1], 
            parent1.bits[cp1:cp2], 
            parent2.bits[cp2:]
        ])

        return child1, child2

class UniformCrossover(CrossoverStrategy):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()

        mask = np.random.randint(0, 2, size=len(parent1.bits)).astype(bool)

        child1.bits = np.where(mask, parent1.bits, parent2.bits)
        child2.bits = np.where(mask, parent2.bits, parent1.bits)

        return child1, child2

class DiscreteCrossover(CrossoverStrategy):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()

        mask = np.random.rand(len(parent1.bits)) < self.prob

        child1.bits = np.where(mask, parent1.bits, parent2.bits)
        child2.bits = np.where(mask, parent2.bits, parent1.bits)

        return child1, child2
