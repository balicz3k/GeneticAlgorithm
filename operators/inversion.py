from abc import ABC, abstractmethod
import numpy as np
from core.chromosome import Chromosome

class InversionStrategy(ABC):
    @abstractmethod
    def invert(self, chromosome: Chromosome) -> None:
        pass

class ClassicalInversion(InversionStrategy):
    def invert(self, chromosome: Chromosome) -> None:
        if len(chromosome.bits) < 2:
            return
            
        random_indices = np.random.choice(len(chromosome.bits), size=2, replace=False)
        start_point = min(random_indices)
        end_point = max(random_indices)
        
        chromosome.bits[start_point:end_point+1] = chromosome.bits[start_point:end_point+1][::-1]