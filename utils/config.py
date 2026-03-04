from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Callable

from operators.selection import SelectionStrategy
from operators.crossover import CrossoverStrategy
from operators.mutation import MutationStrategy
from operators.inversion import InversionStrategy

class OptimizationTarget(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass
class AlgorithmConfig:
    fitness_func: Callable[[List[float]], float] 
    bounds: List[Tuple[float, float]]
    precision: int
    target: OptimizationTarget

    population_size: int     
    epochs: int
    
    elitism: bool = True

    cross_probability: float = 0.8  
    mutation_probability: float = 0.02
    inversion_probability: float = 0.05

    selection_strategy: SelectionStrategy = None
    crossover_strategy: CrossoverStrategy = None
    mutation_strategy: MutationStrategy = None
    inversion_strategy: InversionStrategy = None
