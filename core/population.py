from typing import List, Tuple, Callable
import copy
from core.chromosome import Chromosome

class Population:
    def __init__(self, size: int, bounds: List[Tuple[float, float]], precision: int, fitness_func: Callable[[List[float]], float]):
        self.size = size
        self.bounds = bounds
        self.precision = precision

        self.fitness_func = fitness_func

        self.individuals: List[Chromosome] = []
        for _ in range(self.size):
            self.individuals.append(Chromosome(self.bounds, self.precision))

    def evaluate_fitness(self, is_maximization: bool = True):
        for individual in self.individuals:
            decoded_values = individual.get_decoded_values()
            
            score = self.fitness_func(decoded_values)
            
            if not is_maximization:
                score = -score
                
            individual.fitness = score

    def get_best_individual(self) -> Chromosome:
        best = max(self.individuals, key=lambda chrom: chrom.fitness)
        return best.clone()