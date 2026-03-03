from typing import List, Tuple, Callable
import copy
from core.chromosome import Chromosome

class Population:
    def __init__(self, size: int, is_maximization: bool, bounds: List[Tuple[float, float]], precision: int, fitness_func: Callable[[List[float]], float]):
        self.size = size
        self.is_maximization = is_maximization
        self.bounds = bounds
        self.precision = precision

        self.fitness_func = fitness_func

        self.individuals: List[Chromosome] = []
        for _ in range(self.size):
            self.individuals.append(Chromosome(self.bounds, self.precision))

    def evaluate_fitness(self):
        for individual in self.individuals:
            decoded_value = individual.get_decoded_values()
            individual.fitness = self.fitness_func(decoded_value)
            if not self.is_maximization:
                individual.fitness *= -1

    def get_best_individual(self) -> Chromosome:
        best = max(self.individuals, key=lambda chrom: chrom.fitness)
        return best.clone()
    
    def get_worst_individual(self) -> Chromosome:
        worst = min(self.individuals, key=lambda chrom: chrom.fitness)
        return worst.clone()

    def get_best_fittness(self) -> float:
        best = max(self.individuals, key=lambda chrom: chrom.fitness)
        return best.fitness

    def get_worst_fittness(self) -> float:
        worst = min(self.individuals, key=lambda chrom: chrom.fitness)
        return worst.fitness

    def get_average_fittness(self) -> float:
        return sum(chrom.fitness for chrom in self.individuals) / len(self.individuals)