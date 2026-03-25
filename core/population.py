from typing import List, Tuple, Callable, Union
from core.chromosome import Chromosome
from core.real_chromosome import RealChromosome

class Population:
    def __init__(self, size: int, is_maximization: bool, bounds: List[Tuple[float, float]], precision: int, fitness_func: Callable[[List[float]], float], representation: str = "binary"):
        self.size = size
        self.is_maximization = is_maximization
        self.bounds = bounds
        self.precision = precision
        self.representation = representation

        self.fitness_func = fitness_func

        self.individuals: List[Union[Chromosome, RealChromosome]] = []
        for _ in range(self.size):
            if self.representation == "real":
                self.individuals.append(RealChromosome(self.bounds))
            else:
                self.individuals.append(Chromosome(self.bounds, self.precision))

    def evaluate_fitness(self):
        for individual in self.individuals:
            decoded_value = individual.get_decoded_values()
            individual.fitness = self.fitness_func(decoded_value)
            if not self.is_maximization:
                individual.fitness *= -1

    def get_best_individual(self) -> Union[Chromosome, RealChromosome]:
        best = max(self.individuals, key=lambda chrom: chrom.fitness)
        return best.clone()
    
    def get_worst_individual(self) -> Union[Chromosome, RealChromosome]:
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

    def get_std_dev_fitness(self) -> float:
        import numpy as np
        fitnesses = [chrom.fitness for chrom in self.individuals]
        return float(np.std(fitnesses))