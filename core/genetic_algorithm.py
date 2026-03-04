import numpy as np
import copy
import time
from typing import List, Dict, Any
from utils.config import AlgorithmConfig, OptimizationTarget
from core.population import Population
from utils.stats import Stats

class GeneticAlgorithm:
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.is_maximization = (config.target == OptimizationTarget.MAXIMIZE)
        self.stats = []

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        
        current_population = Population(
            size=self.config.population_size,
            is_maximization=self.is_maximization,
            bounds=self.config.bounds,
            precision=self.config.precision,
            fitness_func=self.config.fitness_func
        )
        current_population.evaluate_fitness()
        
        best_overall = current_population.get_best_individual()
        worst_overall = current_population.get_worst_individual()
        self._update_stats(current_population, best_overall.fitness, worst_overall.fitness)

        for epoch in range(1, self.config.epochs + 1):
            new_individuals = []
            
            if self.config.elitism:
                elite_clone = current_population.get_best_individual()
                new_individuals.append(elite_clone)

            needed_children = self.config.population_size - len(new_individuals)
            num_parents_to_select = needed_children
            if num_parents_to_select % 2 != 0:
                num_parents_to_select += 1

            parents = self.config.selection_strategy.select(
                population=current_population, 
                num_parents=num_parents_to_select
            )

            i = 0
            while i < len(parents) and len(new_individuals) < self.config.population_size:
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else parents[i]
                i += 2

                if np.random.rand() < self.config.cross_probability:
                    child1, child2 = self.config.crossover_strategy.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()

                if np.random.rand() < self.config.mutation_probability:
                    self.config.mutation_strategy.mutate(child1)
                
                if self.config.inversion_strategy and np.random.rand() < self.config.inversion_probability:
                    self.config.inversion_strategy.invert(child1)

                if np.random.rand() < self.config.mutation_probability:
                    self.config.mutation_strategy.mutate(child2)
                    
                if self.config.inversion_strategy and np.random.rand() < self.config.inversion_probability:
                    self.config.inversion_strategy.invert(child2)

                new_individuals.append(child1)
                if len(new_individuals) < self.config.population_size:
                    new_individuals.append(child2)

            current_population.individuals = new_individuals
            current_population.evaluate_fitness()

            epoch_best = current_population.get_best_individual()
            epoch_worst = current_population.get_worst_individual()
            
            if epoch_best.fitness > best_overall.fitness:
                best_overall = epoch_best.clone()
            if epoch_worst.fitness < worst_overall.fitness:
                worst_overall = epoch_worst.clone()
            
            self._update_stats(current_population, best_overall.fitness, worst_overall.fitness)

        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "best_chromosome_bits": best_overall.bits.tolist(),
            "best_decoded_values": best_overall.get_decoded_values(),
            "best_fitness_value": best_overall.fitness,
            "execution_time": execution_time,
            "stats": self.stats
        }

    def _update_stats(self, current_population: Population, best_overall: float, worst_overall: float):
        stats = Stats()
        stats.best = current_population.get_best_fittness()
        stats.worst = current_population.get_worst_fittness()
        stats.avg = current_population.get_average_fittness()
        stats.best_overall = best_overall
        stats.worst_overall = worst_overall
        self.stats.append(stats)
