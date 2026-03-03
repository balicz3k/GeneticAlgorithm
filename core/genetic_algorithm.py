import numpy as np
import copy
import time
from typing import List, Dict, Any
from utils.config import AlgorithmConfig, OptimizationTarget
from core.population import Population
from core.chromosome import Chromosome

class GeneticAlgorithm:
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        
        self.is_max = (config.target == OptimizationTarget.MAXIMIZE)

        self.best_history: List[float] = []

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        
        # --- PHASE 0: POPULATION INITIALIZATION ---
        current_population = Population(
            size=self.config.population_size,
            bounds=self.config.bounds,
            precision=self.config.precision,
            fitness_func=self.config.fitness_func
        )
        current_population.evaluate_fitness(is_maximization=self.is_max)
        
        best_overall = current_population.get_best_individual()
        
        actual_fitness = best_overall.fitness if self.is_max else -best_overall.fitness
        self.best_history.append(actual_fitness)

        for epoch in range(1, self.config.epochs + 1):
            new_individuals = []
            
            # --- PHASE 1: ELITISM ---
            if self.config.elitism:
                elite_clone = current_population.get_best_individual()
                new_individuals.append(elite_clone)

            # --- PHASE 2: SELECTION ---
            needed_children = self.config.population_size - len(new_individuals)
            num_parents_to_select = needed_children
            if num_parents_to_select % 2 != 0:
                num_parents_to_select += 1

            parents = self.config.selection_strategy.select(
                population=current_population, 
                num_parents=num_parents_to_select
            )

            # --- PHASE 3: CROSSOVER AND MUTATION ---
            i = 0
            while i < len(parents) and len(new_individuals) < self.config.population_size:
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else parents[i]
                i += 2

                # Crossover
                if np.random.rand() < self.config.cross_probability:
                    child1, child2 = self.config.crossover_strategy.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()

                # Mutation
                if np.random.rand() < self.config.mutation_probability:
                    self.config.mutation_strategy.mutate(child1)
                if np.random.rand() < self.config.mutation_probability:
                    self.config.mutation_strategy.mutate(child2)

                new_individuals.append(child1)
                if len(new_individuals) < self.config.population_size:
                    new_individuals.append(child2)

            # --- PHASE 4: POPULATION EVALUATION ---
            current_population.individuals = new_individuals
            current_population.evaluate_fitness(is_maximization=self.is_max)

            # --- PHASE 5: STATISTICS UPDATE ---
            epoch_best = current_population.get_best_individual()
            
            if epoch_best.fitness > best_overall.fitness:
                best_overall = epoch_best.clone()

            run_fitness = best_overall.fitness if self.is_max else -best_overall.fitness
            self.best_history.append(run_fitness)

        # --- PHASE 6: RETURN RESULTS ---
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "best_chromosome": best_overall.bits.tolist(),
            "best_decoded_values": best_overall.get_decoded_values(),
            "best_fitness_value": best_overall.fitness if self.is_max else -best_overall.fitness,
            "history_stats": self.best_history,
            "execution_time": execution_time
        }
