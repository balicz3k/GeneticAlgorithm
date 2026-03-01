import numpy as np
from typing import List

from utils.config import AlgorithmConfig, OptimizationTarget
from core.genetic_algorithm import GeneticAlgorithm
from operators.selection import RouletteSelection, BestSelection, TournamentSelection
from operators.crossover import OnePointCrossover, TwoPointCrossover, UniformCrossover, DiscreteCrossover
from operators.mutation import MarginalMutation, OnePointMutation, TwoPointMutation

def sphere_function(variables: List[float]) -> float:
    return sum(x ** 2 for x in variables)

if __name__ == "__main__":
    config = AlgorithmConfig(
        fitness_func=sphere_function,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)], 
        precision=3,
        target=OptimizationTarget.MINIMIZE,
        population_size=50,
        epochs=100,
        elitism=True,
        cross_probability=0.8,
        mutation_probability=0.05,
        inversion_probability=0.0,
        selection_strategy=TournamentSelection(tournament_size=3),
        crossover_strategy=TwoPointCrossover(),
        mutation_strategy=OnePointMutation()
    )

    print("Initializing algorithm...")
    ga = GeneticAlgorithm(config)
    
    print(f"Running evolution... Looking for the minimum over {config.epochs} epochs.")
    result = ga.run()

    print("\nDONE! Best solution found:")
    print(f"Best variables [x, y]: {result['best_decoded_values']}")
    print(f"Function value (Y): {result['best_fitness_value']}")
    print(f"Genetic DNA (Bits): {result['best_chromosome']}")
