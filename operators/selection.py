from abc import ABC, abstractmethod
from typing import List
import numpy as np
from core.population import Population
from core.chromosome import Chromosome

class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        pass

class BestSelection(SelectionStrategy):
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        best_individuals = sorted(population.individuals, key=lambda chrom: chrom.fitness, reverse=True)[:num_parents]
        return [chrom.clone() for chrom in best_individuals]

class RouletteSelection(SelectionStrategy):
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        fitness_scores = np.array([chrom.fitness for chrom in population.individuals])

        min_fitness = np.min(fitness_scores)
        if(min_fitness < 0):
            fitness_scores += np.abs(min_fitness)

        total_fitness = np.sum(fitness_scores)

        if total_fitness == 0:
            probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
        else:
            probabilities = fitness_scores / total_fitness

        selected_indices = np.random.choice(len(population.individuals), size=num_parents, p=probabilities, replace=True)
        return [population.individuals[i].clone() for i in selected_indices]

class TournamentSelection(SelectionStrategy):
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: Population, num_parents: int) -> List[Chromosome]:
        selected_individuals = []
        for _ in range(num_parents):
            tournament_indices = np.random.choice(len(population.individuals), size=self.tournament_size, replace=False)
            tournament_combatants = [population.individuals[i] for i in tournament_indices]
            winner = max(tournament_combatants, key=lambda chrom: chrom.fitness)
            selected_individuals.append(winner.clone())
        return selected_individuals
        