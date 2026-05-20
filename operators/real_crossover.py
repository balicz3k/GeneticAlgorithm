from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from core.real_chromosome import RealChromosome


class RealCrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        pass


class ArithmeticCrossover(RealCrossoverStrategy):
    """Krzyżowanie arytmetyczne: child = k*p1 + (1-k)*p2."""

    def crossover(self, parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        k = np.random.rand()
        child1 = parent1.clone()
        child2 = parent2.clone()

        child1.genes = k * parent1.genes + (1 - k) * parent2.genes
        child2.genes = (1 - k) * parent1.genes + k * parent2.genes

        child1.clip_to_bounds()
        child2.clip_to_bounds()
        return child1, child2


class LinearCrossover(RealCrossoverStrategy):
    """Krzyżowanie liniowe — generuje 3 potomków, wybiera 2 najlepszych.

    Uwaga: aby poprawnie wybrać dwóch najlepszych kandydatów, operator
    musi wiedzieć, czy problem jest minimalizacją czy maksymalizacją.
    Domyślnie zakładamy minimalizację (zgodnie z funkcją Martin & Gaddy).
    """

    def __init__(self, fitness_func=None, is_maximization: bool = False):
        self.fitness_func = fitness_func
        self.is_maximization = is_maximization

    def crossover(self, parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        p1 = parent1.genes
        p2 = parent2.genes

        candidates_genes = [
            0.5 * p1 + 0.5 * p2,
            1.5 * p1 - 0.5 * p2,
            -0.5 * p1 + 1.5 * p2,
        ]

        candidates = []
        for g in candidates_genes:
            c = RealChromosome(parent1.bounds, genes=g.copy())
            c.clip_to_bounds()
            if self.fitness_func is not None:
                c.fitness = self.fitness_func(c.get_decoded_values())
            candidates.append(c)

        if self.fitness_func is not None:
            # Dla maksymalizacji sortujemy malejąco (najwyższa wartość = najlepsza),
            # dla minimalizacji rosnąco (najniższa wartość = najlepsza).
            reverse = self.is_maximization
            sentinel = float('-inf') if self.is_maximization else float('inf')
            candidates.sort(
                key=lambda c: c.fitness if c.fitness is not None else sentinel,
                reverse=reverse,
            )
        return candidates[0], candidates[1]


class BlendAlphaCrossover(RealCrossoverStrategy):
    """Krzyżowanie mieszające typu alfa (BLX-α)."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(self, parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()

        for i in range(parent1.num_variables):
            mn = min(parent1.genes[i], parent2.genes[i])
            mx = max(parent1.genes[i], parent2.genes[i])
            d = mx - mn
            low = mn - self.alpha * d
            high = mx + self.alpha * d

            child1.genes[i] = np.random.uniform(low, high)
            child2.genes[i] = np.random.uniform(low, high)

        child1.clip_to_bounds()
        child2.clip_to_bounds()
        return child1, child2


class BlendAlphaBetaCrossover(RealCrossoverStrategy):
    """Krzyżowanie mieszające typu alfa-beta (BLX-αβ)."""

    def __init__(self, alpha: float = 0.75, beta: float = 0.25):
        self.alpha = alpha
        self.beta = beta

    def crossover(self, parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()

        for i in range(parent1.num_variables):
            mn = min(parent1.genes[i], parent2.genes[i])
            mx = max(parent1.genes[i], parent2.genes[i])
            d = mx - mn
            low = mn - self.alpha * d
            high = mx + self.beta * d

            child1.genes[i] = np.random.uniform(low, high)
            child2.genes[i] = np.random.uniform(low, high)

        child1.clip_to_bounds()
        child2.clip_to_bounds()
        return child1, child2


class AverageCrossover(RealCrossoverStrategy):
    """Krzyżowanie uśredniające: child_i = (p1_i + p2_i) / 2."""

    def crossover(self, parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()

        avg = (parent1.genes + parent2.genes) / 2.0
        range_vec = np.array([b - a for a, b in parent1.bounds])

        # Niewielki, symetryczny szum dla obu potomków, aby nie były identyczne.
        noise1 = np.random.uniform(-0.01, 0.01, size=avg.shape) * range_vec
        noise2 = np.random.uniform(-0.01, 0.01, size=avg.shape) * range_vec
        child1.genes = avg + noise1
        child2.genes = avg + noise2

        child1.clip_to_bounds()
        child2.clip_to_bounds()
        return child1, child2
