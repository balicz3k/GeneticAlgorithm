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
    """Krzyżowanie liniowe — generuje 3 potomków, wybiera 2 najlepszych."""

    def __init__(self, fitness_func=None):
        self.fitness_func = fitness_func

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
            candidates.sort(key=lambda c: c.fitness if c.fitness is not None else float('-inf'), reverse=True)
            return candidates[0], candidates[1]
        else:
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
        child1.genes = avg.copy()
        child2.genes = avg.copy()

        # Dodajemy niewielki szum do child2, aby nie były identyczne
        noise = np.random.uniform(-0.01, 0.01, size=child2.genes.shape) * (
            np.array([b - a for a, b in parent1.bounds])
        )
        child2.genes = child2.genes + noise

        child1.clip_to_bounds()
        child2.clip_to_bounds()
        return child1, child2
