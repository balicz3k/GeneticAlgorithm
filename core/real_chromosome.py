import numpy as np
from typing import List, Tuple


class RealChromosome:
    """Chromosom z reprezentacją rzeczywistą — geny to wektor float."""

    def __init__(self, bounds: List[Tuple[float, float]], genes: np.ndarray = None):
        self.bounds = bounds
        self.num_variables = len(bounds)

        if genes is not None:
            self.genes = np.array(genes, dtype=np.float64)
        else:
            self.genes = np.array(
                [np.random.uniform(a, b) for a, b in bounds],
                dtype=np.float64,
            )

        self.fitness = None

    def get_decoded_values(self) -> List[float]:
        """Geny rzeczywiste = wartości bezpośrednio."""
        return self.genes.tolist()

    def clip_to_bounds(self) -> None:
        """Obcina geny do dozwolonych zakresów."""
        for i, (a, b) in enumerate(self.bounds):
            self.genes[i] = np.clip(self.genes[i], a, b)

    def clone(self) -> "RealChromosome":
        cloned = RealChromosome(self.bounds, genes=self.genes.copy())
        cloned.fitness = self.fitness
        return cloned
