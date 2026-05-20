"""Wrapper PyGAD zgodny ze stylem przykładów (example_01..04).

Najważniejsze założenia (poprawione po przeglądzie przykładów):

* Funkcja celu jest minimalizowana, a PyGAD natywnie maksymalizuje. Idąc za
  example_01/02/03 zamiast negacji ``-f(x)`` (która łamie selekcję ruletkową
  ``rws`` wymagającą dodatnich wartości fitness) używamy postaci
  ``1.0 / (f(x) + EPS)``. Im mniejsza wartość funkcji celu, tym wyższy fitness.
* Bounds dla reprezentacji rzeczywistej są wymuszane przez parametr
  ``gene_space`` (wzorzec z example_04) zamiast jedynie ``init_range_low/high``,
  dzięki czemu mutacje również trzymają się dziedziny.
* Statystyki w callbacku liczymy na **oryginalnej** wartości funkcji celu
  (odwracając trick ``1/y``), aby raportować wartości w skali zadania.
* Opcjonalnie podpinamy ``parallel_processing`` (zgodnie z example_02/03).
* Konfigurowalny ``logger`` (jak w example_02/03).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pygad


EPS = 1e-12


def _build_default_logger() -> logging.Logger:
    """Tworzy ``logger`` zgodny stylem z example_02/03 (tylko gdy brak handlerów)."""
    logger = logging.getLogger("pygad_runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class PyGADRunner:
    """Uruchamia PyGAD w dwóch reprezentacjach (binarnej i rzeczywistej)."""

    def __init__(
        self,
        func: Callable[[Sequence[float]], float],
        bounds: List[Tuple[float, float]],
        num_vars: int,
        bits_per_var: int = 20,
        logger: Optional[logging.Logger] = None,
        log_every_generation: bool = False,
    ) -> None:
        self.func = func
        self.bounds = bounds
        self.num_vars = num_vars
        self.bits_per_var = bits_per_var
        self.num_genes = num_vars * bits_per_var

        self.logger = logger if logger is not None else _build_default_logger()
        self.log_every_generation = log_every_generation

        # Statystyki zbierane w callbacku (na oryginalnej wartości funkcji celu)
        self.best_history: List[float] = []
        self.mean_history: List[float] = []
        self.std_history: List[float] = []

    # ------------------------------------------------------------------
    # Dekodowanie chromosomu binarnego do wektora liczb rzeczywistych
    # ------------------------------------------------------------------
    def decode_binary_individual(self, individual: Sequence[int]) -> List[float]:
        decoded: List[float] = []
        arr = np.asarray(individual, dtype=np.int64)
        powers_of_two = 2 ** np.arange(self.bits_per_var - 1, -1, -1)
        denom = (2 ** self.bits_per_var) - 1
        for i in range(self.num_vars):
            a, b = self.bounds[i]
            segment = arr[i * self.bits_per_var : (i + 1) * self.bits_per_var]
            decimal_val = int(np.sum(segment * powers_of_two))
            real_val = a + decimal_val * (b - a) / denom
            decoded.append(float(real_val))
        return decoded

    # ------------------------------------------------------------------
    # Funkcje przystosowania — trick 1/(f+EPS) zgodny z example_01/02/03
    # ------------------------------------------------------------------
    def fitness_binary(self, ga_instance, solution, sol_idx):
        decoded = self.decode_binary_individual(solution)
        return 1.0 / (self.func(decoded) + EPS)

    def fitness_real(self, ga_instance, solution, sol_idx):
        clipped = [
            float(np.clip(val, self.bounds[i][0], self.bounds[i][1]))
            for i, val in enumerate(solution)
        ]
        return 1.0 / (self.func(clipped) + EPS)

    # ------------------------------------------------------------------
    # Callback statystyk (analogiczny do example_02/03 — odwrotność znów)
    # ------------------------------------------------------------------
    def on_generation(self, ga_instance):
        fitnesses = np.asarray(ga_instance.last_generation_fitness, dtype=np.float64)
        # Powrót z fitness "maksymalizacyjnego" do oryginalnej wartości celu.
        original = 1.0 / np.maximum(fitnesses, EPS) - EPS
        original = np.clip(original, 0.0, None)

        best = float(np.min(original))
        mean = float(np.mean(original))
        std = float(np.std(original))

        self.best_history.append(best)
        self.mean_history.append(mean)
        self.std_history.append(std)

        if self.log_every_generation:
            self.logger.info(
                "Gen=%d  best=%.6g  mean=%.6g  std=%.6g",
                ga_instance.generations_completed,
                best,
                mean,
                std,
            )

    # ------------------------------------------------------------------
    # Główna pętla uruchomieniowa PyGAD
    # ------------------------------------------------------------------
    def run_experiment(
        self,
        is_binary: bool = True,
        num_generations: int = 100,
        sol_per_pop: int = 50,
        parent_selection_type: Union[str, Callable] = "tournament",
        crossover_type: Union[str, Callable] = "two_points",
        mutation_type: Union[str, Callable] = "random",
        K_tournament: int = 3,
        mutation_percent_genes: float = 10,
        keep_elitism: int = 1,
        parallel_processing: Optional[List] = None,
    ) -> dict:
        # Reset statystyk
        self.best_history = []
        self.mean_history = []
        self.std_history = []

        if is_binary:
            gene_type = int
            num_genes = self.num_genes
            init_range_low = 0
            init_range_high = 2
            # Każdy gen może przyjąć jedynie 0 lub 1 — wzorzec z example_04.
            gene_space = [[0, 1]] * num_genes
            fitness_func = self.fitness_binary
        else:
            gene_type = float
            num_genes = self.num_vars
            init_range_low = self.bounds[0][0]
            init_range_high = self.bounds[0][1]
            # Bounds per-zmienna w stylu example_04 (ciągłe przedziały).
            gene_space = [
                {"low": float(a), "high": float(b)} for a, b in self.bounds
            ]
            fitness_func = self.fitness_real

        ga_kwargs = dict(
            num_generations=num_generations,
            num_parents_mating=max(2, sol_per_pop // 2),
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            fitness_func=fitness_func,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            gene_type=gene_type,
            gene_space=gene_space,
            parent_selection_type=parent_selection_type,
            K_tournament=K_tournament,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes,
            keep_elitism=keep_elitism,
            on_generation=self.on_generation,
            logger=self.logger,
            suppress_warnings=True,
        )
        if parallel_processing is not None:
            ga_kwargs["parallel_processing"] = parallel_processing

        ga_instance = pygad.GA(**ga_kwargs)
        ga_instance.run()

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()

        if is_binary:
            decoded_solution = self.decode_binary_individual(best_solution)
        else:
            decoded_solution = [
                float(np.clip(val, self.bounds[i][0], self.bounds[i][1]))
                for i, val in enumerate(best_solution)
            ]

        # Powrót z trick'u 1/(f+EPS) do oryginalnej wartości funkcji celu.
        original_best = max(0.0, 1.0 / max(best_solution_fitness, EPS) - EPS)

        return {
            "best_solution": np.asarray(best_solution).tolist(),
            "decoded_solution": decoded_solution,
            "best_fitness": original_best,
            "best_history": self.best_history,
            "mean_history": self.mean_history,
            "std_history": self.std_history,
        }


# ---------------------------------------------------------------------------
# Autorskie operatory — sygnatury i zachowanie zgodne z example_03_*.
# ---------------------------------------------------------------------------
def custom_crossover_func(parents, offspring_size, ga_instance):
    """Krzyżowanie jednopunktowe z losowym punktem cięcia (jak w example_03)."""
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        random_split_point = np.random.choice(range(offspring_size[1]))
        parent1[random_split_point:] = parent2[random_split_point:]
        offspring.append(parent1)
        idx += 1
    return np.array(offspring)


def custom_mutation_func(offspring, ga_instance):
    """Mutacja addytywna z rozkładu Gaussa (jak w example_03)."""
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))
        offspring[chromosome_idx, random_gene_idx] += np.random.normal(0.0, 1.0)
    return offspring
