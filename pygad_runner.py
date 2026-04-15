import pygad
import numpy as np

class PyGADRunner:
    """Wrapper klasa do uruchamiania algorytmów za pomocą biblioteki PyGAD z dwiema reprezentacjami."""

    def __init__(self, func, bounds, num_vars, bits_per_var=20):
        self.func = func
        self.bounds = bounds
        self.num_vars = num_vars
        self.bits_per_var = bits_per_var
        self.num_genes = num_vars * bits_per_var

        # Statystyki zbierane w callbacku
        self.best_history = []
        self.mean_history = []
        self.std_history = []

    def decode_binary_individual(self, individual):
        """Dekodowanie łańcucha bitów na wektor zmiennych rzeczywistych."""
        decoded = []
        for i in range(self.num_vars):
            a, b = self.bounds[i]
            segment = individual[i * self.bits_per_var : (i + 1) * self.bits_per_var]
            
            # Przeliczanie ciągu bitów na wartość dziesiętną
            # Ciąg bitów traktujemy jako liczbę w systemie dwójkowym
            powers_of_two = 2 ** np.arange(self.bits_per_var - 1, -1, -1)
            decimal_val = np.sum(segment * powers_of_two)
            
            real_val = a + decimal_val * (b - a) / ((2 ** self.bits_per_var) - 1)
            decoded.append(real_val)
        return decoded

    def fitness_binary(self, ga_instance, solution, sol_idx):
        """Funkcja przystosowania dla reprezentacji binarnej."""
        decoded = self.decode_binary_individual(solution)
        # PyGAD maksymalizuje, nasza funkcja minimalizuje, więc zwracamy ujemną wartość
        return -self.func(decoded)

    def fitness_real(self, ga_instance, solution, sol_idx):
        """Funkcja przystosowania dla reprezentacji rzeczywistej."""
        # Wymuszamy obcięcie do zdefiniowanych ograniczeń
        clipped_solution = []
        for i, val in enumerate(solution):
            a, b = self.bounds[i]
            clipped_solution.append(np.clip(val, a, b))
        return -self.func(clipped_solution)

    def on_generation(self, ga_instance):
        """Callback zbierający statystyki z każdej epoki."""
        fitnesses = ga_instance.last_generation_fitness
        
        # Zapamiętujemy oryginalne zminimalizowane wartości fitness (odwracamy znak z powrotem na dodatni, 
        # ponieważ PyGAD operuje na zmakymilizowanym fitness, a my minimalizujemy).
        original_fitnesses = -fitnesses
        
        self.best_history.append(np.min(original_fitnesses))
        self.mean_history.append(np.mean(original_fitnesses))
        self.std_history.append(np.std(original_fitnesses))

    def run_experiment(self, is_binary=True, num_generations=100, sol_per_pop=50, 
                       parent_selection_type="tournament", crossover_type="two_points", 
                       mutation_type="random", K_tournament=3):
        """Główna pętla wywołująca PyGAD."""
        # Reset statystyk
        self.best_history = []
        self.mean_history = []
        self.std_history = []

        if is_binary:
            gene_type = int
            init_range_low = 0
            init_range_high = 2
            num_genes = self.num_genes
            fitness_func = self.fitness_binary
        else:
            gene_type = float
            init_range_low = self.bounds[0][0]
            init_range_high = self.bounds[0][1]
            num_genes = self.num_vars
            fitness_func = self.fitness_real

        # Inicjalizacja instancji PyGAD
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=sol_per_pop // 2,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            fitness_func=fitness_func,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            gene_type=gene_type,
            parent_selection_type=parent_selection_type,
            K_tournament=K_tournament,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=10,
            keep_elitism=1,
            on_generation=self.on_generation,
            suppress_warnings=True
        )

        ga_instance.run()

        best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()
        
        if is_binary:
            decoded_solution = self.decode_binary_individual(best_solution)
        else:
            decoded_solution = [np.clip(val, self.bounds[i][0], self.bounds[i][1]) for i, val in enumerate(best_solution)]

        return {
            "best_solution": best_solution.tolist(),
            "decoded_solution": decoded_solution,
            "best_fitness": -best_solution_fitness,  # Przywracamy ujemny znak do naturalnej minimalizacji
            "best_history": self.best_history,
            "mean_history": self.mean_history,
            "std_history": self.std_history
        }

# Klasy z własnymi operatorami z projektu
def custom_crossover_func(parents, offspring_size, ga_instance):
    """Zadana autorska funkcja krzyżowania (single point modyfikacja) z polecenia Projektu 3."""
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
    """Zadana autorska funkcja mutacji pseudo-Gaussa z polecenia Projektu 3."""
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))
        # Jeśli geny są zmiennoprzecinkowe, to dodaje losową wartość [0, 1)
        # Jeśli są całkowite, dodanie float może powodować błędy ucięcia, ale zachowujemy strukturę.
        offspring[chromosome_idx, random_gene_idx] += np.random.random()
    return offspring
