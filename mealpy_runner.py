"""
Eksperymenty PSO z biblioteki MealPy dla funkcji Martin & Gaddy.
Porównuje różne konfiguracje hiperparametrów i zestawia wyniki
z własnym GA (P2) oraz PyGAD (P3). Wykresy trafiają do img_mealpy/.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mealpy.swarm_based import PSO
from mealpy import FloatVar

from utils.functions import martin_and_gaddy_function
from utils.config import AlgorithmConfig, OptimizationTarget, RepresentationType
from operators.selection import TournamentSelection
from operators.crossover import UniformCrossover
from operators.mutation import TwoPointMutation
from operators.inversion import ClassicalInversion
from operators.real_crossover import BlendAlphaCrossover
from operators.real_mutation import GaussianMutation
from core.genetic_algorithm import GeneticAlgorithm

IMG_DIR = "img_mealpy"
os.makedirs(IMG_DIR, exist_ok=True)

FUNC = martin_and_gaddy_function
BOUNDS = [(-20, 20), (-20, 20)]
EPOCHS = 150
POP_SIZE = 100
N_RUNS = 5

PROBLEM = {
    "bounds": FloatVar(
        lb=[BOUNDS[0][0], BOUNDS[1][0]],
        ub=[BOUNDS[0][1], BOUNDS[1][1]],
        name="x"
    ),
    "minmax": "min",
    "obj_func": FUNC,
}

# Konfiguracje do porównania: inercja w, współczynnik poznawczy c1, społeczny c2
PSO_CONFIGS = [
    {"label": "PSO std (w=0.9, c1=2.0, c2=2.0)",          "w": 0.9, "c1": 2.0, "c2": 2.0},
    {"label": "PSO mała inercja (w=0.4, c1=2.0, c2=2.0)", "w": 0.4, "c1": 2.0, "c2": 2.0},
    {"label": "PSO poznawcze (w=0.7, c1=2.5, c2=1.5)",    "w": 0.7, "c1": 2.5, "c2": 1.5},
    {"label": "PSO społeczne (w=0.7, c1=1.5, c2=2.5)",    "w": 0.7, "c1": 1.5, "c2": 2.5},
]


def run_pso_config(w, c1, c2):
    histories = []
    best_values = []
    times = []
    last_solution = None

    for _ in range(N_RUNS):
        model = PSO.OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE, w=w, c1=c1, c2=c2)
        t0 = time.time()
        model.solve(PROBLEM)
        elapsed = time.time() - t0

        histories.append(model.history.list_global_best_fit)
        best_values.append(model.g_best.target.fitness)
        times.append(elapsed)
        last_solution = model.g_best.solution

    return {
        "mean_history": np.mean(histories, axis=0),
        "best": float(np.min(best_values)),
        "avg_best": float(np.mean(best_values)),
        "avg_time": float(np.mean(times)),
        "solution": last_solution,
    }


def plot_convergence_all(results, configs):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2b2b2b')

    colors = ['#00bcd4', '#4caf50', '#ff9800', '#f44336']
    for i, (res, cfg) in enumerate(zip(results, configs)):
        epochs = np.arange(1, len(res["mean_history"]) + 1)
        ax.plot(epochs, res["mean_history"], label=cfg["label"],
                color=colors[i % len(colors)], linewidth=2)

    ax.set_title("PSO — porównanie konfiguracji hiperparametrów", color='white',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteracja", color='white')
    ax.set_ylabel("Wartość funkcji celu (global best)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#2b2b2b', edgecolor='#555', labelcolor='white', fontsize=9)
    ax.grid(color='#555', linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#555')

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "pso_configs_comparison.png")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Zapisano: {path}")


def plot_best_config(res, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2b2b2b')

    epochs = np.arange(1, len(res["mean_history"]) + 1)
    ax.plot(epochs, res["mean_history"], color='#00bcd4', linewidth=2.5,
            label=f"Średni global best ({label}, {N_RUNS} przebiegów)")

    ax.set_title("PSO — zbieżność najlepszej konfiguracji", color='white',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteracja", color='white')
    ax.set_ylabel("Wartość funkcji celu", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#2b2b2b', edgecolor='#555', labelcolor='white')
    ax.grid(color='#555', linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#555')

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "pso_best_convergence.png")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Zapisano: {path}")


def run_own_ga(representation, crossover_op, mutation_op, inversion_op=None):
    """Uruchamia własne GA N_RUNS razy i zwraca uśrednioną historię best f(x) per epoka."""
    all_best = []
    for _ in range(N_RUNS):
        config = AlgorithmConfig(
            fitness_func=FUNC,
            bounds=BOUNDS,
            precision=6,
            target=OptimizationTarget.MINIMIZE,
            population_size=POP_SIZE,
            epochs=EPOCHS,
            representation=representation,
            selection_strategy=TournamentSelection(3),
            crossover_strategy=crossover_op,
            mutation_strategy=mutation_op,
            inversion_strategy=inversion_op,
            cross_probability=0.8,
            mutation_probability=0.05,
            inversion_probability=0.05 if inversion_op else 0.0,
        )
        ga = GeneticAlgorithm(config)
        result = ga.run()
        # Population.evaluate_fitness neguje wartości dla minimalizacji — odwracamy
        best_series = np.array([s.best for s in result["stats"]])
        all_best.append(-best_series)
    return np.mean(all_best, axis=0)


def plot_vs_ga(ga_real_hist, ga_binary_hist, pso_best_history):
    """Wykres porównujący PSO (P4) z własnym GA binarnym i rzeczywistym (P2).
    Wszystkie krzywe są rzeczywistymi wynikami uruchomień, uśrednionymi z N_RUNS przebiegów."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2b2b2b')

    epochs_ga  = np.arange(1, len(ga_real_hist) + 1)
    epochs_pso = np.arange(1, len(pso_best_history) + 1)

    ax.plot(epochs_ga,  ga_real_hist,   color='#4caf50', linewidth=2,
            label='Własny GA rzeczywisty (P2) — BLX-$\\alpha$ + Gauss')
    ax.plot(epochs_ga,  ga_binary_hist, color='#ff9800', linewidth=2,
            label='Własny GA binarny (P2) — Uniform + TwoPoint + Inwersja')
    ax.plot(epochs_pso, pso_best_history, color='#00bcd4', linewidth=2.5,
            label='PSO MealPy (P4) — mała inercja (w=0.4)')

    ax.set_title("Porównanie algorytmów — Projekty 2 i 4", color='white',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoka / Iteracja", color='white')
    ax.set_ylabel("Wartość funkcji celu — średni best f(x)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#2b2b2b', edgecolor='#555', labelcolor='white', fontsize=9)
    ax.grid(color='#555', linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#555')

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "comparison_p2_p3_p4.png")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Zapisano: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("EKSPERYMENTY PSO — MealPy")
    print(f"Funkcja: Martin & Gaddy, bounds=[-20,20]^2")
    print(f"Pop={POP_SIZE}, Epochs={EPOCHS}, Runs={N_RUNS}")
    print("=" * 60)

    results = []
    for cfg in PSO_CONFIGS:
        print(f"\n[PSO] {cfg['label']}")
        res = run_pso_config(cfg["w"], cfg["c1"], cfg["c2"])
        results.append(res)
        print(f"      best={res['best']:.8f}  avg={res['avg_best']:.8f}"
              f"  time={res['avg_time']:.4f}s  x={res['solution']}")

    print("\n" + "=" * 75)
    print(f"{'Konfiguracja':<45} {'Best f(x)':>12} {'Avg f(x)':>12} {'Czas (s)':>10}")
    print("-" * 75)
    for res, cfg in zip(results, PSO_CONFIGS):
        print(f"{cfg['label']:<45} {res['best']:>12.8f} {res['avg_best']:>12.8f} {res['avg_time']:>10.4f}")
    print("=" * 75)

    best_idx = int(np.argmin([r["avg_best"] for r in results]))
    best_res = results[best_idx]
    best_cfg = PSO_CONFIGS[best_idx]
    print(f"\nNajlepsza konfiguracja: {best_cfg['label']}")

    print("\nUruchamianie własnego GA (P2) dla wykresu porównawczego...")
    ga_real_hist = run_own_ga(
        RepresentationType.REAL,
        BlendAlphaCrossover(alpha=0.5),
        GaussianMutation(sigma=1.0),
    )
    ga_binary_hist = run_own_ga(
        RepresentationType.BINARY,
        UniformCrossover(),
        TwoPointMutation(),
        ClassicalInversion(),
    )

    print("\nGenerowanie wykresów...")
    plot_convergence_all(results, PSO_CONFIGS)
    plot_best_config(best_res, best_cfg["label"])
    plot_vs_ga(ga_real_hist, ga_binary_hist, best_res["mean_history"])
    print("\nGotowe! Wykresy w katalogu img_mealpy/")
