"""
Skrypt porównawczy: Binary vs Real chromosome.
Generuje wyniki i wykresy do sprawozdania P2.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import AlgorithmConfig, OptimizationTarget, RepresentationType
from utils.functions import martin_and_gaddy_function
from operators.selection import TournamentSelection
from operators.crossover import TwoPointCrossover
from operators.mutation import OnePointMutation
from operators.inversion import ClassicalInversion
from operators.real_crossover import ArithmeticCrossover, LinearCrossover, BlendAlphaCrossover, BlendAlphaBetaCrossover, AverageCrossover
from operators.real_mutation import UniformMutation, GaussianMutation
from core.genetic_algorithm import GeneticAlgorithm

# Parametry wspólne
FUNC = martin_and_gaddy_function
BOUNDS = [(-20, 20), (-20, 20)]
PRECISION = 6
POP_SIZE = 100
EPOCHS = 200
N_RUNS = 5  # Powtórzenia do uśrednienia

IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

def run_experiment(representation, crossover_strategy, mutation_strategy, label, inversion_strategy=None):
    """Uruchomienie eksperymentu z powtórzeniami i uśrednieniem."""
    all_best_per_epoch = []
    all_avg_per_epoch = []
    all_std_per_epoch = []
    all_times = []
    all_best_fitness = []

    for run in range(N_RUNS):
        config = AlgorithmConfig(
            fitness_func=FUNC,
            bounds=BOUNDS,
            precision=PRECISION,
            target=OptimizationTarget.MINIMIZE,
            population_size=POP_SIZE,
            epochs=EPOCHS,
            representation=representation,
            selection_strategy=TournamentSelection(3),
            crossover_strategy=crossover_strategy,
            mutation_strategy=mutation_strategy,
            inversion_strategy=inversion_strategy,
            cross_probability=0.8,
            mutation_probability=0.05,
            inversion_probability=0.05 if inversion_strategy else 0.0,
        )
        ga = GeneticAlgorithm(config)
        result = ga.run()

        best_series = [s.best for s in result["stats"]]
        avg_series = [s.avg for s in result["stats"]]
        std_series = [s.std_dev for s in result["stats"]]

        all_best_per_epoch.append(best_series)
        all_avg_per_epoch.append(avg_series)
        all_std_per_epoch.append(std_series)
        all_times.append(result["execution_time"])
        all_best_fitness.append(result["best_fitness_value"])

    # Uśrednienie po powtórzeniach
    mean_best = np.mean(all_best_per_epoch, axis=0)
    mean_avg = np.mean(all_avg_per_epoch, axis=0)
    mean_std = np.mean(all_std_per_epoch, axis=0)

    return {
        "label": label,
        "mean_best": mean_best,
        "mean_avg": mean_avg,
        "mean_std": mean_std,
        "avg_time": np.mean(all_times),
        "avg_best_fitness": np.mean(all_best_fitness),
        "best_fitness": min(all_best_fitness),
        "decoded_values": result["best_decoded_values"],
    }


def plot_comparison(results_list, title, ylabel, key, filename):
    """Generuje wykres porównawczy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2b2b2b')

    colors = ['#4caf50', '#2196f3', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#ffeb3b']
    
    for i, r in enumerate(results_list):
        epochs = np.arange(1, len(r[key]) + 1)
        ax.plot(epochs, r[key], label=r["label"], color=colors[i % len(colors)], linewidth=2)

    ax.set_title(title, color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoka (Generacja)", color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#2b2b2b', edgecolor='#555', labelcolor='white')
    ax.grid(color='#555', linestyle='-', linewidth=0.5, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color('#555')

    fig.tight_layout()
    filepath = os.path.join(IMG_DIR, filename)
    fig.savefig(filepath, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("PORÓWNANIE: Binarna vs Rzeczywista reprezentacja chromosomu")
    print("Funkcja: Martin & Gaddy, bounds=[-20,20]x[-20,20]")
    print(f"Pop={POP_SIZE}, Epochs={EPOCHS}, Runs={N_RUNS}")
    print("=" * 60)

    # --- Eksperymenty BINARNE ---
    print("\n[BINARY] TwoPoint Crossover + OnePoint Mutation + Inversion")
    binary_result = run_experiment(
        RepresentationType.BINARY,
        TwoPointCrossover(),
        OnePointMutation(),
        "Binarna (TwoPoint + OnePoint)",
        ClassicalInversion()
    )

    # --- Eksperymenty RZECZYWISTE ---
    real_results = []

    print("\n[REAL] Arithmetic Crossover + Gaussian Mutation")
    real_results.append(run_experiment(
        RepresentationType.REAL,
        ArithmeticCrossover(),
        GaussianMutation(sigma=1.0),
        "Rzeczywista (Arytmetyczne + Gauss)"
    ))

    print("\n[REAL] BLX-alpha Crossover + Gaussian Mutation")
    real_results.append(run_experiment(
        RepresentationType.REAL,
        BlendAlphaCrossover(alpha=0.5),
        GaussianMutation(sigma=1.0),
        "Rzeczywista (BLX-a + Gauss)"
    ))

    print("\n[REAL] Linear Crossover + Uniform Mutation")
    real_results.append(run_experiment(
        RepresentationType.REAL,
        LinearCrossover(fitness_func=FUNC),
        UniformMutation(),
        "Rzeczywista (Liniowe + Uniform)"
    ))

    print("\n[REAL] BLX-alpha-beta Crossover + Gaussian Mutation")
    real_results.append(run_experiment(
        RepresentationType.REAL,
        BlendAlphaBetaCrossover(alpha=0.75, beta=0.25),
        GaussianMutation(sigma=0.5),
        "Rzeczywista (BLX-ab + Gauss)"
    ))

    print("\n[REAL] Averaging Crossover + Gaussian Mutation")
    real_results.append(run_experiment(
        RepresentationType.REAL,
        AverageCrossover(),
        GaussianMutation(sigma=1.0),
        "Rzeczywista (Uśredniające + Gauss)"
    ))

    all_results = [binary_result] + real_results

    # --- Tabela wyników ---
    print("\n" + "=" * 90)
    print(f"{'Konfiguracja':<45} {'Best Fitness':>14} {'Avg Time (s)':>14} {'Decoded X':>16}")
    print("-" * 90)
    for r in all_results:
        x_str = f"({r['decoded_values'][0]:.3f}, {r['decoded_values'][1]:.3f})"
        print(f"{r['label']:<45} {r['best_fitness']:>14.6f} {r['avg_time']:>14.4f} {x_str:>16}")
    print("=" * 90)

    # --- Wykresy ---
    print("\nGenerowanie wykresów...")

    # 1. Porównanie best fitness per epoch
    plot_comparison(
        all_results,
        "Porównanie zbieżności — Best Fitness (binarna vs rzeczywista)",
        "Wartość funkcji celu (najlepsza w epoce)",
        "mean_best",
        "p2_best_comparison.png"
    )

    # 2. Porównanie average per epoch
    plot_comparison(
        all_results,
        "Porównanie zbieżności — Średnia populacji (binarna vs rzeczywista)",
        "Średnia wartość fitness populacji",
        "mean_avg",
        "p2_avg_comparison.png"
    )

    # 3. Porównanie std dev per epoch
    plot_comparison(
        all_results,
        "Odchylenie standardowe fitness populacji",
        "Odchylenie standardowe",
        "mean_std",
        "p2_std_comparison.png"
    )

    # 4. Porównanie tylko real crossover types
    plot_comparison(
        real_results,
        "Porównanie operatorów krzyżowania (chromosom rzeczywisty)",
        "Wartość funkcji celu (najlepsza w epoce)",
        "mean_best",
        "p2_real_crossover_comparison.png"
    )

    print("\nGOTOWE! Wykresy zapisane w katalogu img/")
