import customtkinter as ctk
import os
import csv
from tkinter import messagebox
from typing import Dict, Any, Optional

from utils.config import AlgorithmConfig, OptimizationTarget, RepresentationType
from utils.functions import AVAILABLE_FUNCTIONS

from operators.selection import BestSelection, RouletteSelection, TournamentSelection
from operators.crossover import OnePointCrossover, TwoPointCrossover, UniformCrossover, DiscreteCrossover
from operators.mutation import MarginalMutation, OnePointMutation, TwoPointMutation
from operators.inversion import ClassicalInversion
from operators.real_crossover import (
    ArithmeticCrossover, LinearCrossover, BlendAlphaCrossover,
    BlendAlphaBetaCrossover, AverageCrossover
)
from operators.real_mutation import UniformMutation, GaussianMutation
from core.genetic_algorithm import GeneticAlgorithm

from gui.charts_panel import ChartsPanel

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Classical Genetic Algorithm Editor")
        self.geometry("1100x800")
        self.minsize(900, 700)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=0, minsize=420)
        self.main_container.grid_columnconfigure(1, weight=1)

        self.config_frame = ctk.CTkScrollableFrame(self.main_container, label_text="Algorithm Parameters", width=400)
        self.config_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.config_frame.grid_columnconfigure(0, weight=1)
        self.config_frame.grid_columnconfigure(1, weight=1)

        self.init_variables()
        self.build_config_ui()

        self.action_frame = ctk.CTkFrame(self.main_container)
        self.action_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        # Layout inside right panel
        self.action_frame.grid_rowconfigure(1, weight=1)
        self.action_frame.grid_columnconfigure(0, weight=1)
        
        self.btn_run = ctk.CTkButton(
            self.action_frame, 
            text="RUN EXPERIMENT", 
            height=60, 
            font=("Inter", 16, "bold"),
            command=self.run_algorithm, 
            fg_color="#2c8f41", 
            hover_color="#1f652e"
        )
        self.btn_run.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.results_frame = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        self.results_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.res_label = ctk.CTkLabel(self.results_frame, text="Execution Results Summary", font=("Inter", 14, "bold"))
        self.res_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.results_textbox = ctk.CTkTextbox(self.results_frame, font=("Courier", 13))
        self.results_textbox.grid(row=1, column=0, sticky="nsew")
        self.results_textbox.insert("0.0", "Configuration ready. Click 'Run Experiment' to begin...\n")
        self.results_textbox.configure(state="disabled")
        
        self.btn_show_charts = ctk.CTkButton(
            self.action_frame, 
            text="VIEW CHARTS 📈", 
            height=40,
            command=self.show_chart_view,
            state="disabled"
        )
        self.btn_show_charts.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.charts_container = ChartsPanel(self, on_back_callback=self.show_main_view)

        self.last_results = None
        self.last_csv_path = None

    def init_variables(self):
        self.var_representation = ctk.StringVar(value="Binary")
        self.var_function = ctk.StringVar(value=list(AVAILABLE_FUNCTIONS.keys())[0])
        self.var_target = ctk.StringVar(value="MINIMIZE")
        self.var_num_vars = ctk.StringVar(value="2")
        self.var_bound_min = ctk.StringVar(value="-65.536")
        self.var_bound_max = ctk.StringVar(value="65.536")
        self.var_precision = ctk.StringVar(value="3")
        
        self.var_pop_size = ctk.StringVar(value="100")
        self.var_epochs = ctk.StringVar(value="150")
        self.var_elitism = ctk.BooleanVar(value=True)

        self.var_cross_prob = ctk.StringVar(value="0.8")
        self.var_mut_prob = ctk.StringVar(value="0.05")
        self.var_inv_prob = ctk.StringVar(value="0.0")

        self.var_sel_strategy = ctk.StringVar(value="Tournament")
        self.var_cross_strategy = ctk.StringVar(value="TwoPoint")
        self.var_mut_strategy = ctk.StringVar(value="OnePoint")
        self.var_inv_strategy = ctk.StringVar(value="Classical")

        self.var_blx_alpha = ctk.StringVar(value="0.5")
        self.var_blx_beta = ctk.StringVar(value="0.25")

    def build_config_ui(self):
        row_id = 0

        # Referencje na widgety, które trzeba dynamicznie chować/pokazywać
        self.dynamic_widgets = {}

        def add_header(text):
            nonlocal row_id
            header = ctk.CTkLabel(self.config_frame, text=text, font=("Inter", 14, "bold"), text_color="#569cd6")
            header.grid(row=row_id, column=0, columnspan=2, pady=(20, 5), sticky="w")
            row_id += 1
            return header

        def add_entry(label_text, variable, tag=None):
            nonlocal row_id
            lbl = ctk.CTkLabel(self.config_frame, text=label_text)
            lbl.grid(row=row_id, column=0, sticky="w", pady=5, padx=(5, 10))
            entry = ctk.CTkEntry(self.config_frame, textvariable=variable, width=140)
            entry.grid(row=row_id, column=1, sticky="e", pady=5, padx=5)
            if tag:
                self.dynamic_widgets[tag] = (lbl, entry, row_id)
            row_id += 1

        def add_dropdown(label_text, variable, options, tag=None, command=None):
            nonlocal row_id
            lbl = ctk.CTkLabel(self.config_frame, text=label_text)
            lbl.grid(row=row_id, column=0, sticky="w", pady=5, padx=(5, 10))
            opt = ctk.CTkOptionMenu(self.config_frame, variable=variable, values=options, width=140, command=command)
            opt.grid(row=row_id, column=1, sticky="e", pady=5, padx=5)
            if tag:
                self.dynamic_widgets[tag] = (lbl, opt, row_id)
            row_id += 1
            return opt

        add_header("0. Representation")
        add_dropdown("Chromosome Type:", self.var_representation, ["Binary", "Real"], command=self._on_representation_change)

        add_header("1. Environment & Objective")
        add_dropdown("Objective Function:", self.var_function, list(AVAILABLE_FUNCTIONS.keys()))
        add_dropdown("Optimization Target:", self.var_target, ["MINIMIZE", "MAXIMIZE"])
        add_entry("Number of Variables:", self.var_num_vars)
        add_entry("Lower Bound (Min):", self.var_bound_min)
        add_entry("Upper Bound (Max):", self.var_bound_max)
        add_entry("Number Precision:", self.var_precision, tag="precision")

        add_header("2. Epochs & Engine")
        add_entry("Population Size:", self.var_pop_size)
        add_entry("Generations (Epochs):", self.var_epochs)
        self.elitism_check = ctk.CTkCheckBox(self.config_frame, text="Enable Elitism (Save the Best)", variable=self.var_elitism)
        self.elitism_check.grid(row=row_id, column=0, columnspan=2, sticky="w", pady=10, padx=5)
        row_id += 1
        
        add_header("3. Evolutionary Probabilities")
        add_entry("Crossover Chance (Pc):", self.var_cross_prob)
        add_entry("Mutation Chance (Pm):", self.var_mut_prob)
        add_entry("Inversion Chance (Pi):", self.var_inv_prob, tag="inv_prob")

        add_header("4. Operator Strategies")
        add_dropdown("Selection Method:", self.var_sel_strategy, ["Tournament", "Roulette", "Best"])
        
        # Crossover dropdown — zmienny zależnie od reprezentacji
        self.cross_dropdown = add_dropdown(
            "Crossover Method:", self.var_cross_strategy, 
            ["OnePoint", "TwoPoint", "Uniform", "Discrete"],
            tag="crossover"
        )
        # Mutation dropdown
        self.mut_dropdown = add_dropdown(
            "Mutation Method:", self.var_mut_strategy,
            ["Marginal", "OnePoint", "TwoPoint"],
            tag="mutation"
        )
        # Inversion dropdown
        add_dropdown("Inversion Method:", self.var_inv_strategy, ["Classical"], tag="inversion")

        add_entry("BLX Alpha (α):", self.var_blx_alpha, tag="blx_alpha")
        add_entry("BLX Beta (β):", self.var_blx_beta, tag="blx_beta")

        # Ukryj pola BLX na starcie (domyślna reprezentacja: Binary)
        for tag in ("blx_alpha", "blx_beta"):
            lbl, entry, _ = self.dynamic_widgets[tag]
            lbl.grid_remove()
            entry.grid_remove()

        # Reaguj na zmianę metody krzyżowania
        if "crossover" in self.dynamic_widgets:
            _, opt, _ = self.dynamic_widgets["crossover"]
            opt.configure(command=self._on_crossover_change)

    def _on_crossover_change(self, value=None):
        """Pokazuje/ukrywa pola α i β zależnie od wybranej metody krzyżowania."""
        method = self.var_cross_strategy.get()
        show_alpha = method in ("BLX-alpha", "BLX-alpha-beta")
        show_beta = method == "BLX-alpha-beta"

        if "blx_alpha" in self.dynamic_widgets:
            lbl, entry, _ = self.dynamic_widgets["blx_alpha"]
            if show_alpha:
                lbl.grid()
                entry.grid()
            else:
                lbl.grid_remove()
                entry.grid_remove()

        if "blx_beta" in self.dynamic_widgets:
            lbl, entry, _ = self.dynamic_widgets["blx_beta"]
            if show_beta:
                lbl.grid()
                entry.grid()
            else:
                lbl.grid_remove()
                entry.grid_remove()

    def _on_representation_change(self, value=None):
        """Dynamicznie przełącza widoczne opcje operatorów zależnie od reprezentacji."""
        is_real = self.var_representation.get() == "Real"

        # Precision — ukryj dla real
        if "precision" in self.dynamic_widgets:
            lbl, entry, _ = self.dynamic_widgets["precision"]
            if is_real:
                lbl.grid_remove()
                entry.grid_remove()
            else:
                lbl.grid()
                entry.grid()

        # Inversion probability — ukryj dla real
        if "inv_prob" in self.dynamic_widgets:
            lbl, entry, _ = self.dynamic_widgets["inv_prob"]
            if is_real:
                lbl.grid_remove()
                entry.grid_remove()
            else:
                lbl.grid()
                entry.grid()

        # Inversion method — ukryj dla real
        if "inversion" in self.dynamic_widgets:
            lbl, opt, _ = self.dynamic_widgets["inversion"]
            if is_real:
                lbl.grid_remove()
                opt.grid_remove()
            else:
                lbl.grid()
                opt.grid()

        # Crossover — zmień opcje
        if "crossover" in self.dynamic_widgets:
            _, opt, _ = self.dynamic_widgets["crossover"]
            if is_real:
                real_options = ["Arithmetic", "Linear", "BLX-alpha", "BLX-alpha-beta", "Averaging"]
                opt.configure(values=real_options)
                self.var_cross_strategy.set("Arithmetic")
            else:
                bin_options = ["OnePoint", "TwoPoint", "Uniform", "Discrete"]
                opt.configure(values=bin_options)
                self.var_cross_strategy.set("TwoPoint")

        # Przy zmianie reprezentacji ukryj/pokaż pola BLX odpowiednio
        self._on_crossover_change()

        # Mutation — zmień opcje
        if "mutation" in self.dynamic_widgets:
            _, opt, _ = self.dynamic_widgets["mutation"]
            if is_real:
                real_options = ["Uniform", "Gaussian"]
                opt.configure(values=real_options)
                self.var_mut_strategy.set("Gaussian")
            else:
                bin_options = ["Marginal", "OnePoint", "TwoPoint"]
                opt.configure(values=bin_options)
                self.var_mut_strategy.set("OnePoint")


    def _validate_inputs(self) -> Optional[AlgorithmConfig]:
        try:
            num_vars = int(self.var_num_vars.get())
            b_min = float(self.var_bound_min.get())
            b_max = float(self.var_bound_max.get())
            prec = int(self.var_precision.get())
            pop = int(self.var_pop_size.get())
            ep = int(self.var_epochs.get())
            p_cross = float(self.var_cross_prob.get())
            p_mut = float(self.var_mut_prob.get())
            p_inv = float(self.var_inv_prob.get())

            is_real = self.var_representation.get() == "Real"
            representation = RepresentationType.REAL if is_real else RepresentationType.BINARY

            if num_vars < 1: raise ValueError("Variable count must be at least 1.")
            if b_min >= b_max: raise ValueError("Lower bound must be strictly less than upper bound.")
            if pop < 2: raise ValueError("Population size must be at least 2. (Evolution needs pairs!)")
            if ep < 1: raise ValueError("Number of epochs must be greater than 0.")
            for p, name in [(p_cross, 'Crossover'), (p_mut, 'Mutation')]:
                if not (0.0 <= p <= 1.0): 
                    raise ValueError(f"{name} probability must be a float between 0.0 and 1.0.")
            if not is_real:
                if not (0.0 <= p_inv <= 1.0):
                    raise ValueError("Inversion probability must be a float between 0.0 and 1.0.")

            bounds = [(b_min, b_max) for _ in range(num_vars)]
            func_pointer = AVAILABLE_FUNCTIONS[self.var_function.get()]
            target = OptimizationTarget.MAXIMIZE if self.var_target.get() == "MAXIMIZE" else OptimizationTarget.MINIMIZE

            # Selection (wspólna dla obu reprezentacji)
            s_map = {
                "Tournament": TournamentSelection(tournament_size=3),
                "Roulette": RouletteSelection(),
                "Best": BestSelection()
            }

            if is_real:
                # Krzyżowanie rzeczywiste
                c_map = {
                    "Arithmetic": ArithmeticCrossover(),
                    "Linear": LinearCrossover(
                        fitness_func=func_pointer,
                        is_maximization=(target == OptimizationTarget.MAXIMIZE),
                    ),
                    "BLX-alpha": BlendAlphaCrossover(alpha=float(self.var_blx_alpha.get())),
                    "BLX-alpha-beta": BlendAlphaBetaCrossover(alpha=float(self.var_blx_alpha.get()), beta=float(self.var_blx_beta.get())),
                    "Averaging": AverageCrossover(),
                }
                # Mutacja rzeczywista
                m_map = {
                    "Uniform": UniformMutation(),
                    "Gaussian": GaussianMutation(sigma=1.0),
                }
                inversion_strategy = None
            else:
                # Krzyżowanie binarne
                c_map = {
                    "OnePoint": OnePointCrossover(),
                    "TwoPoint": TwoPointCrossover(),
                    "Uniform": UniformCrossover(),
                    "Discrete": DiscreteCrossover(prob=0.5)
                }
                # Mutacja binarna
                m_map = {
                    "Marginal": MarginalMutation(),
                    "OnePoint": OnePointMutation(),
                    "TwoPoint": TwoPointMutation()
                }
                i_map = {
                    "Classical": ClassicalInversion()
                }
                inversion_strategy = i_map[self.var_inv_strategy.get()]

            return AlgorithmConfig(
                fitness_func=func_pointer, bounds=bounds, precision=prec, target=target,
                representation=representation,
                population_size=pop, epochs=ep, elitism=self.var_elitism.get(),
                cross_probability=p_cross, mutation_probability=p_mut, 
                inversion_probability=p_inv if not is_real else 0.0,
                selection_strategy=s_map[self.var_sel_strategy.get()],
                crossover_strategy=c_map[self.var_cross_strategy.get()],
                mutation_strategy=m_map[self.var_mut_strategy.get()],
                inversion_strategy=inversion_strategy if not is_real else None
            )

        except ValueError as e:
            messagebox.showerror("Validation Error", f"Incorrect input data detected:\n{str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Parser Error", f"Unexpected parsing error (e.g., characters instead of numbers):\n{str(e)}")
            return None

    def run_algorithm(self):
        config = self._validate_inputs()
        if config is None:
            return 
            
        self.btn_run.configure(state="disabled", text="CALCULATING... PLEASE WAIT")
        self.update()

        try:
            ga = GeneticAlgorithm(config)
            self.last_config = config
            self.last_results = ga.run()
            self._save_stats_to_csv()
            
            result_text = f"Evolution engine completed successfully after {config.epochs} epochs.\n"
            result_text += "-"*75 + "\n\n"
            
            result_text += f"[ REPRESENTATION: {'REAL' if config.representation == RepresentationType.REAL else 'BINARY'} ]\n\n"
            
            result_text += f"[ MATHEMATICAL TARGET VALUE (Y) ]\n"
            result_text += f"> {self.last_results['best_fitness_value']:.{config.precision}f}\n\n"
            
            result_text += f"[ BEST DECODED VARIABLES (X) ]\n"
            vars_str = ',  '.join([f"{v:.{config.precision}f}" for v in self.last_results['best_decoded_values']])
            result_text += f"> [ {vars_str} ]\n\n"
            
            if config.representation == RepresentationType.BINARY:
                result_text += f"[ WINNING GENETIC DNA (BITS) ]\n"
                dna_str = ''.join(map(str, self.last_results['best_chromosome_bits']))
                dna_lines = [dna_str[i:i+60] for i in range(0, len(dna_str), 60)]
                result_text += "> " + "\n  ".join(dna_lines) + "\n\n"
            else:
                result_text += f"[ WINNING GENE VALUES ]\n"
                genes_str = ',  '.join([f"{g:.6f}" for g in self.last_results['best_chromosome_genes']])
                result_text += f"> [ {genes_str} ]\n\n"
            
            result_text += "-"*75 + "\n"
            
            exec_time = self.last_results.get('execution_time', 0.0)
            if exec_time < 0.001:
                time_str = f"{exec_time * 1000:.2f} ms"
            else:
                time_str = f"{exec_time:.4f} s"
                
            result_text += f"Computation Time: {time_str}\n"

            self.results_textbox.configure(state="normal")
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", result_text)
            self.results_textbox.configure(state="disabled")
            
            self.btn_show_charts.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Execution Error", f"Genetic engine crashed during runtime:\n{e}")
        finally:
            self.btn_run.configure(state="normal", text="RUN EXPERIMENT")
    
    def _save_stats_to_csv(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        filename = os.path.join(project_root, "stats.csv")

        if not self.last_results or 'stats' not in self.last_results:
            return
            
        stats_list = self.last_results['stats']
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            writer.writerow([
                "Epoch", 
                "Best In Epoch", 
                "Worst In Epoch", 
                "Average In Epoch",
                "Std Dev In Epoch", 
                "Best Overall", 
                "Worst Overall"
            ])
            
            for epoch_number, stat_obj in enumerate(stats_list, start=1):
                writer.writerow([
                    epoch_number,
                    stat_obj.best,
                    stat_obj.worst,
                    stat_obj.avg,
                    stat_obj.std_dev,
                    stat_obj.best_overall,
                    stat_obj.worst_overall
                ])
                
        self.last_csv_path = filename

    # --- NAVIGATION ---
    def show_chart_view(self):
        self.main_container.grid_remove()
        self.charts_container.grid(row=0, column=0, sticky="nsew")
        self.charts_container.load_data(self.last_csv_path, self.last_results['best_fitness_value'], getattr(self, 'last_config', None))

    def show_main_view(self):
        self.charts_container.grid_remove()
        self.main_container.grid(row=0, column=0, sticky="nsew")

    def on_closing(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        self.quit()
        self.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()
