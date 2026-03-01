import customtkinter as ctk
from tkinter import messagebox
from typing import Dict, Any, Optional

from utils.config import AlgorithmConfig, OptimizationTarget
from utils.functions import AVAILABLE_FUNCTIONS

from operators.selection import BestSelection, RouletteSelection, TournamentSelection
from operators.crossover import OnePointCrossover, TwoPointCrossover, UniformCrossover, DiscreteCrossover
from operators.mutation import MarginalMutation, OnePointMutation, TwoPointMutation
from core.genetic_algorithm import GeneticAlgorithm

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Classical Genetic Algorithm Editor")
        self.geometry("1100x800")
        self.minsize(900, 700)
        
        # We divide the window into two main columns
        # Column 0: Fixed Configuration Panel
        # Column 1: Flexible Results & Plots Panel
        self.grid_columnconfigure(0, weight=0, minsize=420)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: CONFIGURATION ---
        self.config_frame = ctk.CTkScrollableFrame(self, label_text="Algorithm Parameters", width=400)
        self.config_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # Config layout definitions
        self.config_frame.grid_columnconfigure(0, weight=1)
        self.config_frame.grid_columnconfigure(1, weight=1)

        self.init_variables()
        self.build_config_ui()

        # --- RIGHT PANEL: ACTIONS & RESULTS ---
        self.action_frame = ctk.CTkFrame(self)
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

        # Results area
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

        self.last_results = None

    def init_variables(self):
        """ Native TK variables for automatic updates and UI binding """
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

    def build_config_ui(self):
        row_id = 0

        def add_header(text):
            nonlocal row_id
            header = ctk.CTkLabel(self.config_frame, text=text, font=("Inter", 14, "bold"), text_color="#569cd6")
            header.grid(row=row_id, column=0, columnspan=2, pady=(20, 5), sticky="w")
            row_id += 1

        def add_entry(label_text, variable):
            nonlocal row_id
            ctk.CTkLabel(self.config_frame, text=label_text).grid(row=row_id, column=0, sticky="w", pady=5, padx=(5, 10))
            entry = ctk.CTkEntry(self.config_frame, textvariable=variable, width=140)
            entry.grid(row=row_id, column=1, sticky="e", pady=5, padx=5)
            row_id += 1

        def add_dropdown(label_text, variable, options):
            nonlocal row_id
            ctk.CTkLabel(self.config_frame, text=label_text).grid(row=row_id, column=0, sticky="w", pady=5, padx=(5, 10))
            opt = ctk.CTkOptionMenu(self.config_frame, variable=variable, values=options, width=140)
            opt.grid(row=row_id, column=1, sticky="e", pady=5, padx=5)
            row_id += 1

        # Section 1
        add_header("1. Environment & Objective")
        add_dropdown("Objective Function:", self.var_function, list(AVAILABLE_FUNCTIONS.keys()))
        add_dropdown("Optimization Target:", self.var_target, ["MINIMIZE", "MAXIMIZE"])
        add_entry("Number of Variables:", self.var_num_vars)
        add_entry("Lower Bound (Min):", self.var_bound_min)
        add_entry("Upper Bound (Max):", self.var_bound_max)
        add_entry("Number Precision:", self.var_precision)

        # Section 2
        add_header("2. Epochs & Engine")
        add_entry("Population Size:", self.var_pop_size)
        add_entry("Generations (Epochs):", self.var_epochs)
        ctk.CTkCheckBox(self.config_frame, text="Enable Elitism (Save the Best)", variable=self.var_elitism).grid(row=row_id, column=0, columnspan=2, sticky="w", pady=10, padx=5)
        row_id += 1
        
        # Section 3
        add_header("3. Evolutionary Probabilities")
        add_entry("Crossover Chance (Pc):", self.var_cross_prob)
        add_entry("Mutation Chance (Pm):", self.var_mut_prob)
        add_entry("Inversion Chance (Pi):", self.var_inv_prob)

        # Section 4
        add_header("4. Operator Strategies")
        add_dropdown("Selection Method:", self.var_sel_strategy, ["Tournament", "Roulette", "Best"])
        add_dropdown("Crossover Method:", self.var_cross_strategy, ["OnePoint", "TwoPoint", "Uniform", "Discrete"])
        add_dropdown("Mutation Method:", self.var_mut_strategy, ["Marginal", "OnePoint", "TwoPoint"])


    def _validate_inputs(self) -> Optional[AlgorithmConfig]:
        """ Professional silent validation - catches all errors before algorithm execution. """
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

            # Data Logical Checks
            if num_vars < 1: raise ValueError("Variable count must be at least 1.")
            if b_min >= b_max: raise ValueError("Lower bound must be strictly less than upper bound.")
            if pop < 2: raise ValueError("Population size must be at least 2. (Evolution needs pairs!)")
            if ep < 1: raise ValueError("Number of epochs must be greater than 0.")
            for p, name in [(p_cross, 'Crossover'), (p_mut, 'Mutation'), (p_inv, 'Inversion')]:
                if not (0.0 <= p <= 1.0): 
                    raise ValueError(f"{name} probability must be a float between 0.0 and 1.0.")

            # Instantiating the algorithm parameters 
            bounds = [(b_min, b_max) for _ in range(num_vars)]
            func_pointer = AVAILABLE_FUNCTIONS[self.var_function.get()]
            target = OptimizationTarget.MAXIMIZE if self.var_target.get() == "MAXIMIZE" else OptimizationTarget.MINIMIZE

            # Dependency Injection Map for selected GUI Options
            s_map = {
                "Tournament": TournamentSelection(tournament_size=3),
                "Roulette": RouletteSelection(),
                "Best": BestSelection()
            }
            c_map = {
                "OnePoint": OnePointCrossover(),
                "TwoPoint": TwoPointCrossover(),
                "Uniform": UniformCrossover(),
                "Discrete": DiscreteCrossover(prob=0.5)
            }
            m_map = {
                "Marginal": MarginalMutation(),
                "OnePoint": OnePointMutation(),
                "TwoPoint": TwoPointMutation()
            }

            return AlgorithmConfig(
                fitness_func=func_pointer, bounds=bounds, precision=prec, target=target,
                population_size=pop, epochs=ep, elitism=self.var_elitism.get(),
                cross_probability=p_cross, mutation_probability=p_mut, inversion_probability=p_inv,
                selection_strategy=s_map[self.var_sel_strategy.get()],
                crossover_strategy=c_map[self.var_cross_strategy.get()],
                mutation_strategy=m_map[self.var_mut_strategy.get()]
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
            self.last_results = ga.run()
            
            # Formatted Results String
            result_text = f"Evolutiom engine completed successfully after {config.epochs} epochs.\n"
            result_text += "-"*75 + "\n\n"
            
            result_text += f"[ MATHEMATICAL TARGET VALUE (Y) ]\n"
            result_text += f"> {self.last_results['best_fitness_value']:.10f}\n\n"
            
            result_text += f"[ BEST DECODED VARIABLES (X) ]\n"
            vars_str = ',  '.join([f"{v:.5f}" for v in self.last_results['best_decoded_values']])
            result_text += f"> [ {vars_str} ]\n\n"
            
            result_text += f"[ WINNING GENETIC DNA (BITS) ]\n"
            dna_str = ''.join(map(str, self.last_results['best_chromosome']))
            # Slicing the long binary string to fit nicely in the window
            dna_lines = [dna_str[i:i+60] for i in range(0, len(dna_str), 60)]
            result_text += "> " + "\n  ".join(dna_lines) + "\n\n"
            
            result_text += "-"*75 + "\n"
            result_text += "Data stored in application memory. Ready for plotting.\n"

            self.results_textbox.configure(state="normal")
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", result_text)
            self.results_textbox.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Execution Error", f"Genetic engine crashed during runtime:\n{e}")
        finally:
            self.btn_run.configure(state="normal", text="RUN EXPERIMENT")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()
