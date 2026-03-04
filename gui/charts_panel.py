import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os

class ChartsPanel(ctk.CTkFrame):
    def __init__(self, master, on_back_callback):
        super().__init__(master)
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.on_back_callback = on_back_callback
        self.last_csv_path = None
        self.last_target_value = 0.0
        self.last_config = None
        self.current_canvas = None

        self._build_ui()

    def _build_ui(self):
        # Top Bar
        self.top_bar = ctk.CTkFrame(self, height=60)
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        
        self.btn_back = ctk.CTkButton(
            self.top_bar, 
            text="⬅ BACK TO CONFIG", 
            width=150, 
            command=self.on_back_callback
        )
        self.btn_back.pack(side="left", padx=10, pady=10)

        # Chart selection
        self.chart_type_var = ctk.StringVar(value="Best (Epoch)")
        self.chart_selector = ctk.CTkSegmentedButton(
            self.top_bar, 
            values=["Best (Epoch)", "Worst (Epoch)", "Best Overall", "Worst Overall", "Average Value", "Function Shape"],
            variable=self.chart_type_var,
            command=self.render_chart
        )
        self.chart_selector.pack(side="right", padx=10, pady=10)

        # Matplotlib Canvas Container
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    def load_data(self, csv_path, target_value, config=None):
        self.last_csv_path = csv_path
        self.last_target_value = target_value
        self.last_config = config
        self.render_chart(self.chart_type_var.get())

    def render_chart(self, chart_type):
        if not self.last_csv_path or not os.path.exists(self.last_csv_path):
            return

        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            
        plt.close('all')

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#3b3b3b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')

        if chart_type == "Function Shape":
            if not self.last_config:
                return
            
            # Wyczyść poprzedni ax 2D i stwórz wyselekcjonowany subplot pod wykres (ewentualnie 3D)
            fig.clf()
            
            num_vars = len(self.last_config.bounds)
            if num_vars == 1:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#3b3b3b')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#555555')
                    
                b_min, b_max = self.last_config.bounds[0]
                X = np.linspace(b_min, b_max, 300)
                Y = [self.last_config.fitness_func([x]) for x in X]
                
                ax.plot(X, Y, color='#2196f3', linewidth=2)
                ax.set_title("Function Landscape (1D)")
                ax.set_xlabel("X1")
                ax.set_ylabel("Fitness (Y)")
                ax.grid(color='#555555', linestyle='-', linewidth=0.5, alpha=0.5)
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('#3b3b3b')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.zaxis.label.set_color('white')
                ax.title.set_color('white')
                
                # Zmieniamy panele siatki 3D pod ciemny tryb
                ax.xaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
                ax.yaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
                ax.zaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))

                b1_min, b1_max = self.last_config.bounds[0]
                b2_min, b2_max = self.last_config.bounds[1]
                
                # Resampling grid for performance vs quality
                x_vals = np.linspace(b1_min, b1_max, 50)
                y_vals = np.linspace(b2_min, b2_max, 50)
                X, Y = np.meshgrid(x_vals, y_vals)
                
                # Budowanie siatki Z (zmienna dla funkcji, uzupełnienie dla reszty wymiarow zerami jesli funkcji jest N-wymiarowa)
                Z = np.zeros_like(X)
                base_args = [0.0] * num_vars
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        args = base_args.copy()
                        args[0] = X[i, j]
                        args[1] = Y[i, j]
                        Z[i, j] = self.last_config.fitness_func(args)
                        
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
                ax.set_title("Function Landscape (3D Projection)")
                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_zlabel("Fitness (Y)")
        else:
            # RENDER STANDARDOWYCH WYKRESÓW STATYSTYK STATYCZNYCH Z HISTORII
            df = pd.read_csv(self.last_csv_path)
            epochs = df['Epoch']
    
            ax.axhline(y=self.last_target_value, color='r', linestyle='--', linewidth=1.5, label='Target Value')
    
            if chart_type == "Best (Epoch)":
                ax.plot(epochs, df['Best In Epoch'], label='Best In Epoch', color='#4caf50', linewidth=2)
                ax.set_title("Best Individual in Each Epoch")
            elif chart_type == "Worst (Epoch)":
                ax.plot(epochs, df['Worst In Epoch'], label='Worst In Epoch', color='#f44336', linewidth=2)
                ax.set_title("Worst Individual in Each Epoch")
            elif chart_type == "Best Overall":
                ax.plot(epochs, df['Best Overall'], label='Best Overall Found', color='#2196f3', linewidth=2)
                ax.set_title("Historic Best Individual Discovery Progress")
            elif chart_type == "Worst Overall":
                ax.plot(epochs, df['Worst Overall'], label='Worst Overall Found', color='#ff9800', linewidth=2)
                ax.set_title("Historic Worst Individual Tracking")
            elif chart_type == "Average Value":
                ax.plot(epochs, df['Average In Epoch'], label='Population Average', color='#9c27b0', linewidth=2)
                ax.set_title("Average Fitness of the Population over Time")
    
            ax.set_xlabel("Epoch (Generations)")
            ax.set_ylabel("Fitness Value")
            ax.margins(y=0.1)
            ax.legend(facecolor='#2b2b2b', edgecolor='#555555', labelcolor='white')
            ax.grid(color='#555555', linestyle='-', linewidth=0.5, alpha=0.5)

        fig.tight_layout()

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.current_canvas.draw()
        canvas_widget = self.current_canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
