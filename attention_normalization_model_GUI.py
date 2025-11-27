import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter

"""
GUI to explore a 2D normalization model of attention (Reynolds & Heeger).
Comments are written for future researchers who may extend or change the model,
parameters, or GUI structure.
"""

# ====================================================================================================

def gaussian2d(x, theta, mu_x, mu_theta, sd_x, sd_theta):
    """
    2D Gaussian over spatial position × orientation.
    Used to define:
    - stimulus profiles
    - attention fields
    - orientation tuning
    The wrapped angular difference ensures correct periodic orientation behavior (180° symmetry).
    """
    X, T = np.meshgrid(x, theta, indexing='ij')
    dtheta = ((T - mu_theta + 90) % 180) - 90
    return np.exp(-0.5 * (((X - mu_x) / sd_x) ** 2 + (dtheta / sd_theta) ** 2))

def normalization_model_2d(
    x, theta,
    stim1_x, stim1_x_size, stim1_theta, stim1_contrast,
    stim2_x, stim2_x_size, stim2_theta, stim2_contrast,
    attn_x, attn_sd_x, attn_theta, attn_sd_theta,
    suppression_sd_x, suppression_sd_theta,
    theta_tuning,
    sigma=1.0
):
    """Core normalization model (Reynolds & Heeger conceptual structure):
    - Builds two stimuli in 2D (position × orientation)
    - Applies multiplicative attention gain
    - Computes suppressive pool via Gaussian filtering
    - Produces final normalized response:  E / (σ + I)

    Returns intermediates (S1, S2, S, A, E, I, R) for GUI visualization.
    """
    """Modèle de normalisation pour deux stimuli avec nouveaux paramètres"""
    # Créer les deux stimuli
    S1 = stim1_contrast * gaussian2d(x, theta, stim1_x, stim1_theta, stim1_x_size, theta_tuning)
    S2 = stim2_contrast * gaussian2d(x, theta, stim2_x, stim2_theta, stim2_x_size, theta_tuning)
    S = S1 + S2
    
    # Champ attentionnel
    A = 1 + gaussian2d(x, theta, attn_x, attn_theta, attn_sd_x, attn_sd_theta)
    
    # Drive excitateur
    E = A * S
    
    # Pool de suppression
    I = gaussian_filter(E, [suppression_sd_x, suppression_sd_theta])
    
    # Réponse normalisée
    R = E / (sigma + I)
    
    return S1, S2, S, A, E, I, R

class Normalization2DGUI:
    def __init__(self, root):
        """
        Build the entire GUI layout:

        LEFT:  vertical scrolling slider panel
               → controls every parameter of the normalization model

        RIGHT: matplotlib figure containing:
               - 4 heatmaps (Stimulus, Attention, Suppression, Response)
               - 1 spatial profile at θ = 0°

        Sliders update the model continuously in real-time.
        """
        self.root = root
        self.root.title("Modèle de Normalisation d'Attention 2D - Reynolds & Heeger")
        self.root.geometry("1600x900")

        # --- Frame principal ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left frame: Sliders (now on the left) ---
        self.center_frame = tk.Frame(main_frame, width=450)
        # Make the center_frame expand vertically and horizontally to fit all sliders
        self.center_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y, expand=True)
        
        canvas_frame = tk.Frame(self.center_frame)
        # Make canvas_frame expand fully
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for sliders, fill both directions and expand
        self.slider_canvas = tk.Canvas(canvas_frame, width=420)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.slider_canvas.yview)
        self.scrollable_frame = tk.Frame(self.slider_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.slider_canvas.configure(scrollregion=self.slider_canvas.bbox("all"))
        )

        self.slider_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.slider_canvas.configure(yscrollcommand=scrollbar.set)

        # Make sure canvas fills both and expands
        self.slider_canvas.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind du scroll de la souris
        def _on_mousewheel(event):
            self.slider_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.slider_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Container for all Tkinter slider widgets.
        # Allows bulk read/update/reset of all parameters.
        self.sliders = {}

        # Grilles spatiales
        self.x = np.linspace(-50, 50, 101)
        self.theta = np.linspace(-90, 90, 91)
        # Slider specification list.
        # Each entry defines: (label, min, max, default)
        # Structured into sections so researchers can easily add/remove parameters.
        slider_specs = [
            # Stimulus 1
            ("=== STIMULUS 1 ===", None, None, None),
            ("Position X", -50, 50, -25),
            ("Taille (σₓ)", 1, 20, 2),
            ("Orientation (θ)", -90, 90, 0),
            ("Contraste", 0, 2, 1.0),

            ("", None, None, None),

            # Stimulus 2
            ("=== STIMULUS 2 ===", None, None, None),
            ("Position X 2", -50, 50, 25),
            ("Taille (σₓ) 2", 1, 20, 2),
            ("Orientation (θ) 2", -90, 90, 0),
            ("Contraste 2", 0, 2, 1.0),
            
            ("", None, None, None),
            
            # Attention
            ("=== ATTENTION ===", None, None, None),
            ("Centre X", -50, 50, 25),
            ("Étendue (σₓ)", 1, 50, 10),
            ("Centre Orientation (θ)", -90, 90, 0),
            ("Étendue Orientation (σθ)", 1, 180, 180),
            
            ("", None, None, None),
            
            # Suppression
            ("=== SUPPRESSION ===", None, None, None),
            ("Largeur Spatiale", 1, 20, 4),
            ("Largeur Orientation", 1, 90, 18),
            
            ("", None, None, None),
            
            # Tuning & Normalisation
            ("=== TUNING & NORMALISATION ===", None, None, None),
            ("Largeur de Tuning (θ)", 1, 90, 25),
            ("Constante σ", 0.01, 5, 1.0),
        ]
        
        row = 0
        for spec in slider_specs:
            if len(spec) == 4:
                label, mn, mx, val = spec
            else:
                label = spec[0]
                mn = mx = val = None
            
            if mn is None:
                if label:
                    header = tk.Label(self.scrollable_frame, text=label, font=("Arial", 10, "bold"), fg="navy")
                    header.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))
                else:
                    tk.Frame(self.scrollable_frame, height=10).grid(row=row, column=0, columnspan=2)
            else:
                lbl = tk.Label(self.scrollable_frame, text=label, font=("Arial", 9))
                lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                slider = tk.Scale(
                    self.scrollable_frame, 
                    from_=mn, 
                    to=mx, 
                    resolution=0.1, 
                    orient=tk.HORIZONTAL, 
                    length=250,
                    command=lambda val: self.update_plots()
                )

                slider.set(val)
                slider.grid(row=row, column=1, padx=5, pady=2, sticky="e")
                self.sliders[label] = slider
            
            row += 1

        # Reset button
        reset_button = tk.Button(self.center_frame, text="Reset", font=("Arial", 11, "bold"), command=self.reset_sliders)
        reset_button.pack(pady=10)

        # --- Right frame: Figure + Canvas (now on the right) ---
        self.right_frame = tk.Frame(main_frame)
        self.right_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Figure for heatmaps and curves
        self.fig = plt.Figure(figsize=(12, 8))

        # Heatmaps en haut (4 colonnes)
        self.ax_stimulus = self.fig.add_subplot(3, 4, 1)
        self.ax_attention = self.fig.add_subplot(3, 4, 2)
        self.ax_suppression = self.fig.add_subplot(3, 4, 3)
        self.ax_response = self.fig.add_subplot(3, 4, 4)
        
        # Courbes en bas (sur toute la largeur)
        self.ax_curves = self.fig.add_subplot(3, 1, 2)

        # Canvas Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Variables pour les colorbars
        self.cbar_stimulus = None
        self.cbar_attention = None
        self.cbar_suppression = None
        self.cbar_response = None

        # Premier plot
        self.update_plots()
        
    def get_slider_values(self):
        """Return a dictionary mapping slider labels → current numerical values."""
        return {label: slider.get() for label, slider in self.sliders.items()}

    def reset_sliders(self):
        """
        Reset all sliders to default values.
        Useful for experiments where the model must return to a clean baseline.
        """
        initial_values = {
            "Position X": -25,
            "Taille (σₓ)": 2,
            "Orientation (θ)": 0,
            "Contraste": 1.0,
            "Position X 2": 25,
            "Taille (σₓ) 2": 2,
            "Orientation (θ) 2": 0,
            "Contraste 2": 1.0,
            "Centre X": 25,
            "Étendue (σₓ)": 10,
            "Centre Orientation (θ)": 0,
            "Étendue Orientation (σθ)": 180,
            "Largeur Spatiale": 4,
            "Largeur Orientation": 18,
            "Largeur de Tuning (θ)": 25,
            "Constante σ": 1.0
        }
        for label, val in initial_values.items():
            if label in self.sliders:
                self.sliders[label].set(val)
        self.update_plots()

    def update_plots(self):
        """Main update routine.
        1. Reads all slider values
        2. Recomputes the normalization model
        3. Updates:
            - the four heatmaps (stimulus / attention / suppression / response)
            - the spatial response profile at θ = 0°
        Called automatically whenever any slider moves.
        """
        vals = self.get_slider_values()
        
        # Extraire les paramètres
        stim1_x = vals["Position X"]
        stim1_x_size = vals["Taille (σₓ)"]
        stim1_theta = vals["Orientation (θ)"]
        stim1_contrast = vals["Contraste"]
        
        stim2_x = vals["Position X 2"]
        stim2_x_size = vals["Taille (σₓ) 2"]
        stim2_theta = vals["Orientation (θ) 2"]
        stim2_contrast = vals["Contraste 2"]
        
        attn_x = vals["Centre X"]
        attn_sd_x = vals["Étendue (σₓ)"]
        attn_theta = vals["Centre Orientation (θ)"]
        attn_sd_theta = vals["Étendue Orientation (σθ)"]

        suppression_sd_x = vals["Largeur Spatiale"]
        suppression_sd_theta = vals["Largeur Orientation"]

        theta_tuning = vals["Largeur de Tuning (θ)"]
        sigma = vals["Constante σ"]
        
        # Calculer le modèle
        S1, S2, S, A, E, I, R = normalization_model_2d(
            self.x, self.theta,
            stim1_x, stim1_x_size, stim1_theta, stim1_contrast,
            stim2_x, stim2_x_size, stim2_theta, stim2_contrast,
            attn_x, attn_sd_x, attn_theta, attn_sd_theta,
            suppression_sd_x, suppression_sd_theta,
            theta_tuning,
            sigma
        )
        
        # Arguments pour les heatmaps
        extent = [self.x[0], self.x[-1], self.theta[0], self.theta[-1]]
        
        # === HEATMAP 1: STIMULUS DRIVE ===
        self.ax_stimulus.clear()
        im_stimulus = self.ax_stimulus.imshow(
            S.T, 
            origin='lower', 
            aspect='auto',
            extent=extent,
            cmap='gray',
            vmin=0,
            vmax=2.0
        )
        self.ax_stimulus.set_title('Stimulus Drive', fontsize=10, fontweight='bold')
        self.ax_stimulus.set_xlabel('Position', fontsize=8)
        self.ax_stimulus.set_ylabel('Orientation', fontsize=8)
        self.ax_stimulus.tick_params(labelsize=7)
        
        if self.cbar_stimulus is None:
            self.cbar_stimulus = self.fig.colorbar(im_stimulus, ax=self.ax_stimulus, fraction=0.046, pad=0.04)
            self.cbar_stimulus.ax.tick_params(labelsize=7)
        else:
            self.cbar_stimulus.update_normal(im_stimulus)
        
        # === HEATMAP 2: ATTENTION FIELD ===
        self.ax_attention.clear()
        im_attention = self.ax_attention.imshow(
            A.T, 
            origin='lower', 
            aspect='auto',
            extent=extent,
            cmap='gray',
            vmin=1,
            vmax=2.0
        )
        self.ax_attention.set_title('Attention Field', fontsize=10, fontweight='bold')
        self.ax_attention.set_xlabel('Position', fontsize=8)
        self.ax_attention.set_ylabel('Orientation', fontsize=8)
        self.ax_attention.tick_params(labelsize=7)
        
        if self.cbar_attention is None:
            self.cbar_attention = self.fig.colorbar(im_attention, ax=self.ax_attention, fraction=0.046, pad=0.04)
            self.cbar_attention.ax.tick_params(labelsize=7)
        else:
            self.cbar_attention.update_normal(im_attention)
        
        # === HEATMAP 3: SUPPRESSIVE DRIVE ===
        self.ax_suppression.clear()
        im_suppression = self.ax_suppression.imshow(
            I.T, 
            origin='lower', 
            aspect='auto',
            extent=extent,
            cmap='gray',
            vmin=0,
            vmax=np.percentile(I, 99)
        )
        self.ax_suppression.set_title('Suppressive Drive', fontsize=10, fontweight='bold')
        self.ax_suppression.set_xlabel('Position', fontsize=8)
        self.ax_suppression.set_ylabel('Orientation', fontsize=8)
        self.ax_suppression.tick_params(labelsize=7)
        
        if self.cbar_suppression is None:
            self.cbar_suppression = self.fig.colorbar(im_suppression, ax=self.ax_suppression, fraction=0.046, pad=0.04)
            self.cbar_suppression.ax.tick_params(labelsize=7)
        else:
            self.cbar_suppression.update_normal(im_suppression)
        
        # === HEATMAP 4: NORMALIZED RESPONSE ===
        self.ax_response.clear()
        im_response = self.ax_response.imshow(
            R.T, 
            origin='lower', 
            aspect='auto',
            extent=extent,
            cmap='gray',
            vmin=0,
            vmax=np.percentile(R, 99)
        )
        self.ax_response.set_title('Normalized Response', fontsize=10, fontweight='bold')
        self.ax_response.set_xlabel('Position', fontsize=8)
        self.ax_response.set_ylabel('Orientation', fontsize=8)
        self.ax_response.tick_params(labelsize=7)
        
        if self.cbar_response is None:
            self.cbar_response = self.fig.colorbar(im_response, ax=self.ax_response, fraction=0.046, pad=0.04)
            self.cbar_response.ax.tick_params(labelsize=7)
        else:
            self.cbar_response.update_normal(im_response)
        
        # === COURBES (profils à l'orientation centrale) ===
        self.ax_curves.clear()
        
        # Trouver l'index de l'orientation centrale (θ=0)
        theta_idx = np.argmin(np.abs(self.theta - 0))
        
        # Extraire les profils à θ=0
        s1_profile = S1[:, theta_idx]
        s2_profile = S2[:, theta_idx]
        s_profile = S[:, theta_idx]
        a_profile = A[:, theta_idx]
        r_profile = R[:, theta_idx]
        e_profile = E[:, theta_idx]
        i_profile = I[:, theta_idx]
        
        # Limiter la plage x pour avoir des gaussiennes moins étalées
        x_range_mask = (self.x >= -30) & (self.x <= 30)
        x_plot = self.x[x_range_mask]
        
        self.ax_curves.plot(x_plot, s1_profile[x_range_mask], 'b-', linewidth=2.5, label='Stimulus 1')
        self.ax_curves.plot(x_plot, s2_profile[x_range_mask], 'c-', linewidth=2.5, label='Stimulus 2')
        self.ax_curves.plot(x_plot, s_profile[x_range_mask], color='blue', linewidth=2, label='Stimulus Drive (S)', linestyle='--')
        self.ax_curves.plot(x_plot, a_profile[x_range_mask], 'g-', linewidth=2.5, label='Attention Field')
        self.ax_curves.plot(x_plot, i_profile[x_range_mask], 'orange', linewidth=2, label='Suppressive Drive', linestyle=':')
        self.ax_curves.plot(x_plot, r_profile[x_range_mask], 'r-', linewidth=3, label='Neural Response')
        
        self.ax_curves.set_xlabel('Spatial Position', fontsize=11, fontweight='bold')
        self.ax_curves.set_ylabel('Response Strength', fontsize=11, fontweight='bold')
        self.ax_curves.set_title('Normalization Model of Attention (Profile at θ=0°)', fontsize=12, fontweight='bold')
        self.ax_curves.legend(loc='upper right', fontsize=9, framealpha=0.9)
        self.ax_curves.grid(True, alpha=0.3, linestyle='--')
        self.ax_curves.set_xlim([-30, 30])
        
        # Calculer ylim dynamiquement
        all_values = np.concatenate([
            a_profile[x_range_mask], 
            r_profile[x_range_mask], 
            s_profile[x_range_mask]
        ])
        ymax = np.max(all_values) * 1.1 if np.max(all_values) > 0 else 2
        self.ax_curves.set_ylim([0, ymax])
        
        self.fig.tight_layout()
        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = Normalization2DGUI(root)
    root.mainloop()