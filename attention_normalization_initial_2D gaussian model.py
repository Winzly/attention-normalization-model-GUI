import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def gaussian(x, mu, sigma):
    """Simple 1D Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normalization_model(
    x, 
    stimulus_center, stimulus_width,
    attention_center, attention_width, 
    suppression_width, 
    sigma=1.0
):
    """
    x: 1D array (spatial positions)
    stimulus_center: center of stimulus
    stimulus_width: width of stimulus Gaussian
    attention_center: center of attention field
    attention_width: width of attention field
    suppression_width: std of Gaussian for local suppression
    sigma: semi-saturation constant
    """
    # Stimulus and attention profiles
    S = gaussian(x, stimulus_center, stimulus_width)
    A = gaussian(x, attention_center, attention_width)

    # Excitation: attention * stimulus
    E = A * S

    # Suppressive drive: local normalization using Gaussian filter
    I = gaussian_filter1d(E, suppression_width)

    # Normalized response: varies by x (local normalization)
    R = E / (sigma + I)

    # Plot everything
    plt.figure(figsize=(7, 4))
    plt.plot(x, S, label='Stimulus S(x)')
    plt.plot(x, A, label='Attention A(x)')
    plt.plot(x, E, label='Excitation E(x)', linestyle=':')
    plt.plot(x, I, label='Suppressive Drive I(x)', linestyle='--')
    plt.plot(x, R, label='Normalized Response R(x)', linewidth=2)
    plt.xlabel('Position x')
    plt.legend()
    plt.title('Normalization Model with Local Suppression')
    plt.tight_layout()
    plt.show()
    return R

# Example usage:
x = np.linspace(-50, 50, 100)
resp = normalization_model(
    x,
    stimulus_center=0,
    stimulus_width=40,
    attention_center=0,
    attention_width=30,
    suppression_width=15,   # Broad suppressive field
    sigma=1.0
)
