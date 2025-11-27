# Importing packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



def gaussian2d(x, theta, mu_x, mu_theta, sd_x, sd_theta):
    '''
    Generate a 2D Gaussian matrix over space (x) and orientation (theta).
    
    Arguments:
        x: 1D array of spatial positions (neuron preferred position).
        theta: 1D array of orientation preferences (neuron preferred orientation).
        mu_x: stimulus or attention centre in space (centre of the Gaussian)
        mu_theta: stimulus or attention centre in orientation.
        sd_x: spatial standard deviation (width of the Gaussian)
        sd_theta: orientation standard deviation.

    Returns:
        G: 2D numpy array, shape (len(x), len(theta)), value at each (x, theta) pair.
    '''
    
    #Create 2D grids of all x and theta combinations (population grid)
    X, T = np.meshgrid(x, theta, indexing='ij')
    #Circular difference for orientation (wrap between -90 and 90 degrees)
    dtheta = ((T - mu_theta +90) % 180) - 90
    #Calculate 2D Gaussian value for each neuron
    return np.exp(-0.5 * (((X - mu_x) / sd_x) ** 2 + (dtheta / sd_theta) ** 2))


def normalization_model_2d(
    x, theta,
    #stimulus 1 parameters
    stim1_x, stim1_x_size, stim1_theta, stim1_contrast,
    #stimulus 2 parameters
    stim2_x, stim2_x_size, stim2_theta, stim2_contrast,
    #attention field parameters
    attn_x, attn_sd_x, attn_theta, attn_sd_theta,
    #suppression widths (standard deviation)
    suppression_sd_x, suppression_sd_theta,
    #tuning widht 
    theta_tuning,
    #semi-saturation constant
    sigma=1,

):
    '''
    Implements Reynolds & Heeger normalization model of attention 
    (2D: position x orientation.
     
     Arguments:
        stim1_x: position of stimulus 1 in space.
        stim1_x_size: spatial size of stimulus 1 (Gaussian width).
        stim1_theta: orientation of stimulus 1.
       
        stim1_contrast: contrast of stimulus 1.

        stim2_x: position of stimulus 2 in space.
        stim2_x_size: spatial size of stimulus 2.
        stim2_theta: orientation of stimulus 2.
        stim2_theta_size: orientation bandwidth of stimulus 2.
        stim2_contrast: contrast of stimulus 2.

        attn_x: attention centre in space.
        attn_sd_x: attention spatial spread.
        attn_theta: attention centre in orientation.
        attn_sd_theta: attention field orientation spread.
        
        theta_tuning:tuning width

        suppression_sd_x: suppression pool spread in space.
        suppression_sd_theta: suppression pool spread in orientation.

        sigma: semi-saturation constant (prevents division by zero, sets curve steepness).
         
         Returns: 
             R: 2D array, population response after normalization
             '''
    
    # STIMULUS DRIVE: spatial x orientation Gaussian, scaled by contrast
    #Sum of the two stimuli
    
    S1 = stim1_contrast * gaussian2d(x, theta, stim1_x, stim1_theta, stim1_x_size, theta_tuning)
    S2 = stim2_contrast * gaussian2d(x, theta, stim2_x, stim2_theta, stim2_x_size, theta_tuning)
   
    #Total stimulus drive is the sum of the two stimuli
    S = S1 + S2
    
    # ATTENTION FIELD (shape and spread over neuron prefs)
    A = 1 + gaussian2d(x, theta, attn_x, attn_theta, attn_sd_x, attn_sd_theta)
    
    # EXCITATORY DRIVE: attention * stimulus, at each population point (neuron)
    E = A * S
    
    # SUPPRESSIVE DRIVE: local normalization across x and theta
    I = gaussian_filter(E, [suppression_sd_x, suppression_sd_theta])
   
    # NORMALIZED RESPONSE
    R = E / (sigma + I)

#PLOTTING - plot as heatmap
    #Prepare shared axis properties for plotting 
    plt.figure(figsize=(10, 8))
    imargs = dict(origin='lower', aspect='auto',
                  extent=[x[0], x[-1], theta[0], theta[-1]], 
                  cmap = 'gray', clim = [0,2])
    
    #Plot the raw stimulus drive
    plt.subplot(2, 2, 1)
    plt.imshow(S.T, **imargs)
    plt.title('Stimulus Drive (x,θ)')
    plt.xlabel('RF centre')
    plt.ylabel('Orientation θ pref')
    plt.xticks([])      #remove scales
    plt.yticks([])

    #Plot the attention field
    plt.subplot(2, 2, 2)
    plt.imshow(A.T, **imargs)
    plt.title('Attention Field (x,θ)')
    plt.xlabel('RF centre')
    plt.ylabel('Orientation θ pref')
    plt.xticks([])      #remove scales
    plt.yticks([])

    #Plot the suppressive drive
    plt.subplot(2, 2, 3)
    plt.imshow(I.T, **imargs)
    plt.title('Suppressive Drive I(x,θ)')
    plt.xlabel('RF centre')
    plt.ylabel('Orientation θ pref')
    plt.xticks([])      # remove scales
    plt.yticks([])
    
    #Plot the normalized response
    plt.subplot(2, 2, 4)
    plt.imshow(R.T, **imargs)
    plt.title('Normalized Response R(x,θ)')
    plt.xlabel('RF centre')
    plt.ylabel('Orientation θ pref')
    plt.xticks([])      #remove scales
    plt.yticks([])

    plt.tight_layout()
    plt.show()
    return R


# EXAMPLE USAGE AND PARAMETERS
x = np.linspace(-50, 50, 101)  #Possible positions (space)
theta = np.linspace(-90, 90, 91)  #Possible orientations (degrees)

resp = normalization_model_2d(
    x, theta,
    # Stimulus 1 (e.g., right side, vertical)
    stim1_x=25, stim1_x_size=2,
    stim1_theta=0, 
    stim1_contrast=1.0,
    # Stimulus 2 (e.g., left side, same orientation)
    stim2_x=-25, stim2_x_size=2,
    stim2_theta=0,
    stim2_contrast=1.0,
    # Attention parameters (e.g., focus toward stimulus 1)
    attn_x=25, attn_sd_x=10,
    attn_theta=0, attn_sd_theta=180,
    # Suppression field size
    suppression_sd_x=4, suppression_sd_theta=18,
    #tuning width
    theta_tuning = 25,
    # Normalization constant
    sigma=1,
)
