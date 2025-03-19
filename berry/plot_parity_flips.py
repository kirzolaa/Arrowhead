#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Import the potential functions from the arrowhead matrix generator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generalized/example_use/arrowhead_matrix'))
from generate_4x4_arrowhead import ArrowheadMatrix4x4

def plot_parity_flips():
    """
    Create an infographic showing the parity flips for each eigenstate.
    """
    # Load the eigenvectors
    eigenvector_dir = 'generalized/example_use/arrowhead_matrix/results'
    eigenvectors = load_eigenvectors_from_directory(eigenvector_dir)
    
    if eigenvectors is None:
        print("Failed to load eigenvectors. Exiting.")
        return
    
    # Calculate parity flips
    parity_flips = calculate_parity_flips(eigenvectors)
    
    # Create directory for plots
    os.makedirs('parity_plots', exist_ok=True)
    
    # Plot parity flips
    plot_parity_flip_infographic(parity_flips)
    
    # Plot potentials
    plot_potential_functions()
    
    print("Plots saved to parity_plots/ directory.")

def load_eigenvectors_from_directory(directory):
    """
    Load eigenvectors from multiple .npy files stored for each theta value.
    Assumes files are named as 'eigenvectors_theta_XX.npy'.
    """
    import glob
    file_paths = sorted(glob.glob(os.path.join(directory, "eigenvectors_theta_*.npy")))
    
    if not file_paths:
        print("No eigenvector files found! Check directory and filenames.")
        return None

    eigenvectors_list = []
    for file in file_paths:
        eigenvectors = np.load(file)
        eigenvectors_list.append(eigenvectors)

    eigenvectors_array = np.array(eigenvectors_list)  # Shape: (num_theta, matrix_size, matrix_size)
    print(f"Loaded {len(file_paths)} eigenvector files. Shape: {eigenvectors_array.shape}")
    return eigenvectors_array

def calculate_parity_flips(eigenvectors):
    """
    Calculate parity flips for each eigenstate across theta values.
    """
    num_steps, num_states, _ = eigenvectors.shape
    parity_flips = np.zeros((num_states, num_steps), dtype=bool)
    
    # Create a copy of eigenvectors that we'll modify to account for parity changes
    adjusted_eigenvectors = eigenvectors.copy()
    
    for n in range(num_states):  # Loop over eigenstates
        for k in range(num_steps - 1):  # Loop over theta steps
            # Calculate overlap between consecutive eigenvectors
            overlap = np.vdot(adjusted_eigenvectors[k, :, n], eigenvectors[k + 1, :, n])
            
            # Check if we need to flip the parity to maintain continuity
            if np.real(overlap) < 0:  # Negative overlap suggests a parity flip is needed
                # Flip the sign of the eigenvector to maintain continuity
                adjusted_eigenvectors[k + 1, :, n] = -eigenvectors[k + 1, :, n]
                parity_flips[n, k + 1] = True
    
    return parity_flips

def plot_parity_flip_infographic(parity_flips):
    """
    Create an infographic showing the parity flips for each eigenstate.
    """
    num_states, num_steps = parity_flips.shape
    theta_values = np.linspace(0, 360, num_steps)
    
    # Create a figure with subplots for each eigenstate
    fig, axes = plt.subplots(num_states, 1, figsize=(12, 3*num_states))
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for n in range(num_states):
        ax = axes[n] if num_states > 1 else axes
        
        # Plot the parity value (1 for no flip, -1 for flip)
        parity_values = np.ones(num_steps)
        parity_values[parity_flips[n]] = -1
        
        # Plot the parity values
        ax.plot(theta_values, parity_values, '-', color=colors[n], linewidth=1.5)
        
        # Fill between the line and y=0 to highlight flips
        ax.fill_between(theta_values, parity_values, 0, 
                        where=(parity_values < 0), 
                        color=colors[n], alpha=0.3)
        
        # Count total flips
        total_flips = np.sum(parity_flips[n])
        
        # Add horizontal lines at y=1 and y=-1
        ax.axhline(y=1, color='black', linestyle='-', alpha=0.2)
        ax.axhline(y=-1, color='black', linestyle='-', alpha=0.2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.1)
        
        # Set y-axis limits and ticks
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 1])
        ax.set_yticklabels(['Flip', 'No Flip'])
        
        # Set x-axis limits and ticks
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        
        # Add title and labels
        ax.set_title(f'Eigenstate {n} Parity Flips (Total: {total_flips})', fontsize=12)
        if n == num_states - 1:
            ax.set_xlabel('Theta (degrees)', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parity_plots/parity_flips_infographic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary plot showing all eigenstates together
    plt.figure(figsize=(12, 6))
    
    for n in range(num_states):
        # Calculate cumulative parity (product of all previous parity values)
        cumulative_parity = np.cumprod(np.where(parity_flips[n], -1, 1))
        
        # Plot the cumulative parity
        plt.plot(theta_values, cumulative_parity, '-', color=colors[n], linewidth=2, 
                 label=f'Eigenstate {n} (Total flips: {np.sum(parity_flips[n])})')
    
    # Add horizontal lines
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.2)
    plt.axhline(y=-1, color='black', linestyle='-', alpha=0.2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.1)
    
    # Set axis limits and labels
    plt.xlim(0, 360)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Cumulative Parity', fontsize=12)
    plt.title('Cumulative Parity for Each Eigenstate', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('parity_plots/cumulative_parity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_potential_functions():
    """
    Plot the VX and VA potentials to visualize their shapes and differences.
    """
    # Create an instance of the ArrowheadMatrix4x4 class
    arrowhead = ArrowheadMatrix4x4()
    
    # Create a grid of points for plotting
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate VX potential for each point
    Z_vx = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            R = np.array([X[i, j], Y[i, j], 0])
            Z_vx[i, j] = arrowhead.potential_vx(R)
    
    # Calculate VA potential for each point
    Z_va = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            R = np.array([X[i, j], Y[i, j], 0])
            Z_va[i, j] = arrowhead.potential_va(R)
    
    # Plot VX potential - 3D surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_vx, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('VX Potential')
    ax.set_title('VX Potential Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('parity_plots/vx_potential_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot VA potential - 3D surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_va, cmap='plasma', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('VA Potential')
    ax.set_title('VA Potential Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('parity_plots/va_potential_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot both potentials as contours for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # VX contour
    contour1 = ax1.contourf(X, Y, Z_vx, 20, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('VX Potential Contour')
    fig.colorbar(contour1, ax=ax1)
    
    # VA contour
    contour2 = ax2.contourf(X, Y, Z_va, 20, cmap='plasma')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('VA Potential Contour')
    fig.colorbar(contour2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('parity_plots/potential_contours.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 1D slices of both potentials
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # X-axis slice (y=0)
    ax1.plot(x, Z_vx[:, len(y)//2], 'b-', linewidth=2, label='VX')
    ax1.plot(x, Z_va[:, len(y)//2], 'r-', linewidth=2, label='VA')
    ax1.set_xlabel('X (Y=0)')
    ax1.set_ylabel('Potential Value')
    ax1.set_title('Potential Functions Along X-axis (Y=0)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Y-axis slice (x=0)
    ax2.plot(y, Z_vx[len(x)//2, :], 'b-', linewidth=2, label='VX')
    ax2.plot(y, Z_va[len(x)//2, :], 'r-', linewidth=2, label='VA')
    ax2.set_xlabel('Y (X=0)')
    ax2.set_ylabel('Potential Value')
    ax2.set_title('Potential Functions Along Y-axis (X=0)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('parity_plots/potential_slices.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_parity_flips()
