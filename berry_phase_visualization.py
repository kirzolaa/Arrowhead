#!/usr/bin/env python3
"""
Berry Phase Visualization Functions

This module provides functions for visualizing:
1. Eigenvalue evolution
2. Phase contributions
3. Parity flips
4. Overlap magnitudes
5. Eigenstate tracking
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

def plot_eigenvalue_evolution(theta_values, all_eigenvalues, output_dir='berry_phase_plots'):
    """
    Visualize the evolution of eigenvalues with theta.
    
    Parameters:
    -----------
    theta_values : numpy.ndarray
        Array of theta values
    all_eigenvalues : numpy.ndarray
        Array of eigenvalues with shape (num_steps, num_states)
    output_dir : str, optional
        Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Convert theta from radians to degrees for better readability
    theta_degrees = np.degrees(theta_values)
    
    for i in range(all_eigenvalues.shape[1]):
        plt.plot(theta_degrees, all_eigenvalues[:, i], 
                 label=f'Eigenvalue {i}', 
                 linewidth=2, 
                 marker='o', 
                 markersize=3)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title('Eigenvalue Evolution with Theta', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Add vertical lines at potential eigenvalue crossing points
    eigenvalue_differences = np.abs(np.diff(all_eigenvalues, axis=0))
    min_diff_indices = []
    
    # Find points where eigenvalues come close to each other
    for i in range(all_eigenvalues.shape[1] - 1):
        for j in range(i + 1, all_eigenvalues.shape[1]):
            differences = np.abs(all_eigenvalues[:, i] - all_eigenvalues[:, j])
            close_indices = np.where(differences < 0.1)[0]
            min_diff_indices.extend(close_indices)
    
    min_diff_indices = list(set(min_diff_indices))  # Remove duplicates
    
    for idx in min_diff_indices:
        if idx < len(theta_degrees):
            plt.axvline(x=theta_degrees[idx], color='r', linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'eigenvalue_evolution.png'), dpi=300)
    print(f"Saved eigenvalue evolution plot to {os.path.join(output_dir, 'eigenvalue_evolution.png')}")
    plt.close()

def plot_phase_contributions(theta_values, phase_contributions, output_dir='berry_phase_plots'):
    """
    Visualize phase contributions at each step.
    
    Parameters:
    -----------
    theta_values : numpy.ndarray
        Array of theta values
    phase_contributions : numpy.ndarray
        Array of phase contributions with shape (num_states, num_steps-1)
    output_dir : str, optional
        Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Convert theta from radians to degrees for better readability
    theta_degrees = np.degrees(theta_values[:-1])  # One less than theta_values
    
    for n in range(phase_contributions.shape[0]):
        plt.plot(theta_degrees, phase_contributions[n], 
                 label=f'Eigenstate {n}', 
                 linewidth=2)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Phase Contribution (radians)', fontsize=12)
    plt.title('Phase Contributions at Each Step', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Add horizontal lines at 0, π, and -π for reference
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=np.pi, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=-np.pi, color='k', linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'phase_contributions.png'), dpi=300)
    print(f"Saved phase contributions plot to {os.path.join(output_dir, 'phase_contributions.png')}")
    plt.close()
    
    # Also create a cumulative phase plot
    plt.figure(figsize=(12, 8))
    
    for n in range(phase_contributions.shape[0]):
        cumulative_phase = np.cumsum(phase_contributions[n])
        plt.plot(theta_degrees, cumulative_phase, 
                 label=f'Eigenstate {n}', 
                 linewidth=2)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Cumulative Phase (radians)', fontsize=12)
    plt.title('Cumulative Phase Evolution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'cumulative_phase.png'), dpi=300)
    print(f"Saved cumulative phase plot to {os.path.join(output_dir, 'cumulative_phase.png')}")
    plt.close()

def plot_parity_flips(theta_values, parity_flips, output_dir='berry_phase_plots'):
    """
    Visualize parity flips for each eigenstate.
    
    Parameters:
    -----------
    theta_values : numpy.ndarray
        Array of theta values
    parity_flips : numpy.ndarray
        Boolean array indicating parity flips with shape (num_states, num_steps)
    output_dir : str, optional
        Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_states = parity_flips.shape[0]
    
    # Convert theta from radians to degrees for better readability
    theta_degrees = np.degrees(theta_values)
    
    plt.figure(figsize=(12, 8))
    
    for n in range(num_states):
        # Get indices where parity flips occur
        flip_indices = np.where(parity_flips[n])[0]
        
        if len(flip_indices) > 0:
            plt.scatter(theta_degrees[flip_indices], [n] * len(flip_indices), 
                       marker='|', s=100, label=f'Eigenstate {n}' if n == 0 else None)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Eigenstate', fontsize=12)
    plt.title('Parity Flips by Eigenstate', fontsize=14)
    plt.yticks(range(num_states))
    plt.grid(True, axis='x')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'parity_flips.png'), dpi=300)
    print(f"Saved parity flips plot to {os.path.join(output_dir, 'parity_flips.png')}")
    plt.close()

def plot_overlap_magnitudes(theta_values, overlap_magnitudes, output_dir='berry_phase_plots'):
    """
    Visualize overlap magnitudes between consecutive eigenvectors.
    
    Parameters:
    -----------
    theta_values : numpy.ndarray
        Array of theta values
    overlap_magnitudes : numpy.ndarray
        Array of overlap magnitudes with shape (num_states, num_steps-1)
    output_dir : str, optional
        Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Convert theta from radians to degrees for better readability
    theta_degrees = np.degrees(theta_values[:-1])  # One less than theta_values
    
    for n in range(overlap_magnitudes.shape[0]):
        plt.plot(theta_degrees, overlap_magnitudes[n], 
                 label=f'Eigenstate {n}', 
                 linewidth=2)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Overlap Magnitude', fontsize=12)
    plt.title('Overlap Magnitudes Between Consecutive Eigenvectors', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.ylim(0, 1.05)  # Overlaps should be between 0 and 1
    plt.tight_layout()
    
    # Add a horizontal line at 1.0 for reference
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, 'overlap_magnitudes.png'), dpi=300)
    print(f"Saved overlap magnitudes plot to {os.path.join(output_dir, 'overlap_magnitudes.png')}")
    plt.close()

def plot_eigenstate_tracking(theta_values, tracking_indices, output_dir='berry_phase_plots'):
    """
    Visualize how eigenstates were tracked/reordered.
    
    Parameters:
    -----------
    theta_values : numpy.ndarray
        Array of theta values
    tracking_indices : numpy.ndarray
        Array of indices showing how eigenstates were reordered at each step
    output_dir : str, optional
        Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_steps, num_states = tracking_indices.shape
    
    # Convert theta from radians to degrees for better readability
    theta_degrees = np.degrees(theta_values)
    
    plt.figure(figsize=(12, 8))
    
    # Create a colormap for each original eigenstate index
    cmap = plt.cm.get_cmap('tab10', num_states)
    
    for n in range(num_states):
        plt.scatter(theta_degrees, tracking_indices[:, n], 
                   c=[cmap(n)] * num_steps, 
                   label=f'Tracked Eigenstate {n}',
                   s=30, alpha=0.7)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Original Eigenstate Index', fontsize=12)
    plt.title('Eigenstate Tracking/Reordering', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.yticks(range(num_states))
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'eigenstate_tracking.png'), dpi=300)
    print(f"Saved eigenstate tracking plot to {os.path.join(output_dir, 'eigenstate_tracking.png')}")
    plt.close()

def create_all_visualizations(results, theta_values, eigenvalues=None, output_dir='berry_phase_plots'):
    """
    Create all visualizations from the Berry phase calculation results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing Berry phase calculation results
    theta_values : numpy.ndarray
        Array of theta values
    eigenvalues : numpy.ndarray, optional
        Array of eigenvalues with shape (num_steps, num_states)
    output_dir : str, optional
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot eigenvalue evolution if eigenvalues are provided
    if eigenvalues is not None:
        plot_eigenvalue_evolution(theta_values, eigenvalues, output_dir)
    
    # Plot phase contributions
    plot_phase_contributions(theta_values, results['phase_contributions'], output_dir)
    
    # Plot parity flips
    plot_parity_flips(theta_values, results['parity_flips'], output_dir)
    
    # Plot overlap magnitudes
    plot_overlap_magnitudes(theta_values, results['overlap_magnitudes'], output_dir)
    
    # Plot eigenstate tracking
    plot_eigenstate_tracking(theta_values, results['tracking_indices'], output_dir)
    
    print(f"All visualizations created and saved to {output_dir}")
