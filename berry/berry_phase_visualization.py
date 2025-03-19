#!/usr/bin/env python3
"""
Berry Phase Visualization

This module provides functions to create visualizations for Berry phase calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_berry_phases(berry_phases, output_dir):
    """Plot Berry phases for each eigenstate."""
    plt.figure(figsize=(10, 6))
    
    # Handle both dictionary and numpy array formats
    if hasattr(berry_phases, 'keys'):
        # Dictionary format
        eigenstates = sorted(berry_phases.keys())
        phases = [berry_phases[e] for e in eigenstates]
    else:
        # Numpy array format
        eigenstates = list(range(len(berry_phases)))
        phases = berry_phases
    
    plt.bar(eigenstates, phases, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('Eigenstate')
    plt.ylabel('Berry Phase')
    plt.title('Berry Phases for Each Eigenstate')
    plt.xticks(eigenstates)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/berry_phases.png", dpi=300)
    plt.close()

def plot_parity_flips(parity_flips, output_dir):
    """Plot parity flips for each eigenstate."""
    plt.figure(figsize=(10, 6))
    
    # Handle both dictionary and numpy array formats
    if hasattr(parity_flips, 'keys'):
        # Dictionary format
        eigenstates = sorted(parity_flips.keys())
        flips = [parity_flips[e] for e in eigenstates]
    else:
        # Numpy array format
        eigenstates = list(range(len(parity_flips)))
        flips = parity_flips
    
    # Use a different color for eigenstate 3 to highlight it
    colors = ['blue' if e != 3 else 'green' for e in eigenstates]
    
    plt.bar(eigenstates, flips, color=colors, alpha=0.7)
    
    plt.xlabel('Eigenstate')
    plt.ylabel('Number of Parity Flips')
    plt.title('Parity Flips for Each Eigenstate')
    plt.xticks(eigenstates)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for each bar
    for i, v in enumerate(flips):
        plt.text(eigenstates[i], v + 0.5, str(v), ha='center')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parity_flips.png", dpi=300)
    plt.close()

def plot_eigenstate_vs_theta(eigenstate_data, output_dir):
    """Plot eigenstate values vs theta for all eigenstates."""
    if not eigenstate_data:
        return
    
    # Create combined plot with cleaner visualization
    plt.figure(figsize=(12, 8))
    
    # Use distinct colors for each eigenstate
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    for eigenstate, data in sorted(eigenstate_data.items()):
        theta = data[:, 0]
        values = data[:, 1]
        color_idx = eigenstate % len(colors)
        style_idx = eigenstate % len(linestyles)
        plt.plot(theta, values, 
                 color=colors[color_idx], 
                 linestyle=linestyles[style_idx],
                 linewidth=2,
                 label=f'Eigenstate {eigenstate}')
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title('Eigenvalues vs Theta', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    
    # Add vertical lines at 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        plt.axvline(x=angle, color='gray', linestyle='--', alpha=0.5)
    
    # Add a text box with information about the plot
    info_text = (
        "This plot shows the eigenvalues of the system\n"
        "as a function of the parameter θ (theta).\n\n"
        "Key features to observe:\n"
        "- Crossing/avoided crossing points\n"
        "- Periodicity of eigenvalues\n"
        "- Symmetry around specific θ values"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eigenvalues_vs_theta.png", dpi=300)
    plt.close()

def plot_eigenstate_degeneracy(eigenstate_data, output_dir):
    """Plot the degeneracy between eigenstates."""
    if 1 not in eigenstate_data or 2 not in eigenstate_data:
        print("Warning: Eigenstate 1 or 2 not found in data, skipping degeneracy check")
        return
    
    # Calculate the difference between eigenstate 1 and 2
    data1 = eigenstate_data[1]
    data2 = eigenstate_data[2]
    
    # Ensure the theta values match
    if len(data1) != len(data2) or not np.allclose(data1[:, 0], data2[:, 0]):
        print("Warning: Theta values don't match between eigenstates 1 and 2")
        return
    
    theta = data1[:, 0]
    values1 = data1[:, 1]
    values2 = data2[:, 1]
    diff_12 = np.abs(values1 - values2)
    
    # Find points of near-degeneracy (where difference is very small)
    threshold = 0.01  # Threshold for considering eigenvalues nearly degenerate
    near_degenerate_points = theta[diff_12 < threshold]
    near_degenerate_indices = np.where(diff_12 < threshold)[0]
    
    # Find clusters of near-degenerate points
    clusters = []
    if len(near_degenerate_indices) > 0:
        current_cluster = [near_degenerate_indices[0]]
        for i in range(1, len(near_degenerate_indices)):
            if near_degenerate_indices[i] - near_degenerate_indices[i-1] <= 2:  # Adjacent or nearly adjacent points
                current_cluster.append(near_degenerate_indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [near_degenerate_indices[i]]
        clusters.append(current_cluster)  # Add the last cluster
    
    # Create a more informative plot
    plt.figure(figsize=(12, 10))
    
    # Plot both eigenvalues
    plt.subplot(2, 1, 1)
    plt.plot(theta, values1, 'r-', linewidth=2, label='Eigenstate 1')
    plt.plot(theta, values2, 'b-', linewidth=2, label='Eigenstate 2')
    
    # Highlight near-degenerate regions with shaded areas
    for cluster in clusters:
        if len(cluster) > 0:
            start_idx = max(0, cluster[0] - 2)
            end_idx = min(len(theta) - 1, cluster[-1] + 2)
            plt.axvspan(theta[start_idx], theta[end_idx], 
                        color='purple', alpha=0.2, label='Near-degenerate region')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10)
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title('Eigenstates 1 and 2 Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    
    # Add vertical lines at 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        plt.axvline(x=angle, color='gray', linestyle='--', alpha=0.5)
    
    # Plot the difference
    plt.subplot(2, 1, 2)
    plt.plot(theta, diff_12, 'g-', linewidth=2, label='|E₁ - E₂|')
    
    # Highlight regions below threshold
    plt.fill_between(theta, 0, diff_12, where=diff_12 < threshold, 
                     color='yellow', alpha=0.3, label=f'Difference < {threshold}')
    
    plt.xlabel('Theta (degrees)', fontsize=12)
    plt.ylabel('Absolute Difference', fontsize=12)
    plt.title('Degeneracy Between Eigenstates 1 and 2', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.ylim(bottom=0)  # Start y-axis at 0
    
    # Add vertical lines at 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        plt.axvline(x=angle, color='gray', linestyle='--', alpha=0.5)
    
    # Add statistics as text
    mean_diff = np.mean(diff_12)
    min_diff = np.min(diff_12)
    max_diff = np.max(diff_12)
    std_diff = np.std(diff_12)
    num_near_degenerate = len(near_degenerate_points)
    
    # Find the theta values where the minimum difference occurs
    min_diff_indices = np.where(diff_12 == min_diff)[0]
    min_diff_thetas = theta[min_diff_indices]
    min_diff_thetas_str = ', '.join([f"{t:.1f}°" for t in min_diff_thetas])
    
    stats_text = (
        f"Degeneracy Analysis:\n\n"
        f"Mean Difference: {mean_diff:.6f}\n"
        f"Min Difference: {min_diff:.6f} at θ = {min_diff_thetas_str}\n"
        f"Max Difference: {max_diff:.6f}\n"
        f"Std Deviation: {std_diff:.6f}\n"
        f"Near-Degenerate Points: {num_near_degenerate}\n"
        f"Degeneracy Threshold: {threshold}\n\n"
        f"Interpretation:\n"
        f"{'Significant degeneracy detected' if num_near_degenerate > 5 else 'No significant degeneracy detected'}\n"
        f"{'The minimum gap is very small, indicating potential level crossing' if min_diff < 0.001 else ''}"
    )
    plt.figtext(0.15, 0.15, stats_text, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eigenstate1_2_degeneracy.png", dpi=300)
    plt.close()

def create_all_visualizations(results, theta_values, eigenvalues, output_dir):
    """Create all visualizations for Berry phase analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    berry_phases = results.get("berry_phases", {})
    parity_flips = results.get("parity_flips", {})
    
    # Create visualizations
    plot_berry_phases(berry_phases, output_dir)
    
    # Make sure parity_flips is in the correct format
    if isinstance(parity_flips, np.ndarray) and parity_flips.ndim > 1:
        # If it's a 2D array, we need to convert it to a suitable format
        # Take the sum of parity flips for each eigenstate
        parity_flips_sum = np.sum(parity_flips, axis=1) if parity_flips.shape[0] < parity_flips.shape[1] else np.sum(parity_flips, axis=0)
        plot_parity_flips(parity_flips_sum, output_dir)
    else:
        plot_parity_flips(parity_flips, output_dir)
    
    # Create eigenstate data dictionary from theta_values and eigenvalues if available
    eigenstate_data = {}
    if theta_values is not None and eigenvalues is not None and len(theta_values) == len(eigenvalues):
        # Ensure theta values are in degrees for visualization
        theta_degrees = theta_values.copy()
        # If values are in radians (between 0 and 2π), convert to degrees
        if np.max(theta_degrees) <= 2 * np.pi:
            print("Converting theta values from radians to degrees for visualization")
            theta_degrees = np.degrees(theta_degrees)
        
        # Sort the data by theta values to ensure proper plotting
        sort_indices = np.argsort(theta_degrees)
        theta_degrees = theta_degrees[sort_indices]
        sorted_eigenvalues = eigenvalues[sort_indices]
        
        for i in range(sorted_eigenvalues.shape[1]):  # For each eigenstate
            eigenstate_data[i] = np.column_stack((theta_degrees, sorted_eigenvalues[:, i]))
        
        plot_eigenstate_vs_theta(eigenstate_data, output_dir)
        plot_eigenstate_degeneracy(eigenstate_data, output_dir)
    
    return True
