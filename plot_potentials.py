#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Import the potential functions from the arrowhead matrix generator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generalized/example_use/arrowhead_matrix'))
from generate_4x4_arrowhead import ArrowheadMatrix4x4

def plot_potentials():
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
    
    # Create directory for plots
    os.makedirs('potential_plots', exist_ok=True)
    
    # Plot VX potential
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_vx, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('VX Potential')
    ax.set_title('VX Potential Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('potential_plots/vx_potential_3d.png')
    
    # Plot VA potential
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_va, cmap='plasma', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('VA Potential')
    ax.set_title('VA Potential Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('potential_plots/va_potential_3d.png')
    
    # Plot 2D contour of VX potential
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z_vx, 20, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('VX Potential Contour')
    fig.colorbar(contour, ax=ax)
    plt.savefig('potential_plots/vx_potential_contour.png')
    
    # Plot 2D contour of VA potential
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z_va, 20, cmap='plasma')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('VA Potential Contour')
    fig.colorbar(contour, ax=ax)
    plt.savefig('potential_plots/va_potential_contour.png')
    
    # Plot both potentials together for comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    contour_vx = ax.contour(X, Y, Z_vx, 10, colors='blue', alpha=0.7)
    contour_va = ax.contour(X, Y, Z_va, 10, colors='red', alpha=0.7)
    ax.clabel(contour_vx, inline=True, fontsize=8, fmt='VX: %.1f')
    ax.clabel(contour_va, inline=True, fontsize=8, fmt='VA: %.1f')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Comparison of VX and VA Potentials')
    ax.legend(['VX Potential', 'VA Potential'])
    plt.savefig('potential_plots/potential_comparison.png')
    
    print("Potential plots saved to 'potential_plots' directory")

def check_eigenstate_degeneracy():
    """
    Check if eigenstates 1 and 2 are degenerate by analyzing the eigenvalues
    from the simulation results.
    """
    # Directory containing the results
    results_dir = 'generalized/example_use/arrowhead_matrix/results'
    
    # Find all eigenvalue files
    eigenvalue_files = sorted([f for f in os.listdir(results_dir) if f.startswith('eigenvalues_theta_') and f.endswith('.npy')])
    
    if not eigenvalue_files:
        print("No eigenvalue files found in the results directory")
        return
    
    # Load eigenvalues for all theta values
    eigenvalues = []
    theta_values = []
    
    for i, filename in enumerate(eigenvalue_files):
        filepath = os.path.join(results_dir, filename)
        eigenvals = np.load(filepath)
        eigenvalues.append(eigenvals)
        theta_values.append(i * 5)  # Assuming 5-degree steps
    
    eigenvalues = np.array(eigenvalues)
    theta_values = np.array(theta_values)
    
    # Calculate the difference between eigenvalues 1 and 2
    eigenvalue_diff = np.abs(eigenvalues[:, 1] - eigenvalues[:, 2])
    
    # Plot the eigenvalues
    plt.figure(figsize=(12, 8))
    plt.plot(theta_values, eigenvalues[:, 0], 'b-', label='Eigenvalue 0')
    plt.plot(theta_values, eigenvalues[:, 1], 'r-', label='Eigenvalue 1')
    plt.plot(theta_values, eigenvalues[:, 2], 'g-', label='Eigenvalue 2')
    plt.plot(theta_values, eigenvalues[:, 3], 'y-', label='Eigenvalue 3')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues vs. Theta')
    plt.legend()
    plt.grid(True)
    plt.savefig('potential_plots/eigenvalues_vs_theta.png')
    
    # Plot the difference between eigenvalues 1 and 2
    plt.figure(figsize=(12, 8))
    plt.plot(theta_values, eigenvalue_diff, 'r-')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('|Eigenvalue 1 - Eigenvalue 2|')
    plt.title('Difference Between Eigenvalues 1 and 2')
    plt.grid(True)
    plt.savefig('potential_plots/eigenvalue_difference.png')
    
    # Calculate statistics about the eigenvalue differences
    min_diff = np.min(eigenvalue_diff)
    max_diff = np.max(eigenvalue_diff)
    avg_diff = np.mean(eigenvalue_diff)
    
    print(f"Eigenvalue Difference Statistics:")
    print(f"  Minimum difference: {min_diff:.6f}")
    print(f"  Maximum difference: {max_diff:.6f}")
    print(f"  Average difference: {avg_diff:.6f}")
    
    # Check for degeneracy (using a small threshold)
    degeneracy_threshold = 1e-6
    is_degenerate = min_diff < degeneracy_threshold
    
    if is_degenerate:
        print(f"Eigenstates 1 and 2 appear to be degenerate (min difference < {degeneracy_threshold})")
    else:
        print(f"Eigenstates 1 and 2 do not appear to be degenerate (min difference >= {degeneracy_threshold})")
    
    # Save the detailed eigenvalue data to a text file
    with open('potential_plots/eigenvalue_analysis.txt', 'w') as f:
        f.write("Theta (degrees) | Eigenvalue 0 | Eigenvalue 1 | Eigenvalue 2 | Eigenvalue 3 | Diff(1-2)\n")
        f.write("-" * 80 + "\n")
        for i in range(len(theta_values)):
            f.write(f"{theta_values[i]:14.1f} | {eigenvalues[i, 0]:11.6f} | {eigenvalues[i, 1]:11.6f} | "
                   f"{eigenvalues[i, 2]:11.6f} | {eigenvalues[i, 3]:11.6f} | {eigenvalue_diff[i]:9.6f}\n")

if __name__ == "__main__":
    import sys
    
    # Plot the potentials
    plot_potentials()
    
    # Check for eigenstate degeneracy
    check_eigenstate_degeneracy()
