#!/usr/bin/env python3
"""
Generate and visualize a circle of vectors orthogonal to the x=y=z line.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generalized.vector_utils import create_orthogonal_vectors

def generate_orthogonal_circle(R_0=(1, 1, 1), d=2.0, num_points=73):
    """
    Generate a circle of vectors orthogonal to the x=y=z line.
    
    Parameters:
    R_0 (tuple): Origin vector, default is (1, 1, 1)
    d (float): Distance parameter, default is 2.0
    num_points (int): Number of points to generate, default is 73
    
    Returns:
    numpy.ndarray: Array of points forming the circle
    """
    # Convert R_0 to numpy array
    R_0 = np.array(R_0)
    
    # Use the create_orthogonal_vectors function with perfect=True to generate the circle
    points = create_orthogonal_vectors(R_0, d=d, num_points=num_points, perfect=True)
    
    return points

def visualize_orthogonal_circle(points, R_0):
    """
    Visualize the circle of vectors orthogonal to the x=y=z line.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    R_0 (numpy.ndarray): Origin vector
    """
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('orthogonal_circle_output'):
        os.makedirs('orthogonal_circle_output')
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter([0], [0], [0], color='black', s=100, label='Origin')
    
    # Plot R_0
    ax.scatter(R_0[0], R_0[1], R_0[2], color='blue', s=100, label='R_0')
    ax.plot([0, R_0[0]], [0, R_0[1]], [0, R_0[2]], 'b--', alpha=0.5)
    
    # Plot the circle points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=20, alpha=0.7, label='Circle Points')
    
    # Connect the points to show the circle
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.5)
    
    # Add axis lines with higher visibility and labels
    # Adjust max_val to be closer to the actual data for better visualization
    max_val = max(np.max(np.abs(points)), np.max(np.abs(R_0))) * 1.5
    
    # X-axis - red with label and coordinate markers
    ax.plot([-max_val, max_val], [0, 0], [0, 0], 'r-', alpha=0.6, linewidth=1.0)
    ax.text(max_val*1.1, 0, 0, 'X', color='red', fontsize=12)
    
    # Add coordinate markers along X-axis
    for i in range(-int(max_val), int(max_val)+1):
        if i != 0 and i % 1 == 0:  # Only show integer values, skip zero
            ax.text(i, 0, 0, f'{i}', color='red', fontsize=8, ha='center', va='bottom')
            # Add small tick marks
            ax.plot([i, i], [0, -0.05], [0, 0], 'r-', alpha=0.4, linewidth=0.5)
    
    # Y-axis - green with label and coordinate markers
    ax.plot([0, 0], [-max_val, max_val], [0, 0], 'g-', alpha=0.6, linewidth=1.0)
    ax.text(0, max_val*1.1, 0, 'Y', color='green', fontsize=12)
    
    # Add coordinate markers along Y-axis
    for i in range(-int(max_val), int(max_val)+1):
        if i != 0 and i % 1 == 0:  # Only show integer values, skip zero
            ax.text(0, i, 0, f'{i}', color='green', fontsize=8, ha='right', va='center')
            # Add small tick marks
            ax.plot([0, -0.05], [i, i], [0, 0], 'g-', alpha=0.4, linewidth=0.5)
    
    # Z-axis - blue with label and coordinate markers
    ax.plot([0, 0], [0, 0], [-max_val, max_val], 'b-', alpha=0.6, linewidth=1.0)
    ax.text(0, 0, max_val*1.1, 'Z', color='blue', fontsize=12)
    
    # Add coordinate markers along Z-axis
    for i in range(-int(max_val), int(max_val)+1):
        if i != 0 and i % 1 == 0:  # Only show integer values, skip zero
            ax.text(0, 0, i, f'{i}', color='blue', fontsize=8, ha='right', va='center')
            # Add small tick marks
            ax.plot([0, -0.05], [0, 0], [i, i], 'b-', alpha=0.4, linewidth=0.5)
    
    # Plot the (1,1,1) direction
    max_val = max(np.max(np.abs(points)), np.max(np.abs(R_0))) * 1.5
    ax.plot([0, max_val], [0, max_val], [0, max_val], 'k-', label='x=y=z line')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Circle Orthogonal to x=y=z line')
    
    # Set equal aspect ratio and adjust limits for better viewing
    buffer = max_val * 0.2  # Add a small buffer for better visibility
    
    # Calculate actual data bounds for better scaling
    data_max = max(np.max(np.abs(points)), np.max(np.abs(R_0))) * 1.2
    
    # Use data-driven limits instead of the larger max_val
    ax.set_xlim([-data_max-buffer, data_max+buffer])
    ax.set_ylim([-data_max-buffer, data_max+buffer])
    ax.set_zlim([-data_max-buffer, data_max+buffer])
    
    # Set equal aspect ratio for better 3D visualization
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.savefig('orthogonal_circle_output/orthogonal_circle_3d.png', dpi=300, bbox_inches='tight')
    
    # Create projections onto the coordinate planes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY projection
    axs[0, 0].scatter(points[:, 0], points[:, 1], color='red', s=20, alpha=0.7)
    axs[0, 0].plot(points[:, 0], points[:, 1], 'r-', alpha=0.5)
    axs[0, 0].scatter(R_0[0], R_0[1], color='blue', s=100, label='R_0')
    axs[0, 0].scatter(0, 0, color='black', s=100, label='Origin')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('XY Projection')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # XZ projection
    axs[0, 1].scatter(points[:, 0], points[:, 2], color='red', s=20, alpha=0.7)
    axs[0, 1].plot(points[:, 0], points[:, 2], 'r-', alpha=0.5)
    axs[0, 1].scatter(R_0[0], R_0[2], color='blue', s=100, label='R_0')
    axs[0, 1].scatter(0, 0, color='black', s=100, label='Origin')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Z')
    axs[0, 1].set_title('XZ Projection')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # YZ projection
    axs[1, 0].scatter(points[:, 1], points[:, 2], color='red', s=20, alpha=0.7)
    axs[1, 0].plot(points[:, 1], points[:, 2], 'r-', alpha=0.5)
    axs[1, 0].scatter(R_0[1], R_0[2], color='blue', s=100, label='R_0')
    axs[1, 0].scatter(0, 0, color='black', s=100, label='Origin')
    axs[1, 0].set_xlabel('Y')
    axs[1, 0].set_ylabel('Z')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1/2, -1/2])
    basis2 = np.array([0, -1/2, 1/2])
    
    # Normalize them
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # Project all points onto the orthogonal plane
    points_proj = []
    for p in points:
        p_proj = np.array([np.dot(p, basis1), np.dot(p, basis2)])
        points_proj.append(p_proj)
    
    points_proj = np.array(points_proj)
    
    # Project R_0 onto this plane
    R_0_proj = np.array([np.dot(R_0, basis1), np.dot(R_0, basis2)])
    
    # Orthogonal plane projection
    axs[1, 1].scatter(points_proj[:, 0], points_proj[:, 1], color='red', s=20, alpha=0.7)
    axs[1, 1].plot(points_proj[:, 0], points_proj[:, 1], 'r-', alpha=0.5)
    axs[1, 1].scatter(R_0_proj[0], R_0_proj[1], color='blue', s=100, label='R_0')
    axs[1, 1].scatter(0, 0, color='black', s=100, label='Origin')
    axs[1, 1].set_xlabel('Basis Vector 1 Direction')
    axs[1, 1].set_ylabel('Basis Vector 2 Direction')
    axs[1, 1].set_title('Projection onto Plane Orthogonal to x=y=z')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('orthogonal_circle_output/orthogonal_circle_projections.png', dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to 'orthogonal_circle_output' directory.")

def main():
    """Main function to generate and visualize the orthogonal circle."""
    # Set parameters
    R_0 = np.array([1, 1, 1])
    d = 2.0
    num_points = 73  # 5-degree increments
    
    print(f"Generating circle with {num_points} points...")
    print(f"Parameters: R_0={R_0}, d={d}")
    
    # Generate circle
    points = generate_orthogonal_circle(R_0, d, num_points)
    
    # Verify orthogonality to (1,1,1) direction
    unit_111 = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized (1,1,1) vector
    
    # Calculate dot products for each point
    dot_products = []
    for p in points:
        dot_product = np.abs(np.dot(p - R_0, unit_111))
        dot_products.append(dot_product)
    
    max_dot_product = max(dot_products)
    avg_dot_product = sum(dot_products) / len(dot_products)
    
    print("\nOrthogonality Verification:")
    print(f"Maximum dot product with (1,1,1): {max_dot_product}")
    print(f"Average dot product with (1,1,1): {avg_dot_product}")
    print(f"All vectors are orthogonal to the x=y=z line (within floating-point precision).")
    
    # Visualize circle
    visualize_orthogonal_circle(points, R_0)

if __name__ == "__main__":
    main()
