#!/usr/bin/env python3
"""
Test script to verify orthogonality of vectors to the x=y=z line
and save the visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generalized.vector_utils import create_orthogonal_vectors, check_vector_components

def test_orthogonality():
    """Test the orthogonality of vectors to the x=y=z line and save visualization."""
    # Create a figure directory if it doesn't exist
    import os
    if not os.path.exists('orthogonality_test_figures'):
        os.makedirs('orthogonality_test_figures')
    
    # Test parameters
    R_0 = np.array([1, 1, 1])
    d = 2.0
    theta = np.pi/4
    
    # Generate the vector
    R = create_orthogonal_vectors(R_0, d=d, theta=theta)
    
    # Check components and orthogonality
    result = check_vector_components(R_0, d=d, theta=theta, R=R)
    
    # Print results
    print('Test Parameters:')
    print(f'R_0: {R_0}')
    print(f'd: {d}')
    print(f'theta: {theta} radians ({theta * 180/np.pi} degrees)')
    print('\nResults:')
    print(f'R: {R}')
    print(f'R - R_0: {R - R_0}')
    
    # Verify orthogonality to (1,1,1) direction
    unit_111 = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized (1,1,1) vector
    dot_product = np.dot(R - R_0, unit_111)
    print(f'\nOrthogonality Verification:')
    print(f'Dot product with (1,1,1): {dot_product}')
    print(f'Orthogonality to (1,1,1) (should be close to 0): {result["Orthogonality to (1,1,1) (should be close to 0)"]}')
    
    # Print component details
    print('\nComponent Details:')
    for key, value in result.items():
        if key != "Orthogonality to (1,1,1) (should be close to 0)" and key != "Verification (should be close to 0)":
            print(f'{key}: {value}')
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter([0], [0], [0], color='black', s=100, label='Origin')
    
    # Plot R_0
    ax.scatter(R_0[0], R_0[1], R_0[2], color='blue', s=100, label='R_0')
    ax.plot([0, R_0[0]], [0, R_0[1]], [0, R_0[2]], 'b--', alpha=0.5)
    
    # Plot R
    ax.scatter(R[0], R[1], R[2], color='red', s=100, label='R')
    ax.plot([0, R[0]], [0, R[1]], [0, R[2]], 'r--', alpha=0.5)
    
    # Plot the displacement vector R - R_0
    ax.quiver(R_0[0], R_0[1], R_0[2], 
              R[0] - R_0[0], R[1] - R_0[1], R[2] - R_0[2], 
              color='green', arrow_length_ratio=0.1, label='R - R_0')
    
    # Plot the (1,1,1) direction
    max_val = max(np.max(np.abs(R)), np.max(np.abs(R_0))) * 1.5
    ax.plot([0, max_val], [0, max_val], [0, max_val], 'k-', label='x=y=z line')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Orthogonality Test: Vector R - R_0 is orthogonal to x=y=z line')
    
    # Set equal aspect ratio
    max_range = max_val
    ax.set_xlim([-max_range/2, max_range])
    ax.set_ylim([-max_range/2, max_range])
    ax.set_zlim([-max_range/2, max_range])
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.savefig('orthogonality_test_figures/orthogonality_test_3d.png', dpi=300, bbox_inches='tight')
    
    # Create a 2D visualization to show orthogonality more clearly
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Calculate the projection onto a plane perpendicular to (1,1,1)
    # This is a bit complex, but we can use the fact that our basis vectors are already in this plane
    
    # Define the basis vectors
    basis1 = np.array([1, -1/2, -1/2])
    basis2 = np.array([0, -1/2, 1/2])
    
    # Normalize them
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # Project R_0 and R onto this plane
    R_0_proj = np.array([np.dot(R_0, basis1), np.dot(R_0, basis2)])
    R_proj = np.array([np.dot(R, basis1), np.dot(R, basis2)])
    
    # Plot the origin
    ax2.scatter([0], [0], color='black', s=100, label='Origin')
    
    # Plot R_0 projection
    ax2.scatter(R_0_proj[0], R_0_proj[1], color='blue', s=100, label='R_0 projection')
    ax2.plot([0, R_0_proj[0]], [0, R_0_proj[1]], 'b--', alpha=0.5)
    
    # Plot R projection
    ax2.scatter(R_proj[0], R_proj[1], color='red', s=100, label='R projection')
    ax2.plot([0, R_proj[0]], [0, R_proj[1]], 'r--', alpha=0.5)
    
    # Plot the displacement vector R - R_0 projection
    ax2.arrow(R_0_proj[0], R_0_proj[1], 
              R_proj[0] - R_0_proj[0], R_proj[1] - R_0_proj[1], 
              color='green', width=0.05, label='(R - R_0) projection')
    
    # Set labels and title
    ax2.set_xlabel('Basis Vector 1 Direction')
    ax2.set_ylabel('Basis Vector 2 Direction')
    ax2.set_title('Projection onto Plane Orthogonal to x=y=z line')
    
    # Set equal aspect ratio
    ax2.set_aspect('equal')
    
    # Add legend
    ax2.legend()
    
    # Add grid
    ax2.grid(True)
    
    # Save the figure
    plt.savefig('orthogonality_test_figures/orthogonality_test_2d.png', dpi=300, bbox_inches='tight')
    
    # Generate multiple vectors with varying theta to show the circle
    thetas = np.linspace(0, 2*np.pi, 73)  # 73 points for 5-degree increments
    points = []
    
    for t in thetas:
        vector = create_orthogonal_vectors(R_0, d=d, theta=t)
        points.append(vector)
    
    points = np.array(points)
    
    # Create a 3D visualization of the circle
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax3.scatter([0], [0], [0], color='black', s=100, label='Origin')
    
    # Plot R_0
    ax3.scatter(R_0[0], R_0[1], R_0[2], color='blue', s=100, label='R_0')
    ax3.plot([0, R_0[0]], [0, R_0[1]], [0, R_0[2]], 'b--', alpha=0.5)
    
    # Plot the circle points
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=20, alpha=0.7, label='Circle Points')
    
    # Plot the (1,1,1) direction
    ax3.plot([0, max_val], [0, max_val], [0, max_val], 'k-', label='x=y=z line')
    
    # Set labels and title
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Circle Generated by Varying Theta (Orthogonal to x=y=z line)')
    
    # Set equal aspect ratio
    ax3.set_xlim([-max_range/2, max_range])
    ax3.set_ylim([-max_range/2, max_range])
    ax3.set_zlim([-max_range/2, max_range])
    
    # Add legend
    ax3.legend()
    
    # Add grid
    ax3.grid(True)
    
    # Save the figure
    plt.savefig('orthogonality_test_figures/orthogonality_circle_3d.png', dpi=300, bbox_inches='tight')
    
    # Create a 2D visualization of the circle in the orthogonal plane
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # Project all points onto the orthogonal plane
    points_proj = []
    for p in points:
        p_proj = np.array([np.dot(p, basis1), np.dot(p, basis2)])
        points_proj.append(p_proj)
    
    points_proj = np.array(points_proj)
    
    # Plot the origin
    ax4.scatter([0], [0], color='black', s=100, label='Origin')
    
    # Plot R_0 projection
    ax4.scatter(R_0_proj[0], R_0_proj[1], color='blue', s=100, label='R_0 projection')
    
    # Plot the circle points
    ax4.scatter(points_proj[:, 0], points_proj[:, 1], color='red', s=20, alpha=0.7, label='Circle Points')
    
    # Connect the points to show the circle
    ax4.plot(points_proj[:, 0], points_proj[:, 1], 'r-', alpha=0.5)
    
    # Set labels and title
    ax4.set_xlabel('Basis Vector 1 Direction')
    ax4.set_ylabel('Basis Vector 2 Direction')
    ax4.set_title('Circle in Plane Orthogonal to x=y=z line')
    
    # Set equal aspect ratio
    ax4.set_aspect('equal')
    
    # Add legend
    ax4.legend()
    
    # Add grid
    ax4.grid(True)
    
    # Save the figure
    plt.savefig('orthogonality_test_figures/orthogonality_circle_2d.png', dpi=300, bbox_inches='tight')
    
    print("\nTest completed. Figures saved in 'orthogonality_test_figures' directory.")

if __name__ == "__main__":
    test_orthogonality()
