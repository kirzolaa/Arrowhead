#!/usr/bin/env python3
import numpy as np
from generalized.vector_utils import create_orthogonal_vectors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_perfect_circle_ranges():
    """
    Test the perfect circle generation with different d values and theta ranges
    """
    # Test with different parameters
    R_0 = np.array([1, 2, 3])  # Origin
    
    # Test 1: Different d values
    print("Test 1: Different d values")
    d_values = [0.5, 1.0, 2.0, 3.0]
    num_points = 36
    
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for d in d_values:
        # Generate vectors
        vectors = create_orthogonal_vectors(R_0, d, num_points, perfect=True)
        
        # Check properties
        distances = np.array([np.linalg.norm(v - R_0) for v in vectors])
        unit_111 = np.array([1, 1, 1]) / np.sqrt(3)
        dot_products = np.array([np.abs(np.dot(v - R_0, unit_111)) for v in vectors])
        
        # Print results
        print(f"\nCircle with d = {d}:")
        print(f"  Mean distance from origin: {np.mean(distances)}")
        print(f"  Standard deviation of distances: {np.std(distances)}")
        print(f"  Maximum dot product with (1,1,1): {np.max(dot_products)}")
        
        # Plot the circle
        ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], label=f'd = {d}')
    
    # Plot the origin
    ax.scatter(R_0[0], R_0[1], R_0[2], color='red', s=100, marker='o', label='Origin R_0')
    
    # Plot the x=y=z line
    line = np.array([[-1, -1, -1], [7, 7, 7]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r--', label='x=y=z line')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Perfect Circles with Different d Values')
    ax.legend()
    
    # Test 2: Different theta ranges
    print("\nTest 2: Different theta ranges")
    d = 2.0
    num_points = 18
    
    theta_ranges = [
        (0, np.pi/2),          # Quarter circle
        (0, np.pi),            # Half circle
        (np.pi/4, 3*np.pi/4),  # Middle segment
        (0, 2*np.pi)           # Full circle
    ]
    
    # Create figure for 3D visualization
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    for start_theta, end_theta in theta_ranges:
        # Generate vectors
        vectors = create_orthogonal_vectors(R_0, d, num_points, perfect=True, 
                                          start_theta=start_theta, end_theta=end_theta)
        
        # Check properties
        distances = np.array([np.linalg.norm(v - R_0) for v in vectors])
        unit_111 = np.array([1, 1, 1]) / np.sqrt(3)
        dot_products = np.array([np.abs(np.dot(v - R_0, unit_111)) for v in vectors])
        
        # Print results
        range_desc = f"({start_theta:.2f}, {end_theta:.2f})"
        print(f"\nCircle segment with theta range {range_desc}:")
        print(f"  Mean distance from origin: {np.mean(distances)}")
        print(f"  Standard deviation of distances: {np.std(distances)}")
        print(f"  Maximum dot product with (1,1,1): {np.max(dot_products)}")
        print(f"  Number of points: {len(vectors)}")
        
        # Plot the circle segment
        label = f'θ ∈ {range_desc}'
        ax2.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], label=label)
    
    # Plot the origin
    ax2.scatter(R_0[0], R_0[1], R_0[2], color='red', s=100, marker='o', label='Origin R_0')
    
    # Plot the x=y=z line
    line = np.array([[-1, -1, -1], [7, 7, 7]])
    ax2.plot(line[:, 0], line[:, 1], line[:, 2], 'r--', label='x=y=z line')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Perfect Circle Segments with Different Theta Ranges')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('perfect_circle_tests.png')
    plt.show()

if __name__ == "__main__":
    test_perfect_circle_ranges()
