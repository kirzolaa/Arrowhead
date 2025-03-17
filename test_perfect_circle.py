#!/usr/bin/env python3
import numpy as np
from generalized.vector_utils import create_orthogonal_vectors

def test_perfect_circle():
    """
    Test the perfect circle generation with different origins
    """
    # Test with different origins
    origins = [
        np.array([0, 0, 0]),  # Origin at (0,0,0)
        np.array([1, 2, 3]),  # Arbitrary origin
        np.array([5, 5, 5])   # Origin on the x=y=z line
    ]
    
    d = 2.0  # Distance parameter
    num_points = 8  # Number of points
    
    for R_0 in origins:
        print(f"\nTesting with origin R_0 = {R_0}")
        
        # Generate vectors
        vectors = create_orthogonal_vectors(R_0, d, num_points, perfect=True)
        
        # Check properties
        distances = np.array([np.linalg.norm(v - R_0) for v in vectors])
        unit_111 = np.array([1, 1, 1]) / np.sqrt(3)
        dot_products = np.array([np.abs(np.dot(v - R_0, unit_111)) for v in vectors])
        
        # Print results
        print(f"Mean distance from origin: {np.mean(distances)}")
        print(f"Standard deviation of distances: {np.std(distances)}")
        print(f"Min/max distance ratio: {np.min(distances)/np.max(distances)}")
        print(f"Maximum dot product with (1,1,1): {np.max(dot_products)}")
        
        # Print individual vectors
        for i, v in enumerate(vectors):
            print(f"\nVector {i}:")
            print(f"  Position: {v}")
            print(f"  Distance from R_0: {distances[i]}")
            print(f"  Dot product with (1,1,1): {dot_products[i]}")

if __name__ == "__main__":
    test_perfect_circle()
