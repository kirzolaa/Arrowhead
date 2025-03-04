#!/usr/bin/env python3
import numpy as np

def create_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
    """
    Create 3 orthogonal R vectors for R_0
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector, default is (0, 0, 0)
    d (float): The distance parameter, default is 1
    theta (float): The angle parameter in radians, default is 0
    
    Returns:
    tuple: Three orthogonal vectors R_1, R_2, R_3
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Calculate R_1, R_2, R_3 according to the given formulas
    # R_1 = R_0 + d * (cos(theta))*sqrt(2/3)
    R_1 = R_0 + d * np.cos(theta) * np.sqrt(2/3) * np.array([1, -1/2, -1/2])
    
    # R_2 = R_0 + d * (cos(theta)/sqrt(3) + sin(theta))/sqrt(2)
    R_2 = R_0 + d * (np.cos(theta)/np.sqrt(3) + np.sin(theta))/np.sqrt(2) * np.array([1, 1, 1])
    
    # R_3 = R_0 + d * (sin(theta) - cos(theta)/sqrt(3))/sqrt(2)
    R_3 = R_0 + d * (np.sin(theta) - np.cos(theta)/np.sqrt(3))/np.sqrt(2) * np.array([0, -1/2, 1/2]) * np.sqrt(2)
    
    return R_1, R_2, R_3

def check_orthogonality(R_0, R_1, R_2, R_3):
    """
    Check if the vectors R_1, R_2, R_3 are orthogonal with respect to R_0
    
    Parameters:
    R_0, R_1, R_2, R_3 (numpy.ndarray): The vectors to check
    
    Returns:
    dict: Dictionary containing the dot products between pairs of vectors
    """
    dot_1_2 = np.dot(R_1 - R_0, R_2 - R_0)
    dot_1_3 = np.dot(R_1 - R_0, R_3 - R_0)
    dot_2_3 = np.dot(R_2 - R_0, R_3 - R_0)
    
    return {
        "R_1 · R_2": dot_1_2,
        "R_1 · R_3": dot_1_3,
        "R_2 · R_3": dot_2_3
    }
