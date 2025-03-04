#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vectors_3d(R_0, R_1, R_2, R_3, figsize=(10, 8), show_legend=True):
    """
    Plot the vectors in 3D
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R_1, R_2, R_3 (numpy.ndarray): The three orthogonal vectors
    figsize (tuple): Figure size (width, height) in inches
    show_legend (bool): Whether to show the legend
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter(R_0[0], R_0[1], R_0[2], color='black', s=100, label='R_0')
    
    # Plot the vectors as arrows from the origin
    vectors = [R_1, R_2, R_3]
    colors = ['r', 'g', 'b']
    labels = ['R_1', 'R_2', 'R_3']
    
    for i, (vector, color, label) in enumerate(zip(vectors, colors, labels)):
        ax.quiver(R_0[0], R_0[1], R_0[2], 
                 vector[0]-R_0[0], vector[1]-R_0[1], vector[2]-R_0[2], 
                 color=color, label=label, arrow_length_ratio=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Orthogonal Vectors')
    
    # Set equal aspect ratio
    max_range = np.array([
        np.max([R_0[0], R_1[0], R_2[0], R_3[0]]) - np.min([R_0[0], R_1[0], R_2[0], R_3[0]]),
        np.max([R_0[1], R_1[1], R_2[1], R_3[1]]) - np.min([R_0[1], R_1[1], R_2[1], R_3[1]]),
        np.max([R_0[2], R_1[2], R_2[2], R_3[2]]) - np.min([R_0[2], R_1[2], R_2[2], R_3[2]])
    ]).max() / 2.0
    
    mid_x = (np.max([R_0[0], R_1[0], R_2[0], R_3[0]]) + np.min([R_0[0], R_1[0], R_2[0], R_3[0]])) / 2
    mid_y = (np.max([R_0[1], R_1[1], R_2[1], R_3[1]]) + np.min([R_0[1], R_1[1], R_2[1], R_3[1]])) / 2
    mid_z = (np.max([R_0[2], R_1[2], R_2[2], R_3[2]]) + np.min([R_0[2], R_1[2], R_2[2], R_3[2]])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if show_legend:
        ax.legend()
    
    return fig, ax

def plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='xy', figsize=(8, 8), show_legend=True, show_grid=True):
    """
    Plot the 2D projection of the vectors
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R_1, R_2, R_3 (numpy.ndarray): The three orthogonal vectors
    plane (str): The plane to project onto ('xy', 'xz', 'yz', or 'r0')
    figsize (tuple): Figure size (width, height) in inches
    show_legend (bool): Whether to show the legend
    show_grid (bool): Whether to show the grid
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define indices based on the projection plane
    if plane == 'xy':
        i, j = 0, 1
        plane_name = 'XY'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R1_proj = np.array([R_1[i], R_1[j]])
        R2_proj = np.array([R_2[i], R_2[j]])
        R3_proj = np.array([R_3[i], R_3[j]])
        
    elif plane == 'xz':
        i, j = 0, 2
        plane_name = 'XZ'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R1_proj = np.array([R_1[i], R_1[j]])
        R2_proj = np.array([R_2[i], R_2[j]])
        R3_proj = np.array([R_3[i], R_3[j]])
        
    elif plane == 'yz':
        i, j = 1, 2
        plane_name = 'YZ'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R1_proj = np.array([R_1[i], R_1[j]])
        R2_proj = np.array([R_2[i], R_2[j]])
        R3_proj = np.array([R_3[i], R_3[j]])
        
    elif plane == 'r0':
        # For R_0 plane, we need to find two orthogonal vectors in the plane
        # perpendicular to the vector from origin to R_0
        
        # If R_0 is the origin, we can use any plane passing through the origin
        if np.allclose(R_0, np.zeros(3)):
            # Use the plane defined by R_1 and R_2
            basis1 = R_1 - R_0
            basis1 = basis1 / np.linalg.norm(basis1)
            
            basis2 = R_2 - R_0
            basis2 = basis2 / np.linalg.norm(basis2)
            
            # Make sure basis2 is orthogonal to basis1
            basis2 = basis2 - np.dot(basis2, basis1) * basis1
            basis2 = basis2 / np.linalg.norm(basis2)
        else:
            # Define the normal to the plane as the vector from origin to R_0
            normal = R_0 / np.linalg.norm(R_0)
            
            # Find two orthogonal vectors in the plane
            # First basis vector: cross product of normal with [1,0,0] or [0,1,0]
            if not np.allclose(normal, np.array([1, 0, 0])):
                basis1 = np.cross(normal, np.array([1, 0, 0]))
            else:
                basis1 = np.cross(normal, np.array([0, 1, 0]))
            basis1 = basis1 / np.linalg.norm(basis1)
            
            # Second basis vector: cross product of normal with basis1
            basis2 = np.cross(normal, basis1)
            basis2 = basis2 / np.linalg.norm(basis2)
        
        plane_name = 'R_0'
        
        # Project vectors onto the plane defined by basis1 and basis2
        R0_proj = np.array([0, 0])  # Origin in the plane
        
        # Project R_1, R_2, R_3 onto the plane
        v1 = R_1 - R_0
        v2 = R_2 - R_0
        v3 = R_3 - R_0
        
        R1_proj = np.array([np.dot(v1, basis1), np.dot(v1, basis2)])
        R2_proj = np.array([np.dot(v2, basis1), np.dot(v2, basis2)])
        R3_proj = np.array([np.dot(v3, basis1), np.dot(v3, basis2)])
    else:
        raise ValueError("Plane must be 'xy', 'xz', 'yz', or 'r0'")
    
    # Plot the origin
    ax.scatter(R0_proj[0], R0_proj[1], color='black', s=100, label='R_0')
    
    # Plot the vectors as arrows from the origin
    vectors_proj = [R1_proj, R2_proj, R3_proj]
    colors = ['r', 'g', 'b']
    labels = ['R_1', 'R_2', 'R_3']
    
    for vector, color, label in zip(vectors_proj, colors, labels):
        ax.arrow(R0_proj[0], R0_proj[1], 
                vector[0]-R0_proj[0], vector[1]-R0_proj[1], 
                head_width=0.05, head_length=0.1, fc=color, ec=color, label=label)
    
    # Set labels and title
    if plane == 'r0':
        ax.set_xlabel('Basis 1')
        ax.set_ylabel('Basis 2')
        ax.set_title(f'2D Projection on the {plane_name} Plane')
    else:
        ax.set_xlabel(f'{plane_name[0]} axis')
        ax.set_ylabel(f'{plane_name[1]} axis')
        ax.set_title(f'2D Projection on {plane_name} Plane')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid and legend
    if show_grid:
        ax.grid(True)
    
    if show_legend:
        ax.legend()
    
    return fig, ax

def plot_all_projections(R_0, R_1, R_2, R_3, show_r0_plane=True, figsize_3d=(10, 8), figsize_2d=(8, 8)):
    """
    Plot the 3D vectors and all 2D projections
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R_1, R_2, R_3 (numpy.ndarray): The three orthogonal vectors
    show_r0_plane (bool): Whether to show the R_0 plane projection
    figsize_3d (tuple): Figure size for 3D plot
    figsize_2d (tuple): Figure size for 2D plots
    
    Returns:
    dict: Dictionary containing all figure and axis objects
    """
    results = {}
    
    # Plot in 3D
    fig_3d, ax_3d = plot_vectors_3d(R_0, R_1, R_2, R_3, figsize=figsize_3d)
    results['3d'] = (fig_3d, ax_3d)
    
    # Plot 2D projections
    fig_xy, ax_xy = plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='xy', figsize=figsize_2d)
    results['xy'] = (fig_xy, ax_xy)
    
    fig_xz, ax_xz = plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='xz', figsize=figsize_2d)
    results['xz'] = (fig_xz, ax_xz)
    
    fig_yz, ax_yz = plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='yz', figsize=figsize_2d)
    results['yz'] = (fig_yz, ax_yz)
    
    # Plot on the R_0 plane if requested
    if show_r0_plane:
        fig_r0, ax_r0 = plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='r0', figsize=figsize_2d)
        results['r0'] = (fig_r0, ax_r0)
    
    return results
