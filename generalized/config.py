#!/usr/bin/env python3
import numpy as np
import math
import json
import os

class VectorConfig:
    """
    Configuration class for orthogonal vector generation and visualization
    """
    def __init__(self, 
                 R_0=(0, 0, 0), 
                 d=1, 
                 theta=math.pi/4,
                 show_r0_plane=True,
                 figsize_3d=(10, 8),
                 figsize_2d=(8, 8),
                 show_legend=True,
                 show_grid=True):
        """
        Initialize the configuration
        
        Parameters:
        R_0 (tuple or list): The origin vector
        d (float): The distance parameter
        theta (float): The angle parameter in radians
        show_r0_plane (bool): Whether to show the R_0 plane projection
        figsize_3d (tuple): Figure size for 3D plot
        figsize_2d (tuple): Figure size for 2D plots
        show_legend (bool): Whether to show the legend
        show_grid (bool): Whether to show the grid
        """
        self.R_0 = np.array(R_0)
        self.d = d
        self.theta = theta
        self.show_r0_plane = show_r0_plane
        self.figsize_3d = figsize_3d
        self.figsize_2d = figsize_2d
        self.show_legend = show_legend
        self.show_grid = show_grid
    
    def to_dict(self):
        """
        Convert the configuration to a dictionary
        
        Returns:
        dict: Dictionary representation of the configuration
        """
        return {
            'R_0': self.R_0.tolist(),
            'd': self.d,
            'theta': self.theta,
            'show_r0_plane': self.show_r0_plane,
            'figsize_3d': self.figsize_3d,
            'figsize_2d': self.figsize_2d,
            'show_legend': self.show_legend,
            'show_grid': self.show_grid
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a configuration from a dictionary
        
        Parameters:
        config_dict (dict): Dictionary containing configuration parameters
        
        Returns:
        VectorConfig: Configuration object
        """
        return cls(
            R_0=config_dict.get('R_0', (0, 0, 0)),
            d=config_dict.get('d', 1),
            theta=config_dict.get('theta', math.pi/4),
            show_r0_plane=config_dict.get('show_r0_plane', True),
            figsize_3d=config_dict.get('figsize_3d', (10, 8)),
            figsize_2d=config_dict.get('figsize_2d', (8, 8)),
            show_legend=config_dict.get('show_legend', True),
            show_grid=config_dict.get('show_grid', True)
        )
    
    def save_to_file(self, filename):
        """
        Save the configuration to a JSON file
        
        Parameters:
        filename (str): Path to the output file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Load a configuration from a JSON file
        
        Parameters:
        filename (str): Path to the input file
        
        Returns:
        VectorConfig: Configuration object
        """
        if not os.path.exists(filename):
            print(f"Warning: Config file {filename} not found. Using default configuration.")
            return cls()
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# Default configuration
default_config = VectorConfig()
