#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import os
import sys

from vector_utils import create_orthogonal_vectors, check_orthogonality
from visualization import plot_vectors_3d, plot_vectors_2d_projection, plot_all_projections
from config import VectorConfig, default_config

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate and visualize orthogonal vectors')
    
    # Vector parameters
    parser.add_argument('--origin', '-R', type=float, nargs=3, default=[0, 0, 0],
                        help='Origin vector R_0 (x y z)')
    parser.add_argument('--distance', '-d', type=float, default=1,
                        help='Distance parameter d')
    parser.add_argument('--angle', '-a', type=float, default=math.pi/4,
                        help='Angle parameter theta in radians')
    
    # Visualization parameters
    parser.add_argument('--no-r0-plane', action='store_false', dest='show_r0_plane',
                        help='Do not show the R_0 plane projection')
    parser.add_argument('--no-legend', action='store_false', dest='show_legend',
                        help='Do not show the legend')
    parser.add_argument('--no-grid', action='store_false', dest='show_grid',
                        help='Do not show the grid')
    
    # Output parameters
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying them')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots to')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--save-config', type=str,
                        help='Save configuration to file')
    
    return parser.parse_args()

def display_help():
    """
    Display detailed help information
    """
    help_text = """
    Orthogonal Vectors Generator and Visualizer
    =======================================
    
    This tool generates and visualizes three orthogonal vectors from a given origin point.
    
    Basic Usage:
    -----------
    python main.py                              # Use default parameters
    python main.py -R 1 1 1                    # Set origin to (1,1,1)
    python main.py -R 0 0 2 -d 1.5 -a 0.5236   # Custom origin, distance and angle
    python main.py --help                      # Show help
    
    Parameters:
    ----------
    -R, --origin X Y Z    : Set the origin vector R_0 coordinates (default: 0 0 0)
    -d, --distance VALUE  : Set the distance parameter (default: 1)
    -a, --angle VALUE     : Set the angle parameter in radians (default: π/4)
    
    Visualization Options:
    --------------------
    --no-r0-plane        : Do not show the R_0 plane projection
    --no-legend          : Do not show the legend
    --no-grid            : Do not show the grid
    
    Output Options:
    --------------
    --save-plots         : Save plots to files instead of displaying them
    --output-dir DIR     : Directory to save plots to (default: 'plots')
    
    Configuration:
    -------------
    --config FILE        : Load configuration from a JSON file
    --save-config FILE   : Save current configuration to a JSON file
    
    Examples:
    --------
    # Generate vectors with origin at (1,1,1), distance 2, and angle π/3
    python main.py -R 1 1 1 -d 2 -a 1.047
    
    # Save plots to a custom directory
    python main.py -R 0 0 2 --save-plots --output-dir my_plots
    
    # Load configuration from a file
    python main.py --config my_config.json
    """
    print(help_text)
    sys.exit(0)

def main():
    """
    Main function
    """
    # Check for detailed help command
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        display_help()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        config = VectorConfig.load_from_file(args.config)
    else:
        # Create configuration from command line arguments
        config = VectorConfig(
            R_0=args.origin,
            d=args.distance,
            theta=args.angle,
            show_r0_plane=args.show_r0_plane,
            show_legend=args.show_legend,
            show_grid=args.show_grid
        )
    
    # Save configuration if requested
    if args.save_config:
        config.save_to_file(args.save_config)
    
    # Create the orthogonal vectors
    R_0 = config.R_0
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, config.d, config.theta)
    
    # Print vector information
    print("R_0:", R_0)
    print("R_1:", R_1)
    print("R_2:", R_2)
    print("R_3:", R_3)
    
    # Check orthogonality
    orthogonality = check_orthogonality(R_0, R_1, R_2, R_3)
    print("\nChecking orthogonality (dot products should be close to zero):")
    for key, value in orthogonality.items():
        print(f"{key}: {value}")
    
    # Plot the vectors
    plots = plot_all_projections(
        R_0, R_1, R_2, R_3,
        show_r0_plane=config.show_r0_plane,
        figsize_3d=config.figsize_3d,
        figsize_2d=config.figsize_2d
    )
    
    # Save or show the plots
    if args.save_plots:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save each plot
        for name, (fig, _) in plots.items():
            filename = os.path.join(args.output_dir, f"{name}.png")
            fig.savefig(filename)
            print(f"Saved plot to {filename}")
    else:
        # Show the plots
        plt.show()

if __name__ == "__main__":
    main()
