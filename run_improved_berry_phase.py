#!/usr/bin/env python3
"""
Run Improved Berry Phase Calculation

This script:
1. Loads eigenvectors and eigenvalues from the specified directory
2. Computes improved Berry phases with eigenstate tracking
3. Creates visualizations for analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from improved_berry_phase import load_eigenvectors_from_directory, compute_improved_berry_phase
from berry_phase_visualization import create_all_visualizations

def extract_theta_values(file_paths):
    """Extract theta values from file names."""
    theta_values = []
    for file_path in file_paths:
        # Extract theta value from filename (assuming format 'eigenvectors_theta_XXX.npy')
        filename = os.path.basename(file_path)
        theta_str = filename.split('_')[-1].split('.')[0]
        try:
            # Convert to float and then to radians if needed
            theta = float(theta_str)
            # Check if theta is in degrees (assuming values > 6.28 are degrees)
            if theta > 6.28:
                theta = np.radians(theta)
            theta_values.append(theta)
        except ValueError:
            print(f"Warning: Could not extract theta value from {filename}")
    
    return np.array(theta_values)

def save_results_to_file(results, output_dir='berry_phase_results', x_shift=None, y_shift=None, 
                     d_param=None, omega=None, a_vx=None, a_va=None):
    """Save Berry phase results to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with all parameters if provided
    if x_shift is not None and y_shift is not None:
        filename = f'improved_berry_phase_summary_x{x_shift}_y{y_shift}'
        
        # Add additional parameters to filename if provided
        if d_param is not None:
            filename += f'_d{d_param}'
        if omega is not None:
            filename += f'_w{omega}'
        if a_vx is not None:
            filename += f'_avx{a_vx}'
        if a_va is not None:
            filename += f'_ava{a_va}'
            
        filename += '.txt'
    else:
        filename = 'improved_berry_phase_summary.txt'
    
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write("Improved Berry Phase Analysis Results\n")
        f.write("====================================\n\n")
        
        f.write("Berry Phases:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Eigenstate':<10} {'Raw Phase (rad)':<15} {'Winding Number':<15} {'Normalized':<15} {'Quantized':<15} {'Error':<10} {'Full Cycle':<15}\n")
        f.write("-" * 100 + "\n")
        
        for i in range(len(results['berry_phases'])):
            f.write(f"{i:<10} {results['berry_phases'][i]:<15.6f} {results['winding_numbers'][i]:<15d} "
                    f"{results['normalized_phases'][i]:<15.6f} {results['quantized_values'][i]:<15.6f} "
                    f"{results['quantization_errors'][i]:<10.6f} {results['full_cycle_phases'][i]!s:<15}\n")
        
        f.write("\n\nParity Flip Summary:\n")
        f.write("-" * 50 + "\n")
        
        for i in range(results['parity_flips'].shape[0]):
            num_flips = np.sum(results['parity_flips'][i])
            f.write(f"Eigenstate {i}: {num_flips} parity flips\n")
    
    print(f"Results saved to {os.path.join(output_dir, 'improved_berry_phase_summary.txt')}")

def main():
    parser = argparse.ArgumentParser(description='Run improved Berry phase calculation')
    parser.add_argument('--input_dir', type=str, default='generalized/example_use/arrowhead_matrix/results',
                        help='Directory containing eigenvector files')
    parser.add_argument('--output_dir', type=str, default='improved_berry_phase_results',
                        help='Directory to save results')
    parser.add_argument('--plot_dir', type=str, default='improved_berry_phase_plots',
                        help='Directory to save plots')
    parser.add_argument('--x_shift', type=float, default=None,
                        help='VA potential x shift value for filename')
    parser.add_argument('--y_shift', type=float, default=None,
                        help='VA potential y shift value for filename')
    parser.add_argument('--d_param', type=float, default=None,
                        help='Distance parameter d for filename')
    parser.add_argument('--omega', type=float, default=None,
                        help='Angular frequency omega for filename')
    parser.add_argument('--a_vx', type=float, default=None,
                        help='Curvature parameter for VX potential for filename')
    parser.add_argument('--a_va', type=float, default=None,
                        help='Curvature parameter for VA potential for filename')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load eigenvectors and eigenvalues
    eigenvectors, eigenvalues = load_eigenvectors_from_directory(args.input_dir)
    
    if eigenvectors is None:
        print(f"Error: No eigenvector files found in {args.input_dir}")
        return 1
    
    # Extract theta values from file names
    file_paths = sorted(glob.glob(os.path.join(args.input_dir, "eigenvectors_theta_*.npy")))
    theta_values = extract_theta_values(file_paths)
    
    if len(theta_values) == 0:
        print("Warning: Could not extract theta values from filenames. Using evenly spaced values.")
        theta_values = np.linspace(0, 2*np.pi, eigenvectors.shape[0])
    
    # Compute improved Berry phases
    results = compute_improved_berry_phase(eigenvectors, eigenvalues, theta_values)
    
    # Create visualizations with all parameters in filenames if provided
    if args.x_shift is not None and args.y_shift is not None:
        plot_dir_with_params = f"{args.plot_dir}_x{args.x_shift}_y{args.y_shift}"
        
        # Add additional parameters to directory name if provided
        if args.d_param is not None:
            plot_dir_with_params += f"_d{args.d_param}"
        if args.omega is not None:
            plot_dir_with_params += f"_w{args.omega}"
        if args.a_vx is not None:
            plot_dir_with_params += f"_avx{args.a_vx}"
        if args.a_va is not None:
            plot_dir_with_params += f"_ava{args.a_va}"
            
        os.makedirs(plot_dir_with_params, exist_ok=True)
        create_all_visualizations(results, theta_values, eigenvalues, plot_dir_with_params)
    else:
        create_all_visualizations(results, theta_values, eigenvalues, args.plot_dir)
    
    # Save results to file with all parameters in filename if provided
    if args.x_shift is not None and args.y_shift is not None:
        save_results_to_file(results, args.output_dir, 
                          x_shift=args.x_shift, 
                          y_shift=args.y_shift,
                          d_param=args.d_param,
                          omega=args.omega,
                          a_vx=args.a_vx,
                          a_va=args.a_va)
    else:
        save_results_to_file(results, args.output_dir)
    
    print("\nImproved Berry phase calculation complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"Plots saved to {args.plot_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
