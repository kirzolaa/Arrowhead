#!/usr/bin/env python3
"""
Visualization Script for Parameter Analysis Results

This script creates visualizations of the parameter analysis results,
focusing on the relationship between y_shift values and parity flips.
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extract_parameters_from_filename(filename):
    """Extract parameters from a filename."""
    pattern = r"x(\d+\.\d+)_y(\d+\.\d+)_d(\d+\.\d+)_w(\d+\.\d+)_avx(\d+\.\d+)_ava(\d+\.\d+)"
    match = re.search(pattern, filename)
    if match:
        return {
            'x_shift': float(match.group(1)),
            'y_shift': float(match.group(2)),
            'd_param': float(match.group(3)),
            'omega': float(match.group(4)),
            'a_vx': float(match.group(5)),
            'a_va': float(match.group(6))
        }
    return None

def extract_parity_flips(file_path):
    """Extract parity flip counts from a results file."""
    parity_flips = {0: None, 1: None, 2: None, 3: None}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                for i in range(4):
                    if f"Eigenstate {i}:" in line and "parity flips" in line:
                        parity_flips[i] = int(line.split(':')[1].strip().split()[0])
        return parity_flips
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return parity_flips

def collect_results(results_dir):
    """Collect results from all summary files in the directory."""
    results = []
    
    # Find all summary files
    summary_files = glob.glob(os.path.join(results_dir, "improved_berry_phase_summary_*.txt"))
    
    for file_path in summary_files:
        # Extract parameters from filename
        params = extract_parameters_from_filename(os.path.basename(file_path))
        if params:
            # Extract parity flips from file content
            parity_flips = extract_parity_flips(file_path)
            
            # Add to results
            results.append({
                'file_path': file_path,
                'params': params,
                'parity_flips': parity_flips
            })
    
    return results

def plot_parity_flips_vs_y_shift(results, output_dir):
    """
    Plot parity flips vs y_shift for each eigenstate.
    
    Args:
        results: List of dictionaries with simulation results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by x_shift
    x_shifts = set(result['params']['x_shift'] for result in results)
    
    for x_shift in x_shifts:
        # Filter results for this x_shift
        filtered_results = [r for r in results if r['params']['x_shift'] == x_shift]
        
        # Sort by y_shift
        filtered_results.sort(key=lambda r: r['params']['y_shift'])
        
        # Extract data
        y_shifts = [r['params']['y_shift'] for r in filtered_results]
        
        # Create figure with 4 subplots (one for each eigenstate)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Parity Flips vs y_shift (x_shift={x_shift})', fontsize=16)
        
        for i, ax in enumerate(axs.flat):
            # Extract parity flips for this eigenstate
            parity_flips = [r['parity_flips'][i] for r in filtered_results]
            
            # Plot
            ax.plot(y_shifts, parity_flips, 'o-', markersize=8)
            ax.set_title(f'Eigenstate {i}')
            ax.set_xlabel('y_shift')
            ax.set_ylabel('Parity Flips')
            ax.grid(True)
            
            # Set y-axis to integer values
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parity_flips_vs_y_shift_x{x_shift}.png'))
        plt.close()

def plot_eigenstate3_parity_flips_vs_y_shift(results, output_dir):
    """
    Plot eigenstate 3 parity flips vs y_shift for all x_shifts.
    
    Args:
        results: List of dictionaries with simulation results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by x_shift
    x_shifts = sorted(set(result['params']['x_shift'] for result in results))
    
    plt.figure(figsize=(10, 6))
    
    for x_shift in x_shifts:
        # Filter results for this x_shift
        filtered_results = [r for r in results if r['params']['x_shift'] == x_shift]
        
        # Sort by y_shift
        filtered_results.sort(key=lambda r: r['params']['y_shift'])
        
        # Extract data
        y_shifts = [r['params']['y_shift'] for r in filtered_results]
        parity_flips = [r['parity_flips'][3] for r in filtered_results]
        
        # Plot
        plt.plot(y_shifts, parity_flips, 'o-', markersize=8, label=f'x_shift={x_shift}')
    
    plt.title('Eigenstate 3 Parity Flips vs y_shift', fontsize=16)
    plt.xlabel('y_shift')
    plt.ylabel('Parity Flips')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis to integer values
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenstate3_parity_flips_vs_y_shift.png'))
    plt.close()

def create_heatmap(results, output_dir):
    """
    Create a heatmap of eigenstate 3 parity flips vs x_shift and y_shift.
    
    Args:
        results: List of dictionaries with simulation results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique x_shift and y_shift values
    x_shifts = sorted(set(result['params']['x_shift'] for result in results))
    y_shifts = sorted(set(result['params']['y_shift'] for result in results))
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(y_shifts), len(x_shifts)))
    
    # Fill the array with parity flip counts
    for i, y in enumerate(y_shifts):
        for j, x in enumerate(x_shifts):
            # Find the result with this x_shift and y_shift
            for result in results:
                if result['params']['x_shift'] == x and result['params']['y_shift'] == y:
                    heatmap_data[i, j] = result['parity_flips'][3] if result['parity_flips'][3] is not None else np.nan
                    break
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Eigenstate 3 Parity Flips')
    
    # Set x and y tick labels
    plt.xticks(range(len(x_shifts)), [f"{x:.1f}" for x in x_shifts])
    plt.yticks(range(len(y_shifts)), [f"{y:.1f}" for y in y_shifts])
    
    plt.xlabel('x_shift')
    plt.ylabel('y_shift')
    plt.title('Heatmap of Eigenstate 3 Parity Flips')
    
    # Add text annotations
    for i in range(len(y_shifts)):
        for j in range(len(x_shifts)):
            if not np.isnan(heatmap_data[i, j]):
                plt.text(j, i, f"{int(heatmap_data[i, j])}", 
                         ha="center", va="center", color="white" if heatmap_data[i, j] > 2 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenstate3_parity_flips_heatmap.png'))
    plt.close()

def main():
    """Main function to create visualizations."""
    # Define directories
    base_dir = "parameter_analysis"
    new_analysis_dir = os.path.join(base_dir, "new_analysis")
    results_dir = os.path.join(new_analysis_dir, "results")
    plots_dir = os.path.join(new_analysis_dir, "visualization")
    
    # Check if the new analysis directory exists
    if os.path.exists(new_analysis_dir):
        # Collect results from the new analysis
        results = collect_results(results_dir)
        
        if results:
            # Create visualizations
            plot_parity_flips_vs_y_shift(results, plots_dir)
            plot_eigenstate3_parity_flips_vs_y_shift(results, plots_dir)
            
            # Create heatmap if we have enough data points
            if len(set(r['params']['x_shift'] for r in results)) > 1:
                create_heatmap(results, plots_dir)
            
            print(f"Visualizations created in {plots_dir}")
        else:
            print("No results found in the new analysis directory.")
    else:
        print(f"Directory {new_analysis_dir} does not exist.")

if __name__ == "__main__":
    main()
