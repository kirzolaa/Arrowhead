#!/usr/bin/env python3
"""
Plot Phase Transitions

This script collects Berry phase data from multiple simulation runs with different
parameters and creates visualizations of the phase transitions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from berry.berry_phase_visualization import plot_phase_transition

def extract_parameters_from_filename(filename):
    """Extract parameters from a filename."""
    pattern = r'x(\d+\.\d+)_y(\d+\.\d+)_d(\d+\.\d+)_w(\d+\.\d+)_avx(\d+\.\d+)_ava(\d+\.\d+)'
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

def parse_berry_phase_file(filepath):
    """Parse a Berry phase results file and extract phases and winding numbers."""
    berry_phases = {}
    winding_numbers = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the Berry phase table
    table_start = None
    table_end = None
    
    for i, line in enumerate(lines):
        if 'Eigenstate' in line and 'Raw Phase' in line and 'Winding Number' in line:
            table_start = i + 2  # Skip header and separator line
        elif table_start is not None and line.strip() == '':
            table_end = i
            break
    
    if table_start is not None and table_end is not None:
        for i in range(table_start, table_end):
            parts = lines[i].strip().split()
            if len(parts) >= 3:
                eigenstate = int(parts[0])
                # Raw phase is in radians
                berry_phase = float(parts[1])
                # Winding number
                winding_number = float(parts[2])
                
                berry_phases[eigenstate] = berry_phase
                winding_numbers[eigenstate] = winding_number
    
    return berry_phases, winding_numbers

def collect_data_for_parameter_sweep(results_dir, param_name='y_shift'):
    """Collect data from multiple simulation runs with different parameter values."""
    # Find all result files
    result_files = glob.glob(f"{results_dir}/improved_berry_phase_summary_*.txt")
    
    # Extract parameters and data
    param_values = []
    all_berry_phases = {}
    all_winding_numbers = {}
    
    for file_path in result_files:
        params = extract_parameters_from_filename(file_path)
        if params:
            param_value = params[param_name]
            berry_phases, winding_numbers = parse_berry_phase_file(file_path)
            
            param_values.append(param_value)
            all_berry_phases[param_value] = berry_phases
            all_winding_numbers[param_value] = winding_numbers
    
    # Sort by parameter value
    param_values = sorted(param_values)
    
    return param_values, all_berry_phases, all_winding_numbers

def main():
    """Main function to collect data and create phase transition plots."""
    results_dir = "improved_berry_phase_results"
    output_dir = "phase_transition_plots"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for y_shift parameter sweep
    param_name = 'y_shift'
    param_values, berry_phases, winding_numbers = collect_data_for_parameter_sweep(
        results_dir, param_name)
    
    if not param_values:
        print(f"No data found for parameter sweep of {param_name}")
        return
    
    print(f"Found {len(param_values)} different values for {param_name}: {param_values}")
    
    # Check if we have enough data points for a meaningful plot
    if len(param_values) < 2:
        print("Warning: Not enough data points for a meaningful phase transition plot.")
        print("Consider running more simulations with different parameter values.")
        print("You can use run_parameter_sweep.py to generate more data points.")
        
        # If we only have one data point, we'll create a simple bar chart instead
        if len(param_values) == 1:
            param_val = param_values[0]
            create_single_point_visualization(param_val, berry_phases[param_val], 
                                             winding_numbers[param_val], output_dir)
    else:
        # Create phase transition plots
        plot_phase_transition(param_values, berry_phases, winding_numbers, output_dir, param_name)
    
    print(f"Plots created in {output_dir}")

def create_single_point_visualization(param_val, berry_phases, winding_numbers, output_dir):
    """Create visualizations for a single parameter value."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Berry phases
    plt.figure(figsize=(10, 6))
    eigenstates = sorted(berry_phases.keys())
    phases = [berry_phases[e] for e in eigenstates]
    
    plt.bar(eigenstates, phases, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.text(eigenstates[0]-0.5, np.pi, '$\pi$', va='center', ha='right')
    plt.text(eigenstates[0]-0.5, -np.pi, '$-\pi$', va='center', ha='right')
    
    plt.xlabel('Eigenstate')
    plt.ylabel('Berry Phase (radians)')
    plt.title(f'Berry Phases for Each Eigenstate')
    plt.xticks(eigenstates)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/berry_phases_single_point.png", dpi=300)
    plt.close()
    
    # Plot winding numbers
    plt.figure(figsize=(10, 6))
    windings = [winding_numbers[e] for e in eigenstates]
    
    plt.bar(eigenstates, windings, color='green', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(eigenstates[0]-0.5, 0.5, '0.5', va='center', ha='right')
    plt.text(eigenstates[0]-0.5, -0.5, '-0.5', va='center', ha='right')
    
    plt.xlabel('Eigenstate')
    plt.ylabel('Winding Number')
    plt.title(f'Winding Numbers for Each Eigenstate')
    plt.xticks(eigenstates)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/winding_numbers_single_point.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
