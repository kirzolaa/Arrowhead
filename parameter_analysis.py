#!/usr/bin/env python3
"""
Parameter Analysis Script for Arrowhead Simulation

This script systematically explores x_shift and y_shift values around a successful
parameter set to find optimal configurations that minimize parity flips in eigenstate 3.
"""

import os
import subprocess
import numpy as np
import shutil
import time
from datetime import datetime

# Base directory for all analysis results
BASE_DIR = "parameter_analysis"

# Subdirectories for different types of outputs
MATRIX_DIR = os.path.join(BASE_DIR, "matrix_generation")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Fixed parameters from the successful run
D_PARAM = 0.005
OMEGA = 0.025
A_VX = 0.018
A_VA = 0.42
THETA_STEP = 1

# Range for x_shift and y_shift to explore
X_SHIFT_BASE = 25.0
Y_SHIFT_BASE = 550.5
X_SHIFT_RANGE = np.linspace(-5.0, 5.0, 5)  # 5 values around the base
Y_SHIFT_RANGE = np.linspace(-50.0, 50.0, 5)  # 5 values around the base

def create_directories():
    """Create all necessary directories for the analysis."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def run_simulation(x_shift, y_shift):
    """
    Run a single simulation with the given parameters.
    
    Args:
        x_shift: The x shift value for VA potential
        y_shift: The y shift value for VA potential
        
    Returns:
        Dictionary with simulation results and paths
    """
    print(f"\n=== Running simulation with x_shift={x_shift}, y_shift={y_shift} ===")
    
    # Create parameter string for filenames
    param_str = f"x{x_shift}_y{y_shift}_d{D_PARAM}_w{OMEGA}_avx{A_VX}_ava{A_VA}_step{THETA_STEP}"
    
    # Run the simulation
    cmd = [
        "python3", "run_arrowhead_simulation.py",
        f"--x_shift={x_shift}",
        f"--y_shift={y_shift}",
        f"--d_param={D_PARAM}",
        f"--omega={OMEGA}",
        f"--a_vx={A_VX}",
        f"--a_va={A_VA}",
        f"--theta_step={THETA_STEP}",
        "--use_improved_berry"
    ]
    
    # Run the command and capture output
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save the command output to a log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"simulation_{param_str}_{timestamp}.log")
    with open(log_file, "w") as f:
        f.write(process.stdout)
        if process.stderr:
            f.write("\n\nERRORS:\n")
            f.write(process.stderr)
    
    # Copy the matrix generation log
    matrix_log_src = f"logs/generation_{param_str}.txt"
    if os.path.exists(matrix_log_src):
        shutil.copy(matrix_log_src, os.path.join(MATRIX_DIR, f"generation_{param_str}.txt"))
    
    # Copy the berry phase log
    berry_log_src = f"logs/improved_berry_{param_str}.txt"
    if os.path.exists(berry_log_src):
        shutil.copy(berry_log_src, os.path.join(LOGS_DIR, f"improved_berry_{param_str}.txt"))
    
    # Copy the results file
    results_src = f"improved_berry_phase_results/improved_berry_phase_summary_{param_str}.txt"
    results_dest = os.path.join(RESULTS_DIR, f"improved_berry_phase_summary_{param_str}.txt")
    if os.path.exists(results_src):
        shutil.copy(results_src, results_dest)
    
    # Copy the plot directory
    plot_src = f"improved_berry_phase_plots_{param_str}"
    plot_dest = os.path.join(PLOTS_DIR, f"plots_{param_str}")
    if os.path.exists(plot_src):
        if os.path.exists(plot_dest):
            shutil.rmtree(plot_dest)
        shutil.copytree(plot_src, plot_dest)
    
    # Extract parity flip counts from the results file
    parity_flips = {0: None, 1: None, 2: None, 3: None}
    if os.path.exists(results_dest):
        with open(results_dest, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Eigenstate 0:" in line:
                    parity_flips[0] = int(line.split(":")[1].strip().split()[0])
                elif "Eigenstate 1:" in line:
                    parity_flips[1] = int(line.split(":")[1].strip().split()[0])
                elif "Eigenstate 2:" in line:
                    parity_flips[2] = int(line.split(":")[1].strip().split()[0])
                elif "Eigenstate 3:" in line:
                    parity_flips[3] = int(line.split(":")[1].strip().split()[0])
    
    return {
        "x_shift": x_shift,
        "y_shift": y_shift,
        "parity_flips": parity_flips,
        "log_file": log_file,
        "results_file": results_dest,
        "plots_dir": plot_dest
    }

def analyze_results(results):
    """
    Analyze the results of all simulations.
    
    Args:
        results: List of dictionaries with simulation results
        
    Returns:
        Best configuration based on eigenstate 3 parity flips
    """
    # Sort results by eigenstate 3 parity flips
    sorted_results = sorted(results, key=lambda x: (
        x["parity_flips"][3] if x["parity_flips"][3] is not None else float('inf'),
        x["parity_flips"][0] if x["parity_flips"][0] is not None else float('inf')
    ))
    
    # Create a summary file
    summary_file = os.path.join(BASE_DIR, "analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Parameter Analysis Summary\n")
        f.write("=========================\n\n")
        f.write("Fixed Parameters:\n")
        f.write(f"  d_param: {D_PARAM}\n")
        f.write(f"  omega: {OMEGA}\n")
        f.write(f"  a_vx: {A_VX}\n")
        f.write(f"  a_va: {A_VA}\n")
        f.write(f"  theta_step: {THETA_STEP}\n\n")
        
        f.write("Results (sorted by eigenstate 3 parity flips):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'x_shift':<10} {'y_shift':<10} {'E0 flips':<10} {'E1 flips':<10} {'E2 flips':<10} {'E3 flips':<10}\n")
        f.write("-" * 80 + "\n")
        
        for result in sorted_results:
            flips = result["parity_flips"]
            f.write(f"{result['x_shift']:<10.2f} {result['y_shift']:<10.2f} "
                   f"{flips[0] if flips[0] is not None else 'N/A':<10} "
                   f"{flips[1] if flips[1] is not None else 'N/A':<10} "
                   f"{flips[2] if flips[2] is not None else 'N/A':<10} "
                   f"{flips[3] if flips[3] is not None else 'N/A':<10}\n")
        
        f.write("\n\nBest Configuration:\n")
        if sorted_results:
            best = sorted_results[0]
            f.write(f"  x_shift: {best['x_shift']}\n")
            f.write(f"  y_shift: {best['y_shift']}\n")
            f.write(f"  Eigenstate 0 flips: {best['parity_flips'][0]}\n")
            f.write(f"  Eigenstate 1 flips: {best['parity_flips'][1]}\n")
            f.write(f"  Eigenstate 2 flips: {best['parity_flips'][2]}\n")
            f.write(f"  Eigenstate 3 flips: {best['parity_flips'][3]}\n")
            f.write(f"  Results file: {best['results_file']}\n")
            f.write(f"  Plots directory: {best['plots_dir']}\n")
        else:
            f.write("  No valid results found.\n")
    
    print(f"\nAnalysis summary written to {summary_file}")
    return sorted_results[0] if sorted_results else None

def main():
    """Main function to run the parameter analysis."""
    start_time = time.time()
    print("Starting parameter analysis...")
    
    # Create directories
    create_directories()
    
    # Run simulations for all parameter combinations
    results = []
    for x_offset in X_SHIFT_RANGE:
        for y_offset in Y_SHIFT_RANGE:
            x_shift = X_SHIFT_BASE + x_offset
            y_shift = Y_SHIFT_BASE + y_offset
            result = run_simulation(x_shift, y_shift)
            results.append(result)
    
    # Analyze results
    best_config = analyze_results(results)
    
    # Print final summary
    if best_config:
        print("\n=== Analysis Complete ===")
        print(f"Best configuration:")
        print(f"  x_shift: {best_config['x_shift']}")
        print(f"  y_shift: {best_config['y_shift']}")
        print(f"  Eigenstate 3 parity flips: {best_config['parity_flips'][3]}")
    else:
        print("\n=== Analysis Complete ===")
        print("No valid results found.")
    
    elapsed_time = time.time() - start_time
    print(f"Total analysis time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
