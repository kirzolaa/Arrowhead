#!/usr/bin/env python3
"""
Focused Parameter Analysis Script for Arrowhead Simulation

This script explores y_shift values between 525.5 and 550.5 for x_shift=22.5
to find the optimal configuration that minimizes parity flips in eigenstate 3.
"""

import os
import subprocess
import numpy as np
import shutil
import time
from datetime import datetime

# Base directory for all analysis results
BASE_DIR = "parameter_analysis/new_analysis"

# Subdirectories for different types of outputs
MATRIX_DIR = os.path.join(BASE_DIR, "matrix_generation")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Fixed parameters
X_SHIFT = 22.5
D_PARAM = 0.005
OMEGA = 0.025
A_VX = 0.018
A_VA = 0.42
THETA_STEP = 1

# Range for y_shift to explore - 10 values between 525.5 and 550.5
Y_SHIFT_RANGE = np.linspace(525.5, 550.5, 10)

def create_directories():
    """Create all necessary directories for the analysis."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def run_simulation(y_shift):
    """
    Run a single simulation with the given parameters.
    
    Args:
        y_shift: The y shift value for VA potential
        
    Returns:
        Dictionary with simulation results and paths
    """
    print(f"\n=== Running simulation with x_shift={X_SHIFT}, y_shift={y_shift} ===")
    
    # Create parameter string for filenames
    param_str = f"x{X_SHIFT}_y{y_shift}_d{D_PARAM}_w{OMEGA}_avx{A_VX}_ava{A_VA}_step{THETA_STEP}"
    
    # Run the simulation
    cmd = [
        "python3", "run_arrowhead_simulation.py",
        f"--x_shift={X_SHIFT}",
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
        "x_shift": X_SHIFT,
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
        f.write("Focused Parameter Analysis Summary\n")
        f.write("================================\n\n")
        f.write("Fixed Parameters:\n")
        f.write(f"  x_shift: {X_SHIFT}\n")
        f.write(f"  d_param: {D_PARAM}\n")
        f.write(f"  omega: {OMEGA}\n")
        f.write(f"  a_vx: {A_VX}\n")
        f.write(f"  a_va: {A_VA}\n")
        f.write(f"  theta_step: {THETA_STEP}\n\n")
        
        f.write("Results (sorted by eigenstate 3 parity flips):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'y_shift':<10} {'E0 flips':<10} {'E1 flips':<10} {'E2 flips':<10} {'E3 flips':<10}\n")
        f.write("-" * 80 + "\n")
        
        for result in sorted_results:
            flips = result["parity_flips"]
            f.write(f"{result['y_shift']:<10.2f} "
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

def save_best_configuration(best_config):
    """
    Save the best configuration to a dedicated directory.
    
    Args:
        best_config: Dictionary with the best configuration
    """
    if not best_config:
        return
    
    best_dir = os.path.join(BASE_DIR, "best_configuration")
    os.makedirs(best_dir, exist_ok=True)
    
    # Create parameter string for filenames
    param_str = f"x{best_config['x_shift']}_y{best_config['y_shift']}_d{D_PARAM}_w{OMEGA}_avx{A_VX}_ava{A_VA}_step{THETA_STEP}"
    
    # Copy the results file
    if os.path.exists(best_config['results_file']):
        shutil.copy(best_config['results_file'], os.path.join(best_dir, "results.txt"))
    
    # Copy the plots directory
    if os.path.exists(best_config['plots_dir']):
        plots_dest = os.path.join(best_dir, "plots")
        if os.path.exists(plots_dest):
            shutil.rmtree(plots_dest)
        shutil.copytree(best_config['plots_dir'], plots_dest)
    
    # Create a summary file
    summary_file = os.path.join(best_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("Best Configuration Summary\n")
        f.write("========================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  x_shift: {best_config['x_shift']}\n")
        f.write(f"  y_shift: {best_config['y_shift']}\n")
        f.write(f"  d_param: {D_PARAM}\n")
        f.write(f"  omega: {OMEGA}\n")
        f.write(f"  a_vx: {A_VX}\n")
        f.write(f"  a_va: {A_VA}\n\n")
        
        # Extract parity flip counts
        flips = best_config['parity_flips']
        f.write("Parity Flip Summary:\n")
        f.write("------------------\n")
        f.write(f"Eigenstate 0: {flips[0]} parity flips\n")
        f.write(f"Eigenstate 1: {flips[1]} parity flips\n")
        f.write(f"Eigenstate 2: {flips[2]} parity flips\n")
        f.write(f"Eigenstate 3: {flips[3]} parity flips\n")
    
    print(f"Best configuration summary written to {summary_file}")

def main():
    """Main function to run the parameter analysis."""
    start_time = time.time()
    print("Starting focused parameter analysis...")
    print(f"Fixed x_shift: {X_SHIFT}")
    print(f"Exploring y_shift values between {Y_SHIFT_RANGE[0]} and {Y_SHIFT_RANGE[-1]}")
    
    # Create directories
    create_directories()
    
    # Run simulations for all y_shift values
    results = []
    for y_shift in Y_SHIFT_RANGE:
        result = run_simulation(y_shift)
        results.append(result)
    
    # Analyze results
    best_config = analyze_results(results)
    
    # Save best configuration
    if best_config:
        save_best_configuration(best_config)
        
        # Print final summary
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
