#!/usr/bin/env python3
"""
Run a single simulation with the optimal parameters and save results to parameter_analysis directory.
"""

import os
import subprocess
import shutil
from datetime import datetime

# Create parameter_analysis directory and subdirectories
BASE_DIR = "parameter_analysis"
MATRIX_DIR = os.path.join(BASE_DIR, "matrix_generation")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Optimal parameters from previous runs
X_SHIFT = 25.0
Y_SHIFT = 550.5
D_PARAM = 0.005
OMEGA = 0.025
A_VX = 0.018
A_VA = 0.42
THETA_STEP = 1

def create_directories():
    """Create all necessary directories for the analysis."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def run_optimal_simulation():
    """Run a simulation with the optimal parameters and save all outputs."""
    print(f"\n=== Running simulation with optimal parameters ===")
    print(f"x_shift: {X_SHIFT}")
    print(f"y_shift: {Y_SHIFT}")
    print(f"d_param: {D_PARAM}")
    print(f"omega: {OMEGA}")
    print(f"a_vx: {A_VX}")
    print(f"a_va: {A_VA}")
    
    # Create parameter string for filenames
    param_str = f"x{X_SHIFT}_y{Y_SHIFT}_d{D_PARAM}_w{OMEGA}_avx{A_VX}_ava{A_VA}_step{THETA_STEP}"
    
    # Run the simulation
    cmd = [
        "python3", "run_arrowhead_simulation.py",
        f"--x_shift={X_SHIFT}",
        f"--y_shift={Y_SHIFT}",
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
    log_file = os.path.join(LOGS_DIR, f"optimal_simulation_{timestamp}.log")
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
    
    print(f"\n=== Simulation complete ===")
    print(f"All outputs saved to {BASE_DIR} directory")
    print(f"Results file: {results_dest}")
    print(f"Plots directory: {plot_dest}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    create_directories()
    run_optimal_simulation()
