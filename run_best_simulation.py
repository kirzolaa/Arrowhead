#!/usr/bin/env python3
"""
Run a simulation with the best parameters (1 parity flip in eigenstate 3)
and save all results to the parameter_analysis directory.
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
BEST_DIR = os.path.join(BASE_DIR, "best_configuration")

# Best parameters (1 parity flip in eigenstate 3)
X_SHIFT = 22.5
Y_SHIFT = 525.5
D_PARAM = 0.005
OMEGA = 0.025
A_VX = 0.018
A_VA = 0.42
THETA_STEP = 1

def create_directories():
    """Create all necessary directories."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

def run_best_simulation():
    """Run a simulation with the best parameters and save all outputs."""
    print(f"\n=== Running simulation with best parameters (1 parity flip) ===")
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
    log_file = os.path.join(LOGS_DIR, f"best_simulation_{timestamp}.log")
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
    
    # Save copies to the best_configuration directory
    if os.path.exists(matrix_log_src):
        shutil.copy(matrix_log_src, os.path.join(BEST_DIR, f"generation.txt"))
    if os.path.exists(berry_log_src):
        shutil.copy(berry_log_src, os.path.join(BEST_DIR, f"improved_berry.txt"))
    if os.path.exists(results_src):
        shutil.copy(results_src, os.path.join(BEST_DIR, f"results.txt"))
    if os.path.exists(plot_src):
        best_plots_dir = os.path.join(BEST_DIR, "plots")
        if os.path.exists(best_plots_dir):
            shutil.rmtree(best_plots_dir)
        shutil.copytree(plot_src, best_plots_dir)
    
    # Create a summary file
    summary_file = os.path.join(BEST_DIR, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("Best Configuration Summary\n")
        f.write("========================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  x_shift: {X_SHIFT}\n")
        f.write(f"  y_shift: {Y_SHIFT}\n")
        f.write(f"  d_param: {D_PARAM}\n")
        f.write(f"  omega: {OMEGA}\n")
        f.write(f"  a_vx: {A_VX}\n")
        f.write(f"  a_va: {A_VA}\n\n")
        
        # Extract parity flip counts from the results file
        if os.path.exists(results_src):
            with open(results_src, "r") as rf:
                lines = rf.readlines()
                f.write("Parity Flip Summary:\n")
                f.write("------------------\n")
                for line in lines:
                    if "Eigenstate" in line and "parity flips" in line:
                        f.write(f"{line}")
    
    print(f"\n=== Simulation complete ===")
    print(f"All outputs saved to {BASE_DIR} directory")
    print(f"Results file: {results_dest}")
    print(f"Plots directory: {plot_dest}")
    print(f"Log file: {log_file}")
    print(f"Best configuration summary: {summary_file}")

if __name__ == "__main__":
    create_directories()
    run_best_simulation()
