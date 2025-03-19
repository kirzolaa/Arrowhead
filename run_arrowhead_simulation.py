#!/usr/bin/env python3
"""
Combined script to run the Arrowhead matrix simulation and Berry phase calculation.

This script:
1. Generates 4x4 arrowhead matrices with specified parameters
2. Calculates Berry phases from the resulting eigenvectors
3. Optionally plots parity flips
"""

import os
import sys
import subprocess
import argparse
import numpy as np

def run_command(command, output_file=None):
    """Run a shell command and optionally redirect output to a file."""
    print(f"Running: {command}")
    
    if output_file:
        with open(output_file, 'w') as f:
            result = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    return result.returncode

def modify_va_potential_shift(file_path, x_shift, y_shift):
    """Modify the VA potential's x and y shift values in the generate_4x4_arrowhead.py file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'x_shift =' in line:
            lines[i] = f"        x_shift = {x_shift}  # Modified x-shift value\n"
        elif 'y_shift =' in line:
            lines[i] = f"        y_shift = {y_shift}  # Modified y-shift value\n"
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Modified VA potential shifts: x_shift = {x_shift}, y_shift = {y_shift}")

def modify_theta_step(file_path, theta_step):
    """Modify the theta step value in the generate_4x4_arrowhead.py file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'theta_step =' in line:
            lines[i] = f"    theta_step = {theta_step}   # Modified theta step\n"
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Modified theta step to {theta_step} degrees")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Arrowhead matrix simulation and Berry phase calculation')
    parser.add_argument('--x_shift', type=float, default=2.0, help='VA potential x shift value')
    parser.add_argument('--y_shift', type=float, default=2.0, help='VA potential y shift value')
    parser.add_argument('--theta_step', type=int, default=1, help='Theta step in degrees')
    parser.add_argument('--skip_generation', action='store_true', help='Skip matrix generation step')
    parser.add_argument('--skip_berry_phase', action='store_true', help='Skip Berry phase calculation')
    parser.add_argument('--skip_parity_plots', action='store_true', help='Skip parity flip plots')
    args = parser.parse_args()
    
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths
    generate_script = os.path.join(base_dir, 'generalized/example_use/arrowhead_matrix/generate_4x4_arrowhead.py')
    berry_phase_script = os.path.join(base_dir, 'new_berry_phase.py')
    parity_flips_script = os.path.join(base_dir, 'plot_parity_flips.py')
    
    # Output directories and files
    results_dir = os.path.join(base_dir, 'generalized/example_use/arrowhead_matrix/results')
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Log files
    generation_log = os.path.join(log_dir, f'generation_x{args.x_shift}_y{args.y_shift}_step{args.theta_step}.txt')
    berry_phase_log = os.path.join(log_dir, f'berry_phase_x{args.x_shift}_y{args.y_shift}_step{args.theta_step}.txt')
    parity_flips_log = os.path.join(log_dir, f'parity_flips_x{args.x_shift}_y{args.y_shift}_step{args.theta_step}.txt')
    
    # Step 1: Modify the VA potential shifts and theta step
    if not args.skip_generation:
        modify_va_potential_shift(generate_script, args.x_shift, args.y_shift)
        modify_theta_step(generate_script, args.theta_step)
    
    # Step 2: Run the matrix generation script
    if not args.skip_generation:
        print("\n=== Step 1: Generating Arrowhead Matrices ===")
        cmd = f"python3 {generate_script}"
        exit_code = run_command(cmd, generation_log)
        if exit_code != 0:
            print(f"Error generating matrices. Check {generation_log} for details.")
            return exit_code
        print(f"Matrix generation completed. Log saved to {generation_log}")
    
    # Step 3: Run the Berry phase calculation
    if not args.skip_berry_phase:
        print("\n=== Step 2: Calculating Berry Phases ===")
        cmd = f"python3 {berry_phase_script} --input_dir {results_dir} --save_plots"
        exit_code = run_command(cmd, berry_phase_log)
        if exit_code != 0:
            print(f"Error calculating Berry phases. Check {berry_phase_log} for details.")
            return exit_code
        print(f"Berry phase calculation completed. Log saved to {berry_phase_log}")
    
    # Step 4: Generate parity flip plots if the script exists
    if not args.skip_parity_plots and os.path.exists(parity_flips_script):
        print("\n=== Step 3: Generating Parity Flip Plots ===")
        cmd = f"python3 {parity_flips_script}"
        exit_code = run_command(cmd, parity_flips_log)
        if exit_code != 0:
            print(f"Error generating parity flip plots. Check {parity_flips_log} for details.")
            return exit_code
        print(f"Parity flip plots generated. Log saved to {parity_flips_log}")
    
    print("\n=== Simulation Complete ===")
    print(f"Parameters: x_shift={args.x_shift}, y_shift={args.y_shift}, theta_step={args.theta_step}")
    print(f"All logs saved to {log_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
