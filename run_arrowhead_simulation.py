#!/usr/bin/env python3
"""
Combined script to run the Arrowhead matrix simulation and Berry phase calculation.

This script:
1. Generates 4x4 arrowhead matrices with specified parameters
2. Calculates Berry phases from the resulting eigenvectors
3. Creates visualizations for analysis
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

def modify_matrix_parameters(file_path, x_shift, y_shift, d_param=None, omega=None, a_vx=None, a_va=None):
    """Modify various parameters in the generate_4x4_arrowhead.py file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # VA potential shifts
        if 'x_shift =' in line:
            lines[i] = f"        x_shift = {x_shift}  # Modified x-shift value\n"
        elif 'y_shift =' in line:
            lines[i] = f"        y_shift = {y_shift}  # Modified y-shift value\n"
        
        # Distance parameter in constructor
        elif 'd=0.05' in line and d_param is not None:
            lines[i] = line.replace('d=0.05', f'd={d_param}')
        
        # Omega parameter in constructor
        elif 'omega=0.01' in line and omega is not None:
            lines[i] = line.replace('omega=0.01', f'omega={omega}')
        
        # Curvature parameter for VX potential
        elif 'a = 0.1  # Reduced from 0.5' in line and a_vx is not None:
            lines[i] = f"        a = {a_vx}  # Modified curvature parameter for VX\n"
        
        # Curvature parameter for VA potential
        elif 'a = 0.1  # Same curvature as VX' in line and a_va is not None:
            lines[i] = f"        a = {a_va}  # Modified curvature parameter for VA\n"
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Modified parameters: x_shift = {x_shift}, y_shift = {y_shift}")
    if d_param is not None:
        print(f"Modified d parameter: {d_param}")
    if omega is not None:
        print(f"Modified omega parameter: {omega}")
    if a_vx is not None:
        print(f"Modified VX curvature parameter: {a_vx}")
    if a_va is not None:
        print(f"Modified VA curvature parameter: {a_va}")

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
    parser.add_argument('--d_param', type=float, default=None, help='Distance parameter d (default: 0.05)')
    parser.add_argument('--omega', type=float, default=None, help='Angular frequency omega (default: 0.01)')
    parser.add_argument('--a_vx', type=float, default=None, help='Curvature parameter for VX potential (default: 0.1)')
    parser.add_argument('--a_va', type=float, default=None, help='Curvature parameter for VA potential (default: 0.1)')
    parser.add_argument('--theta_step', type=int, default=1, help='Theta step in degrees')
    parser.add_argument('--skip_generation', action='store_true', help='Skip matrix generation step')
    parser.add_argument('--skip_berry_phase', action='store_true', help='Skip Berry phase calculation')
    parser.add_argument('--skip_parity_plots', action='store_true', help='Skip parity flip plots')
    parser.add_argument('--use_improved_berry', action='store_true', help='Use improved Berry phase calculation with eigenstate tracking')
    args = parser.parse_args()
    
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths
    generate_script = os.path.join(base_dir, 'generalized/example_use/arrowhead_matrix/generate_4x4_arrowhead.py')
    berry_phase_script = os.path.join(base_dir, 'new_berry_phase.py')
    improved_berry_script = os.path.join(base_dir, 'run_improved_berry_phase.py')
    parity_flips_script = os.path.join(base_dir, 'plot_parity_flips.py')
    
    # Output directories and files
    results_dir = os.path.join(base_dir, 'generalized/example_use/arrowhead_matrix/results')
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create parameter string for filenames
    param_str = f'x{args.x_shift}_y{args.y_shift}'
    if args.d_param is not None:
        param_str += f'_d{args.d_param}'
    if args.omega is not None:
        param_str += f'_w{args.omega}'
    if args.a_vx is not None:
        param_str += f'_avx{args.a_vx}'
    if args.a_va is not None:
        param_str += f'_ava{args.a_va}'
    param_str += f'_step{args.theta_step}'
    
    # Log files
    generation_log = os.path.join(log_dir, f'generation_{param_str}.txt')
    berry_phase_log = os.path.join(log_dir, f'berry_phase_{param_str}.txt')
    improved_berry_log = os.path.join(log_dir, f'improved_berry_{param_str}.txt')
    parity_flips_log = os.path.join(log_dir, f'parity_flips_{param_str}.txt')
    
    # Step 1: Modify the matrix parameters and theta step
    if not args.skip_generation:
        modify_matrix_parameters(generate_script, args.x_shift, args.y_shift, 
                              args.d_param, args.omega, args.a_vx, args.a_va)
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
        if args.use_improved_berry:
            print("\n=== Step 2: Calculating Improved Berry Phases with Eigenstate Tracking ===")
            cmd = f"python3 {improved_berry_script} --input_dir {results_dir} --x_shift {args.x_shift} --y_shift {args.y_shift}"
            # Add additional parameters if provided
            if args.d_param is not None:
                cmd += f" --d_param {args.d_param}"
            if args.omega is not None:
                cmd += f" --omega {args.omega}"
            if args.a_vx is not None:
                cmd += f" --a_vx {args.a_vx}"
            if args.a_va is not None:
                cmd += f" --a_va {args.a_va}"
            exit_code = run_command(cmd, improved_berry_log)
            if exit_code != 0:
                print(f"Error calculating improved Berry phases. Check {improved_berry_log} for details.")
                return exit_code
            print(f"Improved Berry phase calculation completed. Log saved to {improved_berry_log}")
        else:
            print("\n=== Step 2: Calculating Standard Berry Phases ===")
            cmd = f"python3 {berry_phase_script} --input_dir {results_dir} --save_plots"
            exit_code = run_command(cmd, berry_phase_log)
            if exit_code != 0:
                print(f"Error calculating Berry phases. Check {berry_phase_log} for details.")
                return exit_code
            print(f"Berry phase calculation completed. Log saved to {berry_phase_log}")
    
    # Step 4: Generate parity flip plots if the script exists and not using improved Berry
    if not args.skip_parity_plots and os.path.exists(parity_flips_script) and not args.use_improved_berry:
        print("\n=== Step 3: Generating Parity Flip Plots ===")
        cmd = f"python3 {parity_flips_script}"
        exit_code = run_command(cmd, parity_flips_log)
        if exit_code != 0:
            print(f"Error generating parity flip plots. Check {parity_flips_log} for details.")
            return exit_code
        print(f"Parity flip plots generated. Log saved to {parity_flips_log}")
    
    print("\n=== Simulation Complete ===")
    print(f"Parameters: x_shift={args.x_shift}, y_shift={args.y_shift}, theta_step={args.theta_step}")
    if args.d_param is not None:
        print(f"d parameter: {args.d_param}")
    if args.omega is not None:
        print(f"omega parameter: {args.omega}")
    if args.a_vx is not None:
        print(f"VX curvature parameter: {args.a_vx}")
    if args.a_va is not None:
        print(f"VA curvature parameter: {args.a_va}")
    if args.use_improved_berry:
        print("Used improved Berry phase calculation with eigenstate tracking")
        plot_dir_suffix = f"_x{args.x_shift}_y{args.y_shift}"
        if args.d_param is not None:
            plot_dir_suffix += f"_d{args.d_param}"
        if args.omega is not None:
            plot_dir_suffix += f"_w{args.omega}"
        if args.a_vx is not None:
            plot_dir_suffix += f"_avx{args.a_vx}"
        if args.a_va is not None:
            plot_dir_suffix += f"_ava{args.a_va}"
        print(f"Visualizations saved to improved_berry_phase_plots{plot_dir_suffix}/")
        print(f"Results saved to improved_berry_phase_results/ with filename improved_berry_phase_summary{plot_dir_suffix}.txt")
    print(f"All logs saved to {log_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
