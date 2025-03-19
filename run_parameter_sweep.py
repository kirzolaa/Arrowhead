#!/usr/bin/env python3
"""
Run Parameter Sweep

This script runs multiple simulations with different parameter values to collect
data for phase transition analysis.
"""

import os
import numpy as np
import subprocess
import time

def run_simulation(x_shift, y_shift, d_param, omega, a_vx, a_va, theta_step=1):
    """Run a simulation with the given parameters."""
    cmd = [
        "python3", "run_arrowhead_simulation.py",
        "--x_shift", str(x_shift),
        "--y_shift", str(y_shift),
        "--d_param", str(d_param),
        "--omega", str(omega),
        "--a_vx", str(a_vx),
        "--a_va", str(a_va),
        "--theta_step", str(theta_step),
        "--use_improved_berry"
    ]
    
    print(f"Running simulation with parameters: x_shift={x_shift}, y_shift={y_shift}")
    subprocess.run(cmd, check=True)
    
    # Give the system a moment to complete file operations
    time.sleep(1)

def main():
    """Main function to run parameter sweep."""
    # Base parameters
    x_shift = 22.5
    base_y_shift = 547.7222222222222
    d_param = 0.005
    omega = 0.025
    a_vx = 0.018
    a_va = 0.42
    
    # Create a range of y_shift values around the base value
    # These values are chosen to capture the phase transition
    y_shift_values = np.linspace(base_y_shift - 20, base_y_shift + 20, 9)
    
    for y_shift in y_shift_values:
        run_simulation(x_shift, y_shift, d_param, omega, a_vx, a_va)
    
    print("Parameter sweep completed.")
    print("Run plot_phase_transitions.py to visualize the phase transitions.")

if __name__ == "__main__":
    main()
