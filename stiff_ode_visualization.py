#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
Stiff ODE Visualization
-----------------------
This script visualizes the performance of different numerical methods
for solving stiff ordinary differential equations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import sys

print("Starting stiff ODE visualization...")

# Import our stiff ODE solvers
from stiff_ode_solvers import (
    forward_euler, backward_euler, trapezoidal_method,
    implicit_rk4, adaptive_rk4, rosenbrock_method
)
from integrate_rk4 import rk4

def solve_and_plot_methods(title, f, df_dy, df_dt, analytical_func, y0, t0, tf, stiffness, 
                          step_sizes, filename):
    """
    Solve an ODE using different methods and plot the results
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Generate the analytical solution for comparison
    t_analytical = np.linspace(t0, tf, 1000)
    y_analytical = np.array([analytical_func(t) for t in t_analytical])
    
    # Plot the analytical solution
    plt.plot(t_analytical, y_analytical, 'k-', linewidth=2, label='Analytical')
    
    # Colors for different methods
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange']
    
    # Methods to test
    methods = [
        ("Forward Euler", forward_euler, colors[0]),
        ("Backward Euler", backward_euler, colors[1]),
        ("Trapezoidal", trapezoidal_method, colors[2]),
        ("Standard RK4", rk4, colors[3]),
        ("Implicit RK4", implicit_rk4, colors[4]),
        ("Adaptive RK4", lambda f, y0, t0, tf, h: adaptive_rk4(f, y0, t0, tf, h)[0], colors[5])
    ]
    
    # Add Rosenbrock method
    methods.append(("Rosenbrock", 
                   lambda f, y0, t0, tf, h: rosenbrock_method(f, y0, t0, tf, h, df_dy, df_dt),
                   colors[6]))
    
    # Results table
    print("\nResults for {}:".format(title))
    print("-" * 100)
    header = "{:<15} {:<10} {:<20} {:<15} {:<15}".format(
        "Method", "Step Size", "Result", "Error", "Time (s)")
    print(header)
    print("-" * 100)
    
    # Plot each method with different step sizes
    for method_name, method_func, color in methods:
        for i, h in enumerate(step_sizes):
            # Skip combinations that would be unstable for explicit methods
            if method_name in ["Forward Euler", "Standard RK4"] and h * stiffness > 2.0:
                print("{:<15} {:<10.6f} {:<20} {:<15} {:<15}".format(
                    method_name, h, "Unstable", "N/A", "N/A"))
                continue
            
            try:
                # Generate solution points
                t_values = np.arange(t0, tf + h/2, h)
                y_values = []
                
                # Measure time
                start_time = time.time()
                
                # For methods that return a single value
                if method_name != "Adaptive RK4":
                    # Solve step by step to get the full trajectory
                    y_current = y0
                    for t_current in t_values[:-1]:
                        t_next = t_current + h
                        y_current = method_func(f, y_current, t_current, t_next, h)
                        y_values.append(y_current)
                    
                    # Add the initial value
                    y_values = [y0] + y_values
                else:
                    # For adaptive RK4, get the full history
                    _, info = adaptive_rk4(f, y0, t0, tf, h)
                    t_values = info['t_history']
                    y_values = info['y_history']
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Calculate error at the final point
                if analytical_func:
                    final_result = y_values[-1]
                    analytical_result = analytical_func(tf)
                    error = abs(final_result - analytical_result)
                    error_str = "{:.6e}".format(error)
                else:
                    error_str = "N/A"
                
                print("{:<15} {:<10.6f} {:<20.10e} {:<15} {:<15.6f}".format(
                    method_name, h, y_values[-1], error_str, elapsed))
                
                # Plot the numerical solution
                linestyle = '-' if i == 0 else '--' if i == 1 else ':'
                plt.plot(t_values, y_values, color=color, linestyle=linestyle, 
                         label='{} (h={})'.format(method_name, h))
                
            except Exception as e:
                print("{:<15} {:<10.6f} {:<20} {:<15} {:<15}".format(
                    method_name, h, "Error: " + str(e)[:20], "N/A", "N/A"))
    
    # Set up the plot
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend(loc='best')
    
    # Save the figure
    plt.savefig(filename)
    print("\nPlot saved to {}".format(filename))
    plt.close()

def plot_stability_regions():
    """
    Plot stability regions for different numerical methods
    """
    plt.figure(figsize=(10, 8))
    
    # Define the complex plane grid
    real = np.linspace(-4, 2, 200)
    imag = np.linspace(-3, 3, 200)
    re, im = np.meshgrid(real, imag)
    z = re + 1j * im
    
    # Stability functions for different methods
    # Forward Euler: |1 + z| <= 1
    forward_euler_stability = np.abs(1 + z) <= 1
    
    # Backward Euler: |1 / (1 - z)| <= 1
    backward_euler_stability = np.abs(1 / (1 - z + 1e-16)) <= 1
    
    # Trapezoidal: |1 + z/2| / |1 - z/2| <= 1
    trapezoidal_stability = np.abs((1 + z/2) / (1 - z/2 + 1e-16)) <= 1
    
    # RK4: |1 + z + z^2/2 + z^3/6 + z^4/24| <= 1
    rk4_stability = np.abs(1 + z + z**2/2 + z**3/6 + z**4/24) <= 1
    
    # Plot stability regions
    plt.contourf(re, im, forward_euler_stability, levels=[0, 0.5], colors=['white', 'red'], alpha=0.3)
    plt.contourf(re, im, backward_euler_stability, levels=[0, 0.5], colors=['white', 'blue'], alpha=0.3)
    plt.contourf(re, im, trapezoidal_stability, levels=[0, 0.5], colors=['white', 'green'], alpha=0.3)
    plt.contourf(re, im, rk4_stability, levels=[0, 0.5], colors=['white', 'purple'], alpha=0.3)
    
    # Add labels and legend
    plt.title('Stability Regions for Different Numerical Methods')
    plt.xlabel('Re(lambda*h)')
    plt.ylabel('Im(lambda*h)')
    plt.grid(True)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Forward Euler'),
        Patch(facecolor='blue', alpha=0.3, label='Backward Euler'),
        Patch(facecolor='green', alpha=0.3, label='Trapezoidal'),
        Patch(facecolor='purple', alpha=0.3, label='RK4')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add the imaginary axis
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Save the figure
    plt.savefig('stability_regions.png')
    print("\nStability regions plot saved to stability_regions.png")
    plt.close()

def plot_step_size_adaptation():
    """
    Plot step size adaptation for the adaptive RK4 method
    """
    # Define a stiff equation
    def f_stiff(t, y, k=100):
        return -k * y
    
    # Solve using adaptive RK4
    y0 = 1.0
    t0 = 0.0
    tf = 1.0
    h0 = 0.01
    
    # Get the solution and history
    _, info = adaptive_rk4(f_stiff, y0, t0, tf, h0)
    
    # Extract history
    t_history = info['t_history']
    y_history = info['y_history']
    h_history = info['h_history']
    
    # Plot the solution and step sizes
    plt.figure(figsize=(12, 8))
    
    # Plot the solution
    plt.subplot(2, 1, 1)
    plt.plot(t_history, y_history, 'b-', label='Numerical Solution')
    
    # Plot the analytical solution
    t_analytical = np.linspace(t0, tf, 1000)
    y_analytical = np.exp(-100 * t_analytical)
    plt.plot(t_analytical, y_analytical, 'r--', label='Analytical Solution')
    
    plt.title('Adaptive RK4 Solution for dy/dt = -100y')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    
    # Plot the step sizes
    plt.subplot(2, 1, 2)
    plt.step(t_history[:-1], h_history, 'g-', where='post')
    plt.title('Step Size Adaptation')
    plt.xlabel('Time')
    plt.ylabel('Step Size')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('adaptive_step_size.png')
    print("\nAdaptive step size plot saved to adaptive_step_size.png")
    plt.close()

def main():
    """Main function to run all visualizations"""
    print("Stiff ODE Visualization")
    print("=======================")
    
    # Example 1: Linear stiff equation dy/dt = -ky
    def f_stiff(t, y, k=100):
        return -k * y
    
    def df_dy_stiff(t, y, k=100):
        return -k
    
    def df_dt_stiff(t, y, k=100):
        return 0
    
    def analytical_stiff(t, k=100):
        return np.exp(-k * t)
    
    # Test with different stiffness values
    for stiffness in [10, 100, 1000]:
        f = lambda t, y: f_stiff(t, y, stiffness)
        df_dy = lambda t, y: df_dy_stiff(t, y, stiffness)
        df_dt = lambda t, y: df_dt_stiff(t, y, stiffness)
        analytical = lambda t: analytical_stiff(t, stiffness)
        
        title = "Stiff Equation: dy/dt = -{}y".format(stiffness)
        filename = "stiff_equation_k{}.png".format(stiffness)
        
        # Choose step sizes based on stiffness
        if stiffness == 10:
            step_sizes = [0.1, 0.05, 0.01]
        elif stiffness == 100:
            step_sizes = [0.01, 0.005, 0.001]
        else:  # stiffness == 1000
            step_sizes = [0.001, 0.0005, 0.0001]
        
        solve_and_plot_methods(
            title, f, df_dy, df_dt, analytical, 
            1.0, 0.0, 1.0, stiffness, step_sizes, filename
        )
    
    # Plot stability regions
    plot_stability_regions()
    
    # Plot step size adaptation
    plot_step_size_adaptation()
    
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()
