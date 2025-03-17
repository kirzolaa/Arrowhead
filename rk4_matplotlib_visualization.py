#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
RK4 Integration Visualization using Matplotlib
----------------------------------------------
This script visualizes the numerical integration of differential equations
using the 4th-order Runge-Kutta method (RK4) and compares it with the analytical solution.
It uses Matplotlib for rendering and includes performance metrics.
"""

import numpy as np
import time
import sys
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend which should work on most systems
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import the RK4 integrator from our previous script
from integrate_rk4 import rk4

# Global variables
display_mode = 0  # 0: dy/dt = -100y, 1: dy/dt = t*y
time_scale = 1.0  # Controls simulation speed
step_size = 0.001  # Initial step size for RK4
max_points = 500  # Maximum number of points to display

# Simulation state
t_current = 0.0
y_numerical = 1.0
y_analytical = 1.0
t_history = []
y_numerical_history = []
y_analytical_history = []
performance_history = []
last_update_time = time.time()
integration_steps_per_frame = 10

# Define differential equations
def f1(t, y):
    """dy/dt = -100y"""
    return -100 * y

def f2(t, y):
    """dy/dt = t*y"""
    return t * y

def analytical_solution1(t):
    """Analytical solution to dy/dt = -100y with y(0) = 1.0"""
    return np.exp(-100 * t)

def analytical_solution2(t):
    """Analytical solution to dy/dt = t*y with y(0) = 1.0"""
    return np.exp(t**2 / 2)

def reset_simulation():
    """Reset the simulation to initial state"""
    global t_current, y_numerical, y_analytical, t_history, y_numerical_history, y_analytical_history
    t_current = 0.0
    y_numerical = 1.0
    y_analytical = 1.0
    t_history = []
    y_numerical_history = []
    y_analytical_history = []

def update_simulation():
    """Update the simulation state by one step"""
    global t_current, y_numerical, y_analytical, t_history, y_numerical_history, y_analytical_history
    global performance_history, last_update_time
    
    # Choose the appropriate differential equation and analytical solution
    f = f1 if display_mode == 0 else f2
    analytical_func = analytical_solution1 if display_mode == 0 else analytical_solution2
    
    # Measure performance
    start_time = time.time()
    
    # Perform multiple RK4 integration steps per frame
    for _ in range(integration_steps_per_frame):
        t_next = t_current + step_size
        y_numerical = rk4(f, y_numerical, t_current, t_next, step_size)
        t_current = t_next
    
    # Calculate analytical solution
    y_analytical = analytical_func(t_current)
    
    # Record history
    t_history.append(t_current)
    y_numerical_history.append(y_numerical)
    y_analytical_history.append(y_analytical)
    
    # Limit the number of points to avoid memory issues
    if len(t_history) > max_points:
        t_history = t_history[-max_points:]
        y_numerical_history = y_numerical_history[-max_points:]
        y_analytical_history = y_analytical_history[-max_points:]
    
    # Calculate performance (steps per second)
    end_time = time.time()
    elapsed = end_time - start_time
    fps = 1.0 / (end_time - last_update_time) if (end_time - last_update_time) > 0 else 0
    last_update_time = end_time
    
    if elapsed > 0:
        steps_per_sec = integration_steps_per_frame / elapsed
        performance_history.append(steps_per_sec)
        
        if len(performance_history) > 100:
            performance_history = performance_history[-100:]
    
    return fps

def init_plot():
    """Initialize the plot"""
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title('RK4 Integration Visualization')
    
    # Main plot for the solutions
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.set_title('Numerical vs Analytical Solution')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Value (y)')
    ax1.grid(True)
    
    # Plot for performance metrics
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.set_title('Performance')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Steps/second')
    ax2.grid(True)
    
    # Create empty line objects
    line_numerical, = ax1.plot([], [], 'r-', label='Numerical')
    line_analytical, = ax1.plot([], [], 'g-', label='Analytical')
    line_performance, = ax2.plot([], [], 'b-')
    
    # Add legend
    ax1.legend(loc='upper right')
    
    # Text for displaying current values
    text_info = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Text for displaying controls
    text_controls = fig.text(0.5, 0.01, 
                            'Controls: [r] Reset, [1/2] Change Equation, [+/-] Step Size, [</> or ,/.] Time Scale, [q] Quit', 
                            ha='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    return fig, ax1, ax2, line_numerical, line_analytical, line_performance, text_info

def update_plot(frame, ax1, ax2, line_numerical, line_analytical, line_performance, text_info):
    """Update function for animation"""
    # Update simulation
    fps = update_simulation()
    
    # Update main plot
    if t_history:
        line_numerical.set_data(t_history, y_numerical_history)
        line_analytical.set_data(t_history, y_analytical_history)
        
        # Adjust x and y limits
        ax1.set_xlim(0, max(t_history) * 1.1)
        
        # Use logarithmic scale if values are very small
        if min(y_numerical_history + y_analytical_history) > 0 and min(y_numerical_history + y_analytical_history) < 1e-5:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('linear')
            
        min_y = min(min(y_numerical_history), min(y_analytical_history))
        max_y = max(max(y_numerical_history), max(y_analytical_history))
        range_y = max_y - min_y
        
        if range_y > 0:
            ax1.set_ylim(min_y - 0.1 * range_y, max_y + 0.1 * range_y)
    
    # Update performance plot
    if performance_history:
        line_performance.set_data(range(len(performance_history)), performance_history)
        ax2.set_xlim(0, len(performance_history))
        ax2.set_ylim(0, max(performance_history) * 1.1)
    
    # Update info text
    equation_name = "dy/dt = -100y" if display_mode == 0 else "dy/dt = t*y"
    info_text = (
        f"Equation: {equation_name}\n"
        f"Current time: t = {t_current:.6f}\n"
        f"Numerical: y = {y_numerical:.10e}\n"
        f"Analytical: y = {y_analytical:.10e}\n"
        f"Error: {abs(y_numerical - y_analytical):.10e}\n"
        f"Step size: h = {step_size:.6f}\n"
        f"Time scale: {time_scale:.1f}x\n"
        f"FPS: {fps:.1f}"
    )
    text_info.set_text(info_text)
    
    return line_numerical, line_analytical, line_performance, text_info

def on_key_press(event, fig, animation):
    """Handle keyboard events"""
    global display_mode, step_size, time_scale, integration_steps_per_frame
    
    if event.key == 'r':
        # Reset simulation
        reset_simulation()
    elif event.key == '1':
        # Switch to equation 1
        display_mode = 0
        reset_simulation()
    elif event.key == '2':
        # Switch to equation 2
        display_mode = 1
        reset_simulation()
    elif event.key == '+' or event.key == '=':
        # Increase step size
        step_size *= 2.0
    elif event.key == '-' or event.key == '_':
        # Decrease step size
        step_size /= 2.0
    elif event.key == '.' or event.key == '>':
        # Increase time scale
        time_scale *= 2.0
        integration_steps_per_frame = int(10 * time_scale)
    elif event.key == ',' or event.key == '<':
        # Decrease time scale
        time_scale /= 2.0
        integration_steps_per_frame = max(1, int(10 * time_scale))
    elif event.key == 'q':
        # Quit
        plt.close(fig)
        sys.exit(0)

def main():
    """Main function"""
    # Initialize plot
    fig, ax1, ax2, line_numerical, line_analytical, line_performance, text_info = init_plot()
    
    # Set up animation
    animation = FuncAnimation(
        fig, 
        update_plot, 
        fargs=(ax1, ax2, line_numerical, line_analytical, line_performance, text_info),
        interval=50,  # Update every 50 ms (20 FPS)
        blit=True
    )
    
    # Set up key press event handler
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, fig, animation))
    
    # Print instructions
    print("RK4 Integration Visualization")
    print("----------------------------")
    print("Controls:")
    print("  r: Reset simulation")
    print("  1: Switch to equation dy/dt = -100y")
    print("  2: Switch to equation dy/dt = t*y")
    print("  +/-: Increase/Decrease step size")
    print("  </>: Decrease/Increase time scale")
    print("  q: Quit")
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
