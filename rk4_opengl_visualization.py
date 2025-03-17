#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
RK4 Integration Visualization using OpenGL
------------------------------------------
This script visualizes the numerical integration of differential equations
using the 4th-order Runge-Kutta method (RK4) and compares it with the analytical solution.
It uses PyOpenGL for rendering and includes performance metrics.
"""

import numpy as np
import time
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Import the RK4 integrator from our previous script
sys.path.append('.')
from integrate_rk4 import rk4, analytical_solution

# Global variables
window_width = 800
window_height = 600
display_mode = 0  # 0: dy/dt = -100y, 1: dy/dt = t*y
show_performance = True
pause_simulation = False
time_scale = 1.0  # Controls simulation speed
step_size = 0.001  # Initial step size for RK4

# Simulation state
t_current = 0.0
y_numerical = 1.0
y_analytical = 1.0
t_history = []
y_numerical_history = []
y_analytical_history = []
performance_history = []
last_update_time = 0
frame_count = 0
fps = 0

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

# OpenGL functions
def init_gl():
    """Initialize OpenGL settings"""
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(2.0)
    reshape(window_width, window_height)

def reshape(width, height):
    """Handle window resize"""
    global window_width, window_height
    window_width, window_height = width, height
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 1, 0, 1)
    glMatrixMode(GL_MODELVIEW)

def draw_text(x, y, text, color=(1, 1, 1)):
    """Draw text at the specified position"""
    glColor3f(*color)
    glRasterPos2f(x, y)
    for character in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(character))

def draw_graph():
    """Draw the graph of numerical and analytical solutions"""
    global t_history, y_numerical_history, y_analytical_history
    
    # Determine y-axis scale based on current values
    max_y = max(max(y_numerical_history) if y_numerical_history else 1.0, 
                max(y_analytical_history) if y_analytical_history else 1.0)
    min_y = min(min(y_numerical_history) if y_numerical_history else 0.0, 
                min(y_analytical_history) if y_analytical_history else 0.0)
    
    # Ensure we have some range to display
    if abs(max_y - min_y) < 1e-10:
        max_y = min_y + 1.0
    
    # Draw axes
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_LINES)
    # X-axis
    glVertex2f(0.1, 0.1)
    glVertex2f(0.9, 0.1)
    # Y-axis
    glVertex2f(0.1, 0.1)
    glVertex2f(0.1, 0.9)
    glEnd()
    
    # Draw axis labels
    draw_text(0.5, 0.02, "Time (t)")
    glPushMatrix()
    glTranslatef(0.03, 0.5, 0)
    glRotatef(90, 0, 0, 1)
    draw_text(0, 0, "Value (y)")
    glPopMatrix()
    
    # Draw tick marks and values on axes
    for i in range(11):
        x = 0.1 + 0.8 * i / 10.0
        y = 0.1 + 0.8 * i / 10.0
        
        # X-axis ticks
        glBegin(GL_LINES)
        glVertex2f(x, 0.1)
        glVertex2f(x, 0.09)
        glEnd()
        
        # Y-axis ticks
        glBegin(GL_LINES)
        glVertex2f(0.1, y)
        glVertex2f(0.09, y)
        glEnd()
        
        # Tick labels
        if i % 2 == 0:  # Only show every other tick label to avoid crowding
            t_val = t_current * i / 10.0
            draw_text(x - 0.02, 0.06, "{:.2f}".format(t_val))
            
            y_val = min_y + (max_y - min_y) * i / 10.0
            if abs(y_val) < 1e-10:
                y_str = "0"
            elif abs(y_val) < 0.01 or abs(y_val) > 100:
                y_str = "{:.1e}".format(y_val)
            else:
                y_str = "{:.2f}".format(y_val)
            draw_text(0.02, y, y_str)
    
    # Draw numerical solution (red)
    if len(t_history) > 1:
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for i in range(len(t_history)):
            x = 0.1 + 0.8 * (t_history[i] / t_current if t_current > 0 else 0)
            y_norm = (y_numerical_history[i] - min_y) / (max_y - min_y)
            y = 0.1 + 0.8 * y_norm
            glVertex2f(x, y)
        glEnd()
    
    # Draw analytical solution (green)
    if len(t_history) > 1:
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for i in range(len(t_history)):
            x = 0.1 + 0.8 * (t_history[i] / t_current if t_current > 0 else 0)
            y_norm = (y_analytical_history[i] - min_y) / (max_y - min_y)
            y = 0.1 + 0.8 * y_norm
            glVertex2f(x, y)
        glEnd()
    
    # Draw legend
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex2f(0.65, 0.95)
    glVertex2f(0.7, 0.95)
    glEnd()
    draw_text(0.72, 0.94, "Numerical")
    
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex2f(0.65, 0.92)
    glVertex2f(0.7, 0.92)
    glEnd()
    draw_text(0.72, 0.91, "Analytical")

def draw_performance_graph():
    """Draw performance metrics graph"""
    global performance_history
    
    if not performance_history:
        return
    
    # Draw border
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINE_LOOP)
    glVertex2f(0.1, 0.1)
    glVertex2f(0.9, 0.1)
    glVertex2f(0.9, 0.3)
    glVertex2f(0.1, 0.3)
    glEnd()
    
    # Draw performance data
    max_perf = max(performance_history)
    
    glColor3f(1.0, 0.7, 0.0)  # Orange
    glBegin(GL_LINE_STRIP)
    for i, perf in enumerate(performance_history):
        x = 0.1 + 0.8 * i / (len(performance_history) - 1) if len(performance_history) > 1 else 0.1
        y = 0.1 + 0.2 * (perf / max_perf if max_perf > 0 else 0)
        glVertex2f(x, y)
    glEnd()
    
    # Draw labels
    draw_text(0.12, 0.27, "Performance (steps/sec): {:.1f}".format(performance_history[-1]))
    draw_text(0.12, 0.24, "FPS: {:.1f}".format(fps))

def display():
    """Main display function"""
    global frame_count, last_update_time, fps
    
    # Calculate FPS
    current_time = time.time()
    frame_count += 1
    
    if current_time - last_update_time > 1.0:  # Update FPS every second
        fps = frame_count / (current_time - last_update_time)
        frame_count = 0
        last_update_time = current_time
    
    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    
    # Draw the main graph
    draw_graph()
    
    # Draw performance graph if enabled
    if show_performance:
        draw_performance_graph()
    
    # Draw current values and parameters
    equation_name = "dy/dt = -100y" if display_mode == 0 else "dy/dt = t*y"
    draw_text(0.1, 0.97, "Equation: {}".format(equation_name))
    draw_text(0.1, 0.94, "Current time: t = {:.6f}".format(t_current))
    draw_text(0.1, 0.91, "Numerical: y = {:.10e}".format(y_numerical))
    draw_text(0.1, 0.88, "Analytical: y = {:.10e}".format(y_analytical))
    draw_text(0.1, 0.85, "Error: {:.10e}".format(abs(y_numerical - y_analytical)))
    draw_text(0.1, 0.82, "Step size: h = {:.6f}".format(step_size))
    draw_text(0.1, 0.79, "Time scale: {:.1f}x".format(time_scale))
    
    status = "PAUSED" if pause_simulation else "RUNNING"
    draw_text(0.8, 0.97, status, (1.0, 1.0, 0.0) if pause_simulation else (0.0, 1.0, 0.0))
    
    # Draw help text
    draw_text(0.1, 0.05, "Controls: [Space] Pause, [R] Reset, [1/2] Change Equation, [+/-] Step Size, [</> or ,/.] Time Scale")
    
    glutSwapBuffers()

def update(value):
    """Update simulation state"""
    global t_current, y_numerical, y_analytical, t_history, y_numerical_history, y_analytical_history
    global performance_history, step_size, time_scale
    
    if not pause_simulation:
        # Measure performance
        start_time = time.time()
        
        # Choose the appropriate differential equation and analytical solution
        f = f1 if display_mode == 0 else f2
        analytical_func = analytical_solution1 if display_mode == 0 else analytical_solution2
        
        # Perform RK4 integration step
        old_t = t_current
        integration_steps = int(10 * time_scale)  # Number of integration steps per frame
        
        for _ in range(integration_steps):
            # Single RK4 step
            t_next = t_current + step_size
            y_numerical = rk4(f, y_numerical, t_current, t_next, step_size)
            t_current = t_next
        
        # Calculate analytical solution
        y_analytical = analytical_func(t_current)
        
        # Record history (limit to 1000 points to avoid memory issues)
        t_history.append(t_current)
        y_numerical_history.append(y_numerical)
        y_analytical_history.append(y_analytical)
        
        if len(t_history) > 1000:
            t_history = t_history[-1000:]
            y_numerical_history = y_numerical_history[-1000:]
            y_analytical_history = y_analytical_history[-1000:]
        
        # Calculate performance (steps per second)
        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed > 0:
            steps_per_sec = integration_steps / elapsed
            performance_history.append(steps_per_sec)
            
            if len(performance_history) > 100:
                performance_history = performance_history[-100:]
    
    # Schedule the next update
    glutPostRedisplay()
    glutTimerFunc(16, update, 0)  # ~60 FPS

def keyboard(key, x, y):
    """Handle keyboard input"""
    global pause_simulation, display_mode, step_size, time_scale, t_current
    global y_numerical, y_analytical, t_history, y_numerical_history, y_analytical_history
    
    key = key.decode('utf-8') if isinstance(key, bytes) else key
    
    if key == ' ':
        # Toggle pause
        pause_simulation = not pause_simulation
    elif key == 'r' or key == 'R':
        # Reset simulation
        t_current = 0.0
        y_numerical = 1.0
        y_analytical = 1.0
        t_history = []
        y_numerical_history = []
        y_analytical_history = []
    elif key == '1':
        # Switch to equation 1
        display_mode = 0
        t_current = 0.0
        y_numerical = 1.0
        y_analytical = 1.0
        t_history = []
        y_numerical_history = []
        y_analytical_history = []
    elif key == '2':
        # Switch to equation 2
        display_mode = 1
        t_current = 0.0
        y_numerical = 1.0
        y_analytical = 1.0
        t_history = []
        y_numerical_history = []
        y_analytical_history = []
    elif key == '+' or key == '=':
        # Increase step size
        step_size *= 2.0
    elif key == '-' or key == '_':
        # Decrease step size
        step_size /= 2.0
    elif key == '.' or key == '>':
        # Increase time scale
        time_scale *= 2.0
    elif key == ',' or key == '<':
        # Decrease time scale
        time_scale /= 2.0
    elif key == 'q' or key == 'Q' or key == chr(27):  # ESC key
        # Quit
        sys.exit(0)
    
    glutPostRedisplay()

def main():
    """Main function"""
    # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(window_width, window_height)
    glutCreateWindow(b"RK4 Integration Visualization")
    
    # Register callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(16, update, 0)
    
    # Initialize OpenGL
    init_gl()
    
    # Start the main loop
    print("RK4 Integration Visualization")
    print("----------------------------")
    print("Controls:")
    print("  Space: Pause/Resume simulation")
    print("  R: Reset simulation")
    print("  1: Switch to equation dy/dt = -100y")
    print("  2: Switch to equation dy/dt = t*y")
    print("  +/-: Increase/Decrease step size")
    print("  </>: Decrease/Increase time scale")
    print("  Q/ESC: Quit")
    
    glutMainLoop()

if __name__ == "__main__":
    main()
