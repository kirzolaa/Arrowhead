#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
Test script for the Rosenbrock method
"""

import numpy as np
import time
import sys

print("Starting Rosenbrock test...")

# Import our stiff ODE solvers
from stiff_ode_solvers import rosenbrock_method

# Simple test case: dy/dt = -50y (stiff equation)
def f(t, y):
    return -50.0 * y

# Analytical solution: y(t) = y0 * exp(-50t)
def analytical(t):
    return y0 * np.exp(-50.0 * t)

# Jacobian: df/dy = -50
def df_dy(t, y):
    return -50.0

# Time derivative: df/dt = 0
def df_dt(t, y):
    return 0.0

# Initial conditions
y0 = 1.0
t0 = 0.0
tf = 0.1
h = 0.01

print("Testing Rosenbrock method with stiff equation: dy/dt = -50y")
print("Initial value: y0 =", y0)
print("Time range: t0 =", t0, "to tf =", tf)
print("Step size: h =", h)

# Solve using Rosenbrock method
try:
    print("Solving with Rosenbrock method...")
    start_time = time.time()
    result = rosenbrock_method(f, y0, t0, tf, h, df_dy, df_dt)
    end_time = time.time()
    
    print("Rosenbrock solution at t =", tf, ":", result)
    print("Analytical solution at t =", tf, ":", analytical(tf))
    print("Error:", abs(result - analytical(tf)))
    print("Time taken:", end_time - start_time, "seconds")
except Exception as e:
    print("Error in Rosenbrock method:", str(e))

# Also test without providing derivatives
try:
    print("\nSolving with Rosenbrock method (no derivatives provided)...")
    start_time = time.time()
    result_no_deriv = rosenbrock_method(f, y0, t0, tf, h)
    end_time = time.time()
    
    print("Rosenbrock solution (no derivatives) at t =", tf, ":", result_no_deriv)
    print("Analytical solution at t =", tf, ":", analytical(tf))
    print("Error:", abs(result_no_deriv - analytical(tf)))
    print("Time taken:", end_time - start_time, "seconds")
except Exception as e:
    print("Error in Rosenbrock method (no derivatives):", str(e))

print("Test completed.")
