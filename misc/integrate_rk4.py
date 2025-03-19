#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

def rk4(f, y0, t0, tf, h, print_steps=False):
    """Runge-Kutta 4th order method for solving ODEs.
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value of y at t0
    t0 -- initial time
    tf -- final time
    h -- step size
    print_steps -- if True, print intermediate steps
    
    Returns:
    y -- solution at time tf
    """
    t = t0
    y = y0
    steps = 0
    
    if print_steps:
        print("Step\t Time\t\t Value")
        print("-" * 40)
        print("{0}\t {1:.6f}\t {2:.10e}".format(steps, t, y))
    
    while t < tf:
        # Adjust final step to exactly hit tf
        if t + h > tf:
            h = tf - t
            
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
        steps += 1
        
        if print_steps and steps % 10 == 0:
            print("{0}\t {1:.6f}\t {2:.10e}".format(steps, t, y))
    
    if print_steps:
        print("{0}\t {1:.6f}\t {2:.10e}".format(steps, t, y))
        print("-" * 40)
    
    return y

def analytical_solution(t):
    """Analytical solution to dy/dt = -100y with y(0) = 1.0"""
    return np.exp(-100 * t)

if __name__ == "__main__":
    # Example usage: solving dy/dt = -100y with y(0) = 1.0
    def f(t, y):
        return -100 * y
    
    y0 = 1.0
    t0 = 0.0
    tf = 0.1  # Reduced final time to see meaningful results
    
    print("\nSolving dy/dt = -100y with y(0) = 1.0 using RK4")
    print("Analytical solution at t = {0} is y = {1:.10e}".format(tf, analytical_solution(tf)))
    
    # Try different step sizes
    print("\nTesting different step sizes:")
    for h in [0.01, 0.001, 0.0001]:
        result = rk4(f, y0, t0, tf, h)
        print("Step size h = {0:.6f}: y({1}) = {2:.10e}".format(h, tf, result))
        print("Absolute error: {0:.10e}".format(abs(result - analytical_solution(tf))))
    
    # Show detailed steps for a small problem
    print("\nDetailed solution with h = 0.01:")
    result = rk4(f, y0, t0, tf, 0.01, print_steps=True)
    
    # Try a different equation: dy/dt = t*y with y(0) = 1.0
    print("\nSolving dy/dt = t*y with y(0) = 1.0 using RK4")
    def g(t, y):
        return t * y
    
    # Analytical solution is y = exp(t**2/2)
    def analytical_g(t):
        return np.exp(t**2 / 2)
    
    tf_g = 2.0
    result_g = rk4(g, 1.0, 0.0, tf_g, 0.1)
    print("Numerical result at t = {0}: {1:.10e}".format(tf_g, result_g))
    print("Analytical result at t = {0}: {1:.10e}".format(tf_g, analytical_g(tf_g)))
    print("Absolute error: {0:.10e}".format(abs(result_g - analytical_g(tf_g))))