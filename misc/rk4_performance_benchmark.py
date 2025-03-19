#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
RK4 Integration Performance Benchmark
------------------------------------
This script benchmarks the performance of the 4th-order Runge-Kutta method (RK4)
for solving differential equations with different parameters and configurations.
"""

import numpy as np
import time
import sys
from integrate_rk4 import rk4

def analytical_solution1(t):
    """Analytical solution to dy/dt = -100y with y(0) = 1.0"""
    return np.exp(-100 * t)

def analytical_solution2(t):
    """Analytical solution to dy/dt = t*y with y(0) = 1.0"""
    return np.exp(t**2 / 2)

def f1(t, y):
    """dy/dt = -100y"""
    return -100 * y

def f2(t, y):
    """dy/dt = t*y"""
    return t * y

def f3(t, y):
    """dy/dt = -y^3"""
    return -y**3

def f4(t, y):
    """Lorenz system (simplified to 1D for this benchmark)"""
    return 10 * (y - np.sin(t))

def benchmark_equation(eq_num, eq_name, f, analytical_func=None, y0=1.0, t0=0.0, tf=1.0):
    """Benchmark a specific differential equation with various step sizes"""
    print("\n" + "=" * 80)
    print("Benchmarking Equation {}: {}".format(eq_num, eq_name))
    print("=" * 80)
    
    step_sizes = [0.1, 0.01, 0.001, 0.0001]
    results = []
    
    for h in step_sizes:
        # Calculate number of steps
        num_steps = int((tf - t0) / h)
        
        # Measure time for integration
        start_time = time.time()
        result = rk4(f, y0, t0, tf, h)
        end_time = time.time()
        
        # Calculate performance metrics
        elapsed = end_time - start_time
        steps_per_second = num_steps / elapsed if elapsed > 0 else float('inf')
        
        # Calculate error if analytical solution is available
        error = None
        if analytical_func:
            analytical_result = analytical_func(tf)
            error = abs(result - analytical_result)
            relative_error = error / abs(analytical_result) if abs(analytical_result) > 1e-15 else float('inf')
        
        # Store results
        results.append({
            'step_size': h,
            'num_steps': num_steps,
            'result': result,
            'time': elapsed,
            'steps_per_second': steps_per_second,
            'error': error,
            'relative_error': relative_error if error is not None else None
        })
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    print("{:<10} {:<10} {:<20} {:<10} {:<12} {:<15} {:<15}".format('Step Size', 'Steps', 'Result', 'Time (s)', 'Steps/s', 'Error', 'Rel Error'))
    print("-" * 80)
    
    for r in results:
        error_str = "{:.6e}".format(r['error']) if r['error'] is not None else "N/A"
        rel_error_str = "{:.6e}".format(r['relative_error']) if r['relative_error'] is not None else "N/A"
        
        print("{:<10.6f} {:<10d} {:<20.10e} {:<10.6f} {:<12.2f} {:<15} {:<15}".format(r['step_size'], r['num_steps'], r['result'], r['time'], r['steps_per_second'], error_str, rel_error_str))
    
    if analytical_func:
        print("\nAnalytical solution at t = {}: {:.10e}".format(tf, analytical_func(tf)))
    
    return results

def convergence_analysis(eq_num, eq_name, f, analytical_func, y0=1.0, t0=0.0, tf=1.0):
    """Analyze the convergence rate of the RK4 method for a specific equation"""
    print("\n" + "=" * 80)
    print("Convergence Analysis for Equation {}: {}".format(eq_num, eq_name))
    print("=" * 80)
    
    # Use a wider range of step sizes for convergence analysis
    step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    errors = []
    
    for h in step_sizes:
        result = rk4(f, y0, t0, tf, h)
        analytical_result = analytical_func(tf)
        error = abs(result - analytical_result)
        errors.append(error)
    
    # Calculate convergence rates
    convergence_rates = []
    for i in range(1, len(step_sizes)):
        rate = np.log(errors[i-1] / errors[i]) / np.log(step_sizes[i-1] / step_sizes[i])
        convergence_rates.append(rate)
    
    # Print results
    print("\nConvergence Analysis:")
    print("-" * 80)
    print("{:<10} {:<20} {:<20}".format('Step Size', 'Error', 'Convergence Rate'))
    print("-" * 80)
    
    for i, h in enumerate(step_sizes):
        rate_str = "{:.4f}".format(convergence_rates[i-1]) if i > 0 else "N/A"
        print("{:<10.6f} {:<20.10e} {:<20}".format(h, errors[i], rate_str))
    
    # Calculate average convergence rate
    avg_rate = sum(convergence_rates) / len(convergence_rates) if convergence_rates else 0
    print("\nAverage convergence rate: {:.4f}".format(avg_rate))
    print("Theoretical convergence rate for RK4: 4.0000")
    
    return avg_rate

def vector_performance_test():
    """Test performance with vector operations vs. scalar operations"""
    print("\n" + "=" * 80)
    print("Vector vs. Scalar Operations Performance Test")
    print("=" * 80)
    
    # Define vector versions of the functions
    def f1_vector(t, y):
        return -100 * y
    
    def rk4_step_scalar(f, t, y, h):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def rk4_vector(f, y0, t0, tf, h):
        # Pre-allocate arrays
        n_steps = int((tf - t0) / h) + 1
        t = np.linspace(t0, tf, n_steps)
        y = np.zeros(n_steps)
        y[0] = y0
        
        # Perform integration
        for i in range(1, n_steps):
            y[i] = rk4_step_scalar(f, t[i-1], y[i-1], h)
        
        return y[-1]
    
    # Test parameters
    t0 = 0.0
    tf = 1.0
    y0 = 1.0
    step_sizes = [0.01, 0.001, 0.0001]
    
    print("\nResults:")
    print("-" * 80)
    print("{:<10} {:<15} {:<15} {:<10} {:<15}".format('Step Size', 'Scalar Time (s)', 'Vector Time (s)', 'Speedup', 'Result Match'))
    print("-" * 80)
    
    for h in step_sizes:
        # Scalar version
        start_time = time.time()
        result_scalar = rk4(f1, y0, t0, tf, h)
        scalar_time = time.time() - start_time
        
        # Vector version
        start_time = time.time()
        result_vector = rk4_vector(f1_vector, y0, t0, tf, h)
        vector_time = time.time() - start_time
        
        # Calculate speedup
        speedup = scalar_time / vector_time if vector_time > 0 else float('inf')
        
        # Check if results match
        results_match = abs(result_scalar - result_vector) < 1e-10
        
        print("{:<10.6f} {:<15.6f} {:<15.6f} {:<10.2f} {:<15}".format(h, scalar_time, vector_time, speedup, str(results_match)))

def stiff_equation_test():
    """Test performance on stiff equations with different step sizes"""
    print("\n" + "=" * 80)
    print("Stiff Equation Performance Test")
    print("=" * 80)
    
    # Define a range of stiffness parameters
    stiffness_values = [10, 100, 1000, 10000]
    step_sizes = [0.1, 0.01, 0.001, 0.0001]
    t0 = 0.0
    tf = 1.0
    y0 = 1.0
    
    print("\nResults for dy/dt = -k*y where k is the stiffness parameter:")
    print("-" * 100)
    print("{:<10} {:<10} {:<20} {:<10} {:<12} {:<15}".format('Stiffness', 'Step Size', 'Result', 'Time (s)', 'Steps/s', 'Error'))
    print("-" * 100)
    
    for k in stiffness_values:
        def f_stiff(t, y):
            return -k * y
        
        def analytical_stiff(t):
            return np.exp(-k * t)
        
        for h in step_sizes:
            # Skip combinations that would be unstable
            if h * k > 2.0:
                print("{:<10d} {:<10.6f} {:<20} {:<10} {:<12} {:<15}".format(k, h, 'Unstable', 'N/A', 'N/A', 'N/A'))
                continue
            
            # Measure time for integration
            start_time = time.time()
            result = rk4(f_stiff, y0, t0, tf, h)
            end_time = time.time()
            
            # Calculate performance metrics
            elapsed = end_time - start_time
            num_steps = int((tf - t0) / h)
            steps_per_second = num_steps / elapsed if elapsed > 0 else float('inf')
            
            # Calculate error
            analytical_result = analytical_stiff(tf)
            error = abs(result - analytical_result)
            
            print("{:<10d} {:<10.6f} {:<20.10e} {:<10.6f} {:<12.2f} {:<15.6e}".format(k, h, result, elapsed, steps_per_second, error))

def main():
    """Main function to run all benchmarks"""
    print("RK4 Integration Performance Benchmark")
    print("====================================")
    print("Running on Python {}".format(sys.version))
    print("NumPy version: {}".format(np.__version__))
    print("Time: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print("\n")
    
    # Benchmark different equations
    benchmark_equation(1, "dy/dt = -100y", f1, analytical_solution1)
    benchmark_equation(2, "dy/dt = t*y", f2, analytical_solution2)
    benchmark_equation(3, "dy/dt = -y^3", f3)
    benchmark_equation(4, "dy/dt = 10(y - sin(t))", f4)
    
    # Convergence analysis
    convergence_analysis(1, "dy/dt = -100y", f1, analytical_solution1, tf=0.1)
    convergence_analysis(2, "dy/dt = t*y", f2, analytical_solution2)
    
    # Vector performance test
    vector_performance_test()
    
    # Stiff equation test
    stiff_equation_test()
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
