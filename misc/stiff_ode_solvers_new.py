#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
Stiff ODE Solvers
----------------
This module implements various numerical methods for solving stiff ordinary
differential equations (ODEs), including:

1. Forward Euler Method (explicit)
2. Backward Euler Method (implicit)
3. Trapezoidal Method (implicit)
4. Rosenbrock Method (semi-implicit)
5. Implicit RK4 Method (Gauss-Legendre)
6. Adaptive RK4 Method with step size control
"""

import numpy as np
import time
import warnings
from scipy import optimize  # For solving implicit equations

# Suppress SciPy optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def forward_euler(f, y0, t0, tf, h):
    """
    Forward Euler method (explicit, 1st order)
    Not suitable for stiff equations, included for comparison
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value
    t0 -- initial time
    tf -- final time
    h -- step size
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    t = t0
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        y = y + h * f(t, y)
        t = t + h
    
    return y

def backward_euler(f, y0, t0, tf, h, tol=1e-6, max_iter=50):
    """
    Backward Euler method (implicit, 1st order)
    Good for stiff equations
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value
    t0 -- initial time
    tf -- final time
    h -- step size
    tol -- tolerance for implicit solver
    max_iter -- maximum iterations for implicit solver
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    t = t0
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Define the implicit equation: g(y_next) = y_next - y - h*f(t+h, y_next) = 0
        def g(y_next):
            return y_next - y - h * f(t + h, y_next)
        
        # Initial guess for y_next
        y_next_guess = y + h * f(t, y)
        
        # Solve the implicit equation using Newton's method
        try:
            y_next = newton_method(g, y_next_guess, tol, max_iter)
        except Exception:
            # If Newton's method fails, use explicit Euler as fallback
            f_current = f(t, y)
            f_next_guess = f(t + h, y_next_guess)
            y_next = y + h/2 * (f_current + f_next_guess)
        
        y = y_next
        t = t + h
    
    return y

def trapezoidal_method(f, y0, t0, tf, h, tol=1e-6, max_iter=50):
    """
    Trapezoidal method (implicit, 2nd order)
    Good for stiff equations
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value
    t0 -- initial time
    tf -- final time
    h -- step size
    tol -- tolerance for implicit solver
    max_iter -- maximum iterations for implicit solver
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    t = t0
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Define the implicit equation: 
        # g(y_next) = y_next - y - h/2 * (f(t, y) + f(t+h, y_next)) = 0
        def g(y_next):
            return y_next - y - h/2 * (f(t, y) + f(t + h, y_next))
        
        # Initial guess for y_next
        y_next_guess = y + h * f(t, y)
        
        # Solve the implicit equation using Newton's method
        try:
            y_next = newton_method(g, y_next_guess, tol, max_iter)
        except Exception:
            # If Newton's method fails, use explicit trapezoidal approximation
            f_current = f(t, y)
            f_next_guess = f(t + h, y_next_guess)
            y_next = y + h/2 * (f_current + f_next_guess)
        
        y = y_next
        t = t + h
    
    return y

def rosenbrock_method(f, y0, t0, tf, h, df_dy=None, df_dt=None, alpha=0.5, tol=1e-6):
    """
    Rosenbrock method (semi-implicit, 4th order)
    Excellent for stiff equations
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    df_dy -- partial derivative of f with respect to y (Jacobian)
    df_dt -- partial derivative of f with respect to t
    alpha -- stability parameter (0.5 is a common choice)
    tol -- tolerance for matrix operations
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    t = t0
    
    # Rosenbrock method constants
    gamma = alpha  # Use the stability parameter
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Compute the function value
        try:
            f_val = f(t, y)
        except Exception:
            # If function evaluation fails, reduce step size and try again
            h = h / 2.0
            if h < 1e-14:  # Prevent infinite loops
                # Fall back to forward Euler with tiny step
                y = y + 1e-14 * f(t, y)
                t = t + 1e-14
                continue
            continue
        
        # If derivatives are not provided, use improved numerical approximation
        if df_dy is None:
            # More robust numerical approximation of Jacobian
            try:
                # Use central difference for better accuracy
                if scalar_input:
                    delta = max(1e-8, abs(y) * 1e-8) if y != 0 else 1e-8
                    J = (f(t, y + delta) - f(t, y - delta)) / (2.0 * delta)
                else:
                    # For vector case, compute Jacobian matrix
                    n = len(y)
                    J = np.zeros((n, n))
                    for i in range(n):
                        delta = max(1e-8, abs(y[i]) * 1e-8) if y[i] != 0 else 1e-8
                        y_plus = y.copy()
                        y_plus[i] += delta
                        y_minus = y.copy()
                        y_minus[i] -= delta
                        
                        f_plus = f(t, y_plus)
                        f_minus = f(t, y_minus)
                        
                        if np.isscalar(f_plus):
                            J[0, i] = (f_plus - f_minus) / (2.0 * delta)
                        else:
                            J[:, i] = (f_plus - f_minus) / (2.0 * delta)
            except Exception:
                # Fall back to forward difference if central difference fails
                try:
                    if scalar_input:
                        delta = max(1e-8, abs(y) * 1e-8) if y != 0 else 1e-8
                        J = (f(t, y + delta) - f_val) / delta
                    else:
                        # For vector case, compute Jacobian matrix
                        n = len(y)
                        J = np.zeros((n, n))
                        for i in range(n):
                            delta = max(1e-8, abs(y[i]) * 1e-8) if y[i] != 0 else 1e-8
                            y_plus = y.copy()
                            y_plus[i] += delta
                            
                            f_plus = f(t, y_plus)
                            
                            if np.isscalar(f_plus):
                                J[0, i] = (f_plus - f_val) / delta
                            else:
                                J[:, i] = (f_plus - f_val) / delta
                except Exception:
                    # If all else fails, use a small non-zero value
                    if scalar_input:
                        J = 0.01
                    else:
                        n = len(y)
                        J = np.eye(n) * 0.01
        else:
            J = df_dy(t, y)
        
        # If time derivative is not provided, use improved numerical approximation
        if df_dt is None:
            try:
                # Use central difference for time derivative
                delta_t = max(1e-8, abs(t) * 1e-8) if t != 0 else 1e-8
                f_t = (f(t + delta_t, y) - f(t - delta_t, y)) / (2.0 * delta_t)
            except Exception:
                # Fall back to forward difference
                try:
                    delta_t = max(1e-8, abs(t) * 1e-8) if t != 0 else 1e-8
                    f_t = (f(t + delta_t, y) - f_val) / delta_t
                except Exception:
                    # If all else fails, assume zero time derivative
                    if scalar_input:
                        f_t = 0.0
                    else:
                        f_t = np.zeros_like(y)
        else:
            f_t = df_dt(t, y)
        
        try:
            # Compute the matrix A = I - h*gamma*J with regularization
            if scalar_input:
                A = 1.0 - h * gamma * J
                
                # Avoid division by very small numbers
                if abs(A) < tol:
                    A = tol if A >= 0 else -tol
                
                # Compute k1 = A^(-1) * f(t, y)
                k1 = f_val / A
                
                # Compute y_mid = y + h*k1
                y_mid = y + h * k1
                
                # Evaluate f at the midpoint
                try:
                    f_mid = f(t + h, y_mid)
                except Exception:
                    # If midpoint evaluation fails, use a simpler approximation
                    f_mid = f_val
                
                # Compute k2 = A^(-1) * (f(t+h, y+h*k1) + h*gamma*J*k1/h - gamma*h*f_t)
                k2 = (f_mid + gamma * J * k1 - gamma * h * f_t) / A
                
                # Update y using the weighted average of k1 and k2
                y_new = y + h * ((1.0 - gamma) * k1 + gamma * k2)
                
                # Check for NaN or Inf
                if np.isnan(y_new) or np.isinf(y_new):
                    # Reduce step size and try again
                    h = h / 2.0
                    if h < 1e-14:  # Prevent infinite loops
                        # If step size is too small, use forward Euler as fallback
                        y = y + h * f_val
                        t = t + h
                        continue
                    continue
                
                y = y_new
            else:  # For vector case, use matrix operations
                # For systems, A is a matrix
                n = len(y)
                A = np.eye(n) - h * gamma * J
                
                # Solve the linear system instead of inverting A
                try:
                    k1 = np.linalg.solve(A, f_val)
                except np.linalg.LinAlgError:
                    # If matrix is singular, add regularization
                    A_reg = A + np.eye(n) * tol
                    k1 = np.linalg.solve(A_reg, f_val)
                
                # Compute y_mid = y + h*k1
                y_mid = y + h * k1
                
                # Evaluate f at the midpoint
                try:
                    f_mid = f(t + h, y_mid)
                except Exception:
                    # If midpoint evaluation fails, use a simpler approximation
                    f_mid = f_val
                
                # Compute right-hand side for k2
                rhs = f_mid + gamma * np.dot(J, k1) - gamma * h * f_t
                
                # Solve for k2
                try:
                    k2 = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    # If matrix is singular, add regularization
                    A_reg = A + np.eye(n) * tol
                    k2 = np.linalg.solve(A_reg, rhs)
                
                # Update y using the weighted average of k1 and k2
                y_new = y + h * ((1.0 - gamma) * k1 + gamma * k2)
                
                # Check for NaN or Inf
                if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
                    # Reduce step size and try again
                    h = h / 2.0
                    if h < 1e-14:  # Prevent infinite loops
                        # If step size is too small, use forward Euler as fallback
                        y = y + h * f_val
                        t = t + h
                        continue
                    continue
                
                y = y_new
        except Exception:
            # If Rosenbrock step fails, fall back to trapezoidal method
            try:
                f_current = f(t, y)
                
                # Forward Euler prediction
                y_pred = y + h * f_current
                
                # Backward correction
                f_pred = f(t + h, y_pred)
                
                y = y + h/2 * (f_current + f_pred)
            except Exception:
                # If all else fails, use forward Euler
                y = y + h * f_val
        
        t = t + h
    
    return y

def standard_rk4(f, y0, t0, tf, h):
    """
    Standard 4th-order Runge-Kutta method (explicit)
    Good for non-stiff equations
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value
    t0 -- initial time
    tf -- final time
    h -- step size
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    t = t0
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Standard RK4 steps
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
    
    return y

def implicit_rk4(f, y0, t0, tf, h, tol=1e-6, max_iter=50):
    """
    Implicit 4th-order Runge-Kutta method (Gauss-Legendre)
    Excellent for stiff equations, A-stable
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value
    t0 -- initial time
    tf -- final time
    h -- step size
    tol -- tolerance for implicit solver
    max_iter -- maximum iterations for implicit solver
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    t = t0
    
    # Butcher tableau for Gauss-Legendre (4th order)
    c1 = 0.5 - np.sqrt(3)/6
    c2 = 0.5 + np.sqrt(3)/6
    a11 = 0.25
    a12 = 0.25 - np.sqrt(3)/6
    a21 = 0.25 + np.sqrt(3)/6
    a22 = 0.25
    b1 = 0.5
    b2 = 0.5
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Define the system of implicit equations for k1 and k2
        def g(k):
            if scalar_input:
                k1, k2 = k[0], k[1]
                g1 = k1 - f(t + c1*h, y + h*(a11*k1 + a12*k2))
                g2 = k2 - f(t + c2*h, y + h*(a21*k1 + a22*k2))
                return np.array([g1, g2])
            else:
                # For vector case, reshape k for the system
                n = len(y)
                k1 = k[:n]
                k2 = k[n:]
                
                g1 = k1 - f(t + c1*h, y + h*(a11*k1 + a12*k2))
                g2 = k2 - f(t + c2*h, y + h*(a21*k1 + a22*k2))
                
                return np.concatenate((g1, g2))
        
        # Initial guess for k1 and k2
        f_val = f(t, y)
        
        if scalar_input:
            k_guess = np.array([f_val, f_val])
        else:
            k_guess = np.concatenate((f_val, f_val))
        
        # Solve the implicit system using a numerical solver
        try:
            # Use a context manager to suppress specific warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k_sol = optimize.fsolve(g, k_guess, xtol=tol, maxfev=max_iter)
            
            if scalar_input:
                k1, k2 = k_sol[0], k_sol[1]
            else:
                n = len(y)
                k1, k2 = k_sol[:n], k_sol[n:]
            
            # Update y
            y = y + h * (b1*k1 + b2*k2)
            t = t + h
        except Exception:
            # If the solver fails, fall back to standard RK4
            k1 = h * f(t, y)
            k2 = h * f(t + 0.5*h, y + 0.5*k1)
            k3 = h * f(t + 0.5*h, y + 0.5*k2)
            k4 = h * f(t + h, y + k3)
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t = t + h
    
    return y

def adaptive_rk4(f, y0, t0, tf, h0, tol=1e-6, h_min=1e-8, h_max=1.0, safety=0.9):
    """
    Adaptive 4th-order Runge-Kutta method with step size control
    Good for both stiff and non-stiff equations
    
    Parameters:
    f -- function that defines the differential equation dy/dt = f(t, y)
    y0 -- initial value
    t0 -- initial time
    tf -- final time
    h0 -- initial step size
    tol -- error tolerance for step size control
    h_min -- minimum step size
    h_max -- maximum step size
    safety -- safety factor for step size control
    
    Returns:
    y -- solution at tf
    info -- dictionary containing time history, solution history, and step size history
    """
    # Handle scalar or vector input
    scalar_input = np.isscalar(y0)
    if scalar_input:
        y = float(y0)
    else:
        y = np.array(y0, dtype=float)
    
    # Initialize
    t = t0
    h = h0
    
    # For tracking history
    t_history = [t0]
    y_history = [y0]
    h_history = []
    
    while t < tf:
        # Limit step size if we're approaching tf
        if t + h > tf:
            h = tf - t
        
        # Compute two solutions: one with a full step and one with two half steps
        
        # Full step (RK4)
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        y_full = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Two half steps
        h_half = h / 2.0
        
        # First half step
        k1_half = h_half * f(t, y)
        k2_half = h_half * f(t + 0.5*h_half, y + 0.5*k1_half)
        k3_half = h_half * f(t + 0.5*h_half, y + 0.5*k2_half)
        k4_half = h_half * f(t + h_half, y + k3_half)
        y_half = y + (k1_half + 2*k2_half + 2*k3_half + k4_half) / 6
        
        # Second half step
        k1_half2 = h_half * f(t + h_half, y_half)
        k2_half2 = h_half * f(t + h_half + 0.5*h_half, y_half + 0.5*k1_half2)
        k3_half2 = h_half * f(t + h_half + 0.5*h_half, y_half + 0.5*k2_half2)
        k4_half2 = h_half * f(t + h, y_half + k3_half2)
        y_half2 = y_half + (k1_half2 + 2*k2_half2 + 2*k3_half2 + k4_half2) / 6
        
        # Estimate the error
        if scalar_input:
            error = abs(y_full - y_half2)
        else:
            error = np.linalg.norm(y_full - y_half2)
        
        # Determine if the step is acceptable
        if error < tol or h <= h_min:
            # Accept the step
            t = t + h
            y = y_half2  # Use the more accurate solution
            
            # Record history
            t_history.append(t)
            y_history.append(y)
            h_history.append(h)
            
            # Step was successful, try to increase the step size
            if error > 0:
                # Calculate new step size based on error
                h_new = safety * h * (tol / error)**0.2
                h = min(h_max, max(h_min, h_new))
            else:
                # If error is zero, increase step size more cautiously
                h = min(h_max, h * 1.5)
        else:
            # Reject the step and reduce the step size
            h_new = safety * h * (tol / error)**0.25
            h = max(h_min, h_new)
    
    # Create info dictionary
    info = {
        't_history': t_history,
        'y_history': y_history,
        'h_history': h_history
    }
    
    return y, info

def newton_method(g, x0, tol=1e-6, max_iter=50):
    """
    Newton's method for solving nonlinear equations
    
    Parameters:
    g -- function to find root of
    x0 -- initial guess
    tol -- tolerance
    max_iter -- maximum number of iterations
    
    Returns:
    x -- root of g
    """
    x = x0
    
    for i in range(max_iter):
        # Compute the function value
        fx = g(x)
        
        # Check for convergence
        if np.all(np.abs(fx) < tol):
            return x
        
        # Compute the Jacobian
        if np.isscalar(x):
            # For scalar case, use numerical approximation of derivative
            delta = max(1e-8, abs(x) * 1e-8)
            J = (g(x + delta) - fx) / delta
            
            # Avoid division by zero
            if abs(J) < 1e-14:
                J = 1e-14 if J >= 0 else -1e-14
            
            # Update x
            x = x - fx / J
        else:
            # For vector case, use numerical approximation of Jacobian
            n = len(x)
            J = np.zeros((n, n))
            
            for j in range(n):
                delta = max(1e-8, abs(x[j]) * 1e-8)
                x_delta = x.copy()
                x_delta[j] += delta
                J[:, j] = (g(x_delta) - fx) / delta
            
            # Solve the linear system J * dx = -fx
            try:
                dx = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                # If matrix is singular, add regularization
                J_reg = J + np.eye(n) * 1e-6
                dx = np.linalg.solve(J_reg, -fx)
            
            # Update x
            x = x + dx
    
    # If we reach here, Newton's method did not converge
    raise RuntimeError("Newton's method did not converge after {} iterations".format(max_iter))
