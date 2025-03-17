#!/usr/bin/env python
"""
Analyze eigenvalues from the arrowhead system to check for dips and discontinuities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set up paths
data_dir = "berry_phase_r0_000_theta_0_360_5/berry_phase_logs"
results_dir = "arrowhead_r0_00_theta_0_360_cpu_results/plots"

# Load eigenvalues
eigenvalues = []
num_theta_steps = 72  # Based on the number of files

for i in range(num_theta_steps):
    eigenvalue_file = os.path.join(data_dir, f"eigenvalues_theta_{i}.npy")
    if os.path.exists(eigenvalue_file):
        eigenvalue = np.load(eigenvalue_file)
        eigenvalues.append(eigenvalue)
    else:
        print(f"Warning: Missing eigenvalue file {eigenvalue_file}")

eigenvalues = np.array(eigenvalues)
print(f"Eigenvalues shape: {eigenvalues.shape}")

# Create theta values (0 to 360 degrees)
theta_values = np.linspace(0, 360, num_theta_steps, endpoint=False)

# Analyze eigenvalues for dips
print("\nEigenvalue Analysis:")
for state_idx in range(eigenvalues.shape[1]):
    state_values = eigenvalues[:, state_idx]
    dips = np.where(np.diff(state_values) < -0.01)[0]
    
    print(f"\nState {state_idx}:")
    print(f"  Min value: {np.min(state_values):.6f}")
    print(f"  Max value: {np.max(state_values):.6f}")
    print(f"  Range: {np.max(state_values) - np.min(state_values):.6f}")
    
    if len(dips) > 0:
        print(f"  Found {len(dips)} dips at theta values:")
        for dip_idx in dips:
            print(f"    Theta = {theta_values[dip_idx]:.2f}° → {theta_values[dip_idx+1]:.2f}°, " 
                  f"Drop: {state_values[dip_idx]:.6f} → {state_values[dip_idx+1]:.6f} "
                  f"(Δ = {state_values[dip_idx+1] - state_values[dip_idx]:.6f})")

# Create a detailed visualization
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# Plot 1: Full eigenvalue spectrum
ax1 = fig.add_subplot(gs[0, :])
for state_idx in range(eigenvalues.shape[1]):
    ax1.plot(theta_values, eigenvalues[:, state_idx], 
             label=f"State {state_idx}", linewidth=2)
ax1.set_xlabel("Theta (degrees)")
ax1.set_ylabel("Energy")
ax1.set_title("Eigenvalues vs Theta (Full Range)")
ax1.legend()
ax1.grid(True)

# Plot 2: Eigenvalue differences between adjacent points
ax2 = fig.add_subplot(gs[1, 0])
for state_idx in range(eigenvalues.shape[1]):
    diffs = np.diff(eigenvalues[:, state_idx])
    # Add the difference between last and first point to complete the circle
    diffs = np.append(diffs, eigenvalues[0, state_idx] - eigenvalues[-1, state_idx])
    ax2.plot(theta_values, diffs, label=f"State {state_idx}", linewidth=2)
ax2.set_xlabel("Theta (degrees)")
ax2.set_ylabel("Energy Difference")
ax2.set_title("Energy Differences Between Adjacent Points")
ax2.legend()
ax2.grid(True)
# Add a horizontal line at y=0
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Plot 3: Zoom in on problematic regions
ax3 = fig.add_subplot(gs[1, 1])
# Find the state with the most dips
dip_counts = []
for state_idx in range(eigenvalues.shape[1]):
    dips = np.where(np.diff(eigenvalues[:, state_idx]) < -0.01)[0]
    dip_counts.append(len(dips))

if max(dip_counts) > 0:
    problem_state = np.argmax(dip_counts)
    dips = np.where(np.diff(eigenvalues[:, problem_state]) < -0.01)[0]
    
    if len(dips) > 0:
        # Plot the problematic state
        ax3.plot(theta_values, eigenvalues[:, problem_state], 
                 label=f"State {problem_state}", linewidth=2)
        
        # Highlight the dips
        for dip_idx in dips:
            ax3.plot(theta_values[dip_idx:dip_idx+2], 
                     eigenvalues[dip_idx:dip_idx+2, problem_state], 
                     'ro-', linewidth=3, markersize=8)
            
        ax3.set_xlabel("Theta (degrees)")
        ax3.set_ylabel("Energy")
        ax3.set_title(f"Zoom on State {problem_state} with {len(dips)} Dips")
        ax3.legend()
        ax3.grid(True)
else:
    ax3.text(0.5, 0.5, "No significant dips found", 
             horizontalalignment='center', verticalalignment='center',
             transform=ax3.transAxes, fontsize=14)

plt.tight_layout()
plt.savefig("eigenvalue_analysis.png", dpi=300)
print("\nAnalysis complete. Saved visualization to eigenvalue_analysis.png")

# Physical interpretation of the dips
print("\nPhysical Interpretation:")
print("In quantum systems, eigenvalues represent energy levels of the system.")
print("Dips or discontinuities in eigenvalues as a function of a parameter can indicate:")
print("1. Avoided crossings - when energy levels approach but don't cross")
print("2. Actual level crossings - when energy levels genuinely cross")
print("3. Numerical artifacts - when the numerical diagonalization has issues")

# Check if the dips align with problematic overlaps in Berry phase calculation
print("\nConnection to Berry Phase Calculation:")
print("The dips in eigenvalues often correlate with regions where:")
print("1. Eigenvectors change rapidly, causing small overlaps")
print("2. The system undergoes a topological transition")
print("3. The Berry phase calculation might require special handling")

# Recommendations
print("\nRecommendations:")
print("1. For small dips (< 1% of energy scale): Likely numerical artifacts that can be smoothed")
print("2. For significant dips at isolated points: Check for degeneracies or near-degeneracies")
print("3. For systematic patterns of dips: Investigate the physical meaning in the context of the arrowhead system")
