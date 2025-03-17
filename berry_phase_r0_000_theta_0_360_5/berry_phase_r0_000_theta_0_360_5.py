#!/usr/bin/env python3
"""
Berry Phase Calculation for Perfect Circle

This script generates a perfect circle with 72 points between 0 and 360 degrees,
visualizes it, calculates the eigenproblem for a 4x4 arrowhead matrix for each point,
and calculates the Berry phase.

The script performs the following steps:
1. Generate a perfect circle orthogonal to the x=y=z line
2. Visualize the circle with enhanced visualization
3. Calculate the eigenproblem for a 4x4 arrowhead matrix for each point
4. Calculate and visualize the Berry phase
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import subprocess
import cmath
import argparse

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../generalized')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../generalized/example_use/arrowhead_matrix')))

# Import necessary modules
from generalized.example_use.arrowhead_matrix.arrowhead import ArrowheadMatrixAnalyzer
from generalized.example_use.arrowhead_matrix.file_utils import organize_results_directory, get_file_path

class BerryPhaseCalculator:
    """
    A class for calculating and visualizing Berry phase for a perfect circle.
    """
    
    def __init__(self, 
                 R_0=(0, 0, 0), 
                 d=1.0, 
                 theta_steps=72,
                 coupling_constant=0.1, 
                 omega=1.0, 
                 matrix_size=4, 
                 output_dir=None):
        """
        Initialize the BerryPhaseCalculator.
        
        Parameters:
        -----------
        R_0 : tuple
            Origin vector (x, y, z)
        d : float
            Distance parameter
        theta_steps : int
            Number of theta values to generate matrices for
        coupling_constant : float
            Coupling constant for off-diagonal elements
        omega : float
            Angular frequency for the energy term h*ω
        matrix_size : int
            Size of the matrix to generate
        output_dir : str
            Directory to save results
        """
        self.R_0 = R_0
        self.d = d
        self.theta_steps = theta_steps
        self.coupling_constant = coupling_constant
        self.omega = omega
        self.matrix_size = matrix_size
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'berry_phase_logs')
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the ArrowheadMatrixAnalyzer
        self.analyzer = ArrowheadMatrixAnalyzer(
            R_0=self.R_0,
            d=self.d,
            theta_start=0,
            theta_end=2*np.pi,
            theta_steps=self.theta_steps,
            coupling_constant=self.coupling_constant,
            omega=self.omega,
            matrix_size=self.matrix_size,
            perfect=True,
            output_dir=self.output_dir
        )
        
        # Initialize data structures for results
        self.berry_phases = []
        self.eigenvalues = []
        self.eigenvectors = []
    
    def generate_circle_and_matrices(self):
        """
        Generate the perfect circle and arrowhead matrices.
        
        Returns:
        --------
        list
            List of generated matrices
        """
        print("Generating perfect circle and arrowhead matrices...")
        
        # Generate matrices
        matrices = self.analyzer.generate_matrices()
        
        # Calculate eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = self.analyzer.calculate_eigenvalues_eigenvectors()
        
        return matrices
    
    def calculate_berry_phase(self):
        """
        Calculate the Berry phase for each eigenstate using multiple methods:
        1. Phase method: γ = sum of phases between consecutive eigenvectors
        2. Connection method: γ = integral of Berry connection
        
        For a perfect circle in parameter space, the Berry phase should be related to
        the solid angle subtended by the path, which is 2π(1-cos(θ)) where θ is the
        angle between the normal to the plane and the reference axis.
        
        For a circle in the plane orthogonal to (1,1,1), the solid angle is 2π
        and the Berry phase is π (half the solid angle).
        
        Returns:
        --------
        list
            List of Berry phases for each eigenstate
        """
        print("\nCalculating Berry phase...")
        
        # Number of eigenstates
        n_states = self.matrix_size
        
        # Initialize Berry phases
        self.berry_phases = [0.0] * n_states
        self.berry_phases_connection = [0.0] * n_states
        
        # Calculate the theoretical geometric phase for a perfect circle
        # For a circle in the plane orthogonal to (1,1,1), the Berry phase is π
        theoretical_phase = np.pi
        print(f"Theoretical geometric phase for perfect circle: {theoretical_phase:.6f} radians = {theoretical_phase * 180 / np.pi:.6f} degrees")
        
        # For each eigenstate
        for state_idx in range(n_states):
            # Calculate the Berry phase for this eigenstate using phase method
            berry_phase = 0.0
            berry_phase_connection = 0.0
            
            # Store all overlaps and phase corrections for logging
            overlaps = []
            phases = []
            phase_corrections = []
            connections = []
            
            # Fix the phase of the eigenvectors to ensure continuity
            fixed_eigenvectors = []
            
            # Start with the first eigenvector
            fixed_eigenvectors.append(self.eigenvectors[0][:, state_idx])
            
            # Fix the phase of subsequent eigenvectors
            for i in range(1, self.theta_steps):
                current_eigenvector = fixed_eigenvectors[-1]  # Last fixed eigenvector
                next_eigenvector = self.eigenvectors[i][:, state_idx]
                
                # Calculate the overlap
                overlap = np.vdot(current_eigenvector, next_eigenvector)
                
                # Record if phase correction was needed
                phase_correction = False
                
                # If the overlap is negative, flip the sign of the next eigenvector
                if np.real(overlap) < 0:
                    next_eigenvector = -next_eigenvector
                    overlap = -overlap
                    phase_correction = True
                
                phase_corrections.append(phase_correction)
                fixed_eigenvectors.append(next_eigenvector)
            
            # Calculate the Berry phase using the fixed eigenvectors
            for i in range(self.theta_steps):
                # Get the current and next eigenvectors (with fixed phases)
                current_idx = i
                next_idx = (i + 1) % self.theta_steps
                
                # For the last step, we need to compare with the first eigenvector
                if next_idx == 0:
                    # Get the first eigenvector and ensure its phase is consistent
                    first_eigenvector = fixed_eigenvectors[0]
                    last_eigenvector = fixed_eigenvectors[-1]
                    
                    # Calculate the overlap
                    overlap = np.vdot(last_eigenvector, first_eigenvector)
                    
                    # Record if phase correction was needed
                    phase_correction = False
                    
                    # If the overlap is negative, we need to adjust the phase
                    if np.real(overlap) < 0:
                        first_eigenvector = -first_eigenvector
                        overlap = -overlap
                        phase_correction = True
                    
                    # Calculate the phase
                    phase = np.angle(overlap)
                    
                    # For the last step, also record the phase correction
                    if i == self.theta_steps - 1:
                        phase_corrections.append(phase_correction)
                else:
                    # Calculate the overlap between consecutive eigenvectors
                    overlap = np.vdot(fixed_eigenvectors[current_idx], fixed_eigenvectors[next_idx])
                    
                    # Calculate the phase
                    phase = np.angle(overlap)
                
                overlaps.append(overlap)
                phases.append(phase)
                
                # Add the phase to the Berry phase
                berry_phase += phase
                
                # Calculate Berry connection (Method 2)
                # A_n(R) = i⟨ψ_n(R)|∇_R|ψ_n(R)⟩
                # For discrete points, we approximate the gradient using finite differences
                if next_idx != 0:  # Skip the last point to avoid double counting
                    # Calculate the Berry connection
                    # For a circular path, we can use the phase difference divided by the angle step
                    dtheta = 2 * np.pi / self.theta_steps
                    connection = -1j * (overlap - 1) / dtheta
                    connections.append(connection.imag)  # Only the imaginary part contributes
                    
                    # Add to the Berry phase calculated using the connection method
                    berry_phase_connection += connection.imag * dtheta
            
            # Store the Berry phases
            self.berry_phases[state_idx] = berry_phase
            self.berry_phases_connection[state_idx] = berry_phase_connection
            
            print(f"Berry phase for eigenstate {state_idx} (Phase method): {berry_phase:.6f} radians = {berry_phase * 180 / np.pi:.6f} degrees")
            print(f"Berry phase for eigenstate {state_idx} (Connection method): {berry_phase_connection:.6f} radians = {berry_phase_connection * 180 / np.pi:.6f} degrees")
            print(f"Difference from theoretical for eigenstate {state_idx} (Connection method): {abs(berry_phase_connection - theoretical_phase):.6f} radians")
            
            # Save detailed overlap information for this eigenstate
            overlap_file = os.path.join(self.output_dir, 'text', f'eigenstate_{state_idx}_overlaps.txt')
            os.makedirs(os.path.dirname(overlap_file), exist_ok=True)
            
            with open(overlap_file, 'w') as f:
                f.write(f"Overlap Details for Eigenstate {state_idx}\n")
                f.write("=====================================\n\n")
                f.write("Point | Overlap | Phase Corrected | Phase (radians) | Phase (degrees) | Berry Connection\n")
                f.write("-----------------------------------------------------------------------------------\n")
                
                for i, (ovlp, phase, corrected) in enumerate(zip(overlaps, phases, phase_corrections)):
                    connection_val = connections[i] if i < len(connections) else "N/A"
                    f.write(f"{i} -> {(i+1) % self.theta_steps} | {abs(ovlp):.6f} | {'Yes' if corrected else 'No'} | {phase:.6f} | {phase * 180 / np.pi:.6f} | {connection_val}\n")
                
                f.write(f"\nTotal Berry Phase (Phase method): {berry_phase:.6f} radians = {berry_phase * 180 / np.pi:.6f} degrees\n")
                f.write(f"Total Berry Phase (Connection method): {berry_phase_connection:.6f} radians = {berry_phase_connection * 180 / np.pi:.6f} degrees\n")
            
            # Visualize the eigenvector evolution
            self._visualize_eigenvector_evolution(state_idx, fixed_eigenvectors)
    
    def _visualize_eigenvector_evolution(self, state_idx, fixed_eigenvectors):
        """
        Visualize the evolution of eigenvectors for a specific eigenstate
        
        Parameters:
        -----------
        state_idx : int
            Index of the eigenstate
        fixed_eigenvectors : list
            List of phase-corrected eigenvectors
        """
        # Create a directory for eigenvector visualizations
        vis_dir = os.path.join(self.output_dir, 'plots', 'eigenvector_evolution')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Convert eigenvectors to numpy array for easier manipulation
        eigenvectors_array = np.array(fixed_eigenvectors)
        
        # Create theta values array
        theta_values = np.linspace(0, 2*np.pi, self.theta_steps, endpoint=False)
        
        # Plot the real and imaginary parts of each component of the eigenvectors
        plt.figure(figsize=(12, 10))
        
        # Get the number of components in each eigenvector
        n_components = eigenvectors_array.shape[1]
        
        # Plot real parts
        for i in range(n_components):
            plt.subplot(2, n_components, i+1)
            plt.plot(theta_values, eigenvectors_array[:, i].real, 'b-', label=f'Real part')
            plt.title(f'Component {i} (Real)')
            plt.xlabel('Theta (radians)')
            plt.ylabel('Value')
            plt.grid(True)
        
        # Plot imaginary parts
        for i in range(n_components):
            plt.subplot(2, n_components, n_components+i+1)
            plt.plot(theta_values, eigenvectors_array[:, i].imag, 'r-', label=f'Imaginary part')
            plt.title(f'Component {i} (Imaginary)')
            plt.xlabel('Theta (radians)')
            plt.ylabel('Value')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'eigenstate_{state_idx}_components.png'))
        plt.close()
        
        # Plot the magnitude and phase of each component
        plt.figure(figsize=(12, 10))
        
        # Plot magnitudes
        for i in range(n_components):
            plt.subplot(2, n_components, i+1)
            plt.plot(theta_values, np.abs(eigenvectors_array[:, i]), 'g-', label=f'Magnitude')
            plt.title(f'Component {i} (Magnitude)')
            plt.xlabel('Theta (radians)')
            plt.ylabel('Value')
            plt.grid(True)
        
        # Plot phases
        for i in range(n_components):
            plt.subplot(2, n_components, n_components+i+1)
            plt.plot(theta_values, np.angle(eigenvectors_array[:, i]), 'm-', label=f'Phase')
            plt.title(f'Component {i} (Phase)')
            plt.xlabel('Theta (radians)')
            plt.ylabel('Radians')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'eigenstate_{state_idx}_mag_phase.png'))
        plt.close()
        
        # Plot the evolution of the eigenvector in the complex plane for each component
        plt.figure(figsize=(15, 12))
        
        for i in range(n_components):
            plt.subplot(2, 2, i+1)
            plt.plot(eigenvectors_array[:, i].real, eigenvectors_array[:, i].imag, 'o-', label=f'Component {i}')
            plt.scatter(eigenvectors_array[0, i].real, eigenvectors_array[0, i].imag, c='g', s=100, label='Start')
            plt.scatter(eigenvectors_array[-1, i].real, eigenvectors_array[-1, i].imag, c='r', s=100, label='End')
            plt.title(f'Component {i} in Complex Plane')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.grid(True)
            plt.legend()
            
            # Add arrows to show direction
            for j in range(0, len(eigenvectors_array)-1, max(1, len(eigenvectors_array)//20)):
                dx = eigenvectors_array[j+1, i].real - eigenvectors_array[j, i].real
                dy = eigenvectors_array[j+1, i].imag - eigenvectors_array[j, i].imag
                plt.arrow(eigenvectors_array[j, i].real, eigenvectors_array[j, i].imag, dx, dy, 
                         head_width=0.02, head_length=0.03, fc='k', ec='k', length_includes_head=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'eigenstate_{state_idx}_complex_plane.png'))
        plt.close()
        
        # Save Berry phases
        np_dir = os.path.join(self.output_dir, 'numpy')
        os.makedirs(np_dir, exist_ok=True)
        np.save(os.path.join(np_dir, "berry_phases.npy"), np.array(self.berry_phases))
        
        return self.berry_phases
    
    def visualize_berry_phase(self):
        """
        Visualize the Berry phase results.
        """
        print("\nVisualizing Berry phase results...")
        
        # Create figure for Berry phase visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Berry phases
        state_indices = np.arange(self.matrix_size)
        ax.bar(state_indices, [phase * 180 / np.pi for phase in self.berry_phases], 
               color='blue', alpha=0.7)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Eigenstate Index')
        ax.set_ylabel('Berry Phase (degrees)')
        ax.set_title('Berry Phase for Each Eigenstate')
        
        # Set x-ticks to be integer values
        ax.set_xticks(state_indices)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add text annotations with the exact values
        for i, phase in enumerate(self.berry_phases):
            phase_deg = phase * 180 / np.pi
            ax.text(i, phase_deg + np.sign(phase_deg) * 5, 
                   f'{phase_deg:.2f}°', 
                   ha='center', va='center', 
                   fontsize=9, color='black')
        
        # Save the figure
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'berry_phase_bar_chart.png'), dpi=300, bbox_inches='tight')
        
        # Create figure for eigenvalue evolution
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot eigenvalues as a function of theta
        theta_values = np.linspace(0, 360, self.theta_steps, endpoint=False)
        
        for state_idx in range(self.matrix_size):
            eigenvals = [self.eigenvalues[i][state_idx] for i in range(self.theta_steps)]
            ax.plot(theta_values, eigenvals, '-', linewidth=2, 
                   label=f'Eigenstate {state_idx} (Berry Phase: {self.berry_phases[state_idx] * 180 / np.pi:.2f}°)')
        
        # Set labels and title
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Eigenvalue Evolution Around the Circle')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='best')
        
        # Save the figure
        plt.savefig(os.path.join(plots_dir, 'eigenvalue_evolution.png'), dpi=300, bbox_inches='tight')
        
        # Create a log file with detailed results
        log_file = os.path.join(self.output_dir, 'text', 'berry_phase_results.txt')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write("Berry Phase Calculation Results\n")
            f.write("==============================\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  R_0: {self.R_0}\n")
            f.write(f"  d: {self.d}\n")
            f.write(f"  theta_steps: {self.theta_steps}\n")
            f.write(f"  coupling_constant: {self.coupling_constant}\n")
            f.write(f"  omega: {self.omega}\n")
            f.write(f"  matrix_size: {self.matrix_size}\n\n")
            
            f.write("Theoretical Geometric Phase:\n")
            f.write(f"  Perfect Circle: {np.pi:.6f} radians = {180.0:.6f} degrees\n\n")
            
            f.write("Berry Phase Results (Phase Method):\n")
            for i, phase in enumerate(self.berry_phases):
                f.write(f"  Eigenstate {i}: {phase:.6f} radians = {phase * 180 / np.pi:.6f} degrees\n")
            
            f.write("\nBerry Phase Results (Connection Method):\n")
            for i, phase in enumerate(self.berry_phases_connection):
                f.write(f"  Eigenstate {i}: {phase:.6f} radians = {phase * 180 / np.pi:.6f} degrees\n")
                f.write(f"  Difference from theoretical: {abs(phase - np.pi):.6f} radians\n")
            
            f.write("\nEigenvalue Statistics:\n")
            for state_idx in range(self.matrix_size):
                eigenvals = [self.eigenvalues[i][state_idx] for i in range(self.theta_steps)]
                f.write(f"  Eigenstate {state_idx}:\n")
                f.write(f"    Mean: {np.mean(eigenvals):.6f}\n")
                f.write(f"    Min: {np.min(eigenvals):.6f}\n")
                f.write(f"    Max: {np.max(eigenvals):.6f}\n")
                f.write(f"    Range: {np.max(eigenvals) - np.min(eigenvals):.6f}\n")
        
        print(f"Berry phase results saved to {log_file}")
        print(f"Visualizations saved to {plots_dir}")

def main():
    """
    Main function to run the Berry phase calculation.
    """
    parser = argparse.ArgumentParser(description='Calculate Berry phase for a perfect circle.')
    parser.add_argument('--r0', type=float, nargs=3, default=[0, 0, 0],
                        help='Origin vector (x, y, z)')
    parser.add_argument('--d', type=float, default=1.0,
                        help='Distance parameter')
    parser.add_argument('--theta-steps', type=int, default=72,
                        help='Number of theta values to generate matrices for')
    parser.add_argument('--coupling', type=float, default=0.1,
                        help='Coupling constant for off-diagonal elements')
    parser.add_argument('--omega', type=float, default=1.0,
                        help='Angular frequency for the energy term h*ω')
    parser.add_argument('--matrix-size', type=int, default=4,
                        help='Size of the matrix to generate')
    parser.add_argument('--output-dir', type=str, default='berry_phase_logs',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create the Berry phase calculator
    calculator = BerryPhaseCalculator(
        R_0=tuple(args.r0),
        d=args.d,
        theta_steps=args.theta_steps,
        coupling_constant=args.coupling,
        omega=args.omega,
        matrix_size=args.matrix_size,
        output_dir=args.output_dir
    )
    
    # Generate circle and matrices
    calculator.generate_circle_and_matrices()
    
    # Calculate Berry phase
    calculator.calculate_berry_phase()
    
    # Visualize results
    calculator.visualize_berry_phase()
    
    print("\nBerry phase calculation complete!")

if __name__ == "__main__":
    main()
