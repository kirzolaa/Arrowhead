#!/usr/bin/env python3
"""
Comprehensive script to run the improved Berry phase calculation with the optimal
configuration (0 parity flips for eigenstate 3) and generate all requested plots.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import time
import shutil
import glob

# Add berry directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'berry'))

# Create output directory
OUTPUT_DIR = "optimal_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

# Optimal parameters (0 parity flips in eigenstate 3)
OPTIMAL_PARAMS = {
    "x_shift": 22.5,
    "y_shift": 547.7222222222222,
    "d_param": 0.005,
    "omega": 0.025,
    "a_vx": 0.018,
    "a_va": 0.42
}

def run_berry_phase_calculation():
    """Run the improved Berry phase calculation with optimal parameters."""
    print("Running Berry phase calculation with optimal parameters...")
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Script paths
    berry_phase_script = os.path.join(base_dir, 'berry/new_berry_phase.py')
    improved_berry_script = os.path.join(base_dir, 'run_improved_berry_phase.py')
    parity_flips_script = os.path.join(base_dir, 'berry/plot_parity_flips.py')
    
    # Construct command with optimal parameters using run_arrowhead_simulation.py
    cmd = [
        "python3", "run_arrowhead_simulation.py",
        "--x_shift", str(OPTIMAL_PARAMS["x_shift"]),
        "--y_shift", str(OPTIMAL_PARAMS["y_shift"]),
        "--d_param", str(OPTIMAL_PARAMS["d_param"]),
        "--omega", str(OPTIMAL_PARAMS["omega"]),
        "--a_vx", str(OPTIMAL_PARAMS["a_vx"]),
        "--a_va", str(OPTIMAL_PARAMS["a_va"]),
        "--theta_step", "1",
        "--use_improved_berry"
    ]
    
    print("Executing:", " ".join(cmd))
    
    # Run the command and capture output
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Get return code
    return_code = process.poll()
    
    if return_code != 0:
        print(f"Error running Berry phase calculation: {process.stderr.read()}")
        sys.exit(1)
    
    # Copy the results to our output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find the berry phase results file with the matching parameters
    param_str = f'x{OPTIMAL_PARAMS["x_shift"]}_y{OPTIMAL_PARAMS["y_shift"]}'
    if "d_param" in OPTIMAL_PARAMS:
        param_str += f'_d{OPTIMAL_PARAMS["d_param"]}'
    if "omega" in OPTIMAL_PARAMS:
        param_str += f'_w{OPTIMAL_PARAMS["omega"]}'
    if "a_vx" in OPTIMAL_PARAMS:
        param_str += f'_avx{OPTIMAL_PARAMS["a_vx"]}'
    if "a_va" in OPTIMAL_PARAMS:
        param_str += f'_ava{OPTIMAL_PARAMS["a_va"]}'
    param_str += f'_step1'
    
    # Look for berry phase results files
    berry_results_dir = "berry_phase_results"
    if os.path.exists(berry_results_dir):
        for filename in os.listdir(berry_results_dir):
            if param_str in filename and filename.endswith(".txt"):
                src_path = os.path.join(berry_results_dir, filename)
                dst_path = os.path.join(OUTPUT_DIR, "berry_phase_results.txt")
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"Copied results from {src_path} to {dst_path}")
    
    print("Berry phase calculation completed successfully.")
    return True

def parse_results_file():
    """Parse the results file to extract Berry phases and parity flips."""
    # First try the copied results file
    results_filename = f"{OUTPUT_DIR}/berry_phase_results.txt"
    
    if not os.path.exists(results_filename):
        print(f"Results file not found: {results_filename}")
        
        # Try to find the results file in berry_phase_results directory
        berry_results_dir = "berry_phase_results"
        if os.path.exists(berry_results_dir):
            # Construct the parameter string
            param_str = f'x{OPTIMAL_PARAMS["x_shift"]}_y{OPTIMAL_PARAMS["y_shift"]}'
            if "d_param" in OPTIMAL_PARAMS:
                param_str += f'_d{OPTIMAL_PARAMS["d_param"]}'
            if "omega" in OPTIMAL_PARAMS:
                param_str += f'_w{OPTIMAL_PARAMS["omega"]}'
            if "a_vx" in OPTIMAL_PARAMS:
                param_str += f'_avx{OPTIMAL_PARAMS["a_vx"]}'
            if "a_va" in OPTIMAL_PARAMS:
                param_str += f'_ava{OPTIMAL_PARAMS["a_va"]}'
            
            # Look for files matching the parameter pattern
            for filename in os.listdir(berry_results_dir):
                if param_str in filename and filename.endswith(".txt"):
                    results_filename = os.path.join(berry_results_dir, filename)
                    print(f"Found results file: {results_filename}")
                    break
        
        # If still not found, try to find any improved berry phase results
        if not os.path.exists(results_filename):
            print("No matching results file found. Looking for any improved berry phase results...")
            for root, dirs, files in os.walk("."):
                for filename in files:
                    if "improved_berry_phase" in filename and filename.endswith(".txt"):
                        results_filename = os.path.join(root, filename)
                        print(f"Using alternative results file: {results_filename}")
                        break
                if os.path.exists(results_filename):
                    break
            
            # If still not found, try to look in the logs directory for berry phase logs
            if not os.path.exists(results_filename):
                print("Looking for berry phase logs in the logs directory...")
                logs_dir = "logs"
                if os.path.exists(logs_dir):
                    for filename in os.listdir(logs_dir):
                        if ("berry_phase" in filename or "improved_berry" in filename) and filename.endswith(".txt"):
                            results_filename = os.path.join(logs_dir, filename)
                            print(f"Using log file: {results_filename}")
                            break
            
            if not os.path.exists(results_filename):
                print("No results files found.")
                return None
    
    # Parse the results file
    with open(results_filename, 'r') as f:
        content = f.read()
    
    # Extract Berry phases
    berry_phases = {}
    berry_pattern = r"Eigenstate (\d+) Berry phase: ([-+]?\d*\.\d+)"
    for match in re.finditer(berry_pattern, content):
        eigenstate = int(match.group(1))
        phase = float(match.group(2))
        berry_phases[eigenstate] = phase
    
    # Extract parity flips
    parity_flips = {}
    # Try the standard format first
    parity_pattern = r"Eigenstate (\d+) parity flips: (\d+)"
    matches = list(re.finditer(parity_pattern, content))
    
    # If no matches found, try the alternative format from the log file
    if not matches:
        parity_pattern = r"Eigenstate (\d+) had (\d+) parity flips during the cycle"
        matches = list(re.finditer(parity_pattern, content))
    
    for match in matches:
        eigenstate = int(match.group(1))
        flips = int(match.group(2))
        parity_flips[eigenstate] = flips
    
    return {
        "berry_phases": berry_phases,
        "parity_flips": parity_flips,
        "filename": results_filename
    }

def plot_berry_phases(results):
    """Plot Berry phases for all eigenstates."""
    print("Plotting Berry phases...")
    
    # Extract Berry phases
    berry_phases = results["berry_phases"]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot Berry phases
    eigenstates = sorted(berry_phases.keys())
    phases = [berry_phases[e] for e in eigenstates]
    
    plt.bar(eigenstates, phases, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Eigenstate')
    plt.ylabel('Berry Phase')
    plt.title('Berry Phases for Each Eigenstate (Optimal Configuration)')
    plt.xticks(eigenstates)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/berry_phases.png", dpi=300)
    plt.close()

def plot_parity_flips(results):
    """Plot parity flips for all eigenstates."""
    print("Plotting parity flips...")
    
    # Extract parity flips
    parity_flips = results["parity_flips"]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot parity flips
    eigenstates = sorted(parity_flips.keys())
    flips = [parity_flips[e] for e in eigenstates]
    
    # Use a different color for eigenstate 3 to highlight it
    colors = ['blue' if e != 3 else 'green' for e in eigenstates]
    
    plt.bar(eigenstates, flips, color=colors, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Eigenstate')
    plt.ylabel('Number of Parity Flips')
    plt.title('Parity Flips for Each Eigenstate (Optimal Configuration)')
    plt.xticks(eigenstates)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for each bar
    for i, v in enumerate(flips):
        plt.text(eigenstates[i], v + 0.5, str(v), ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/parity_flips.png", dpi=300)
    plt.close()

def plot_potentials():
    """Plot the VX and VA potentials with the optimal configuration."""
    print("Plotting potentials...")
    
    # Parameters
    x_shift = OPTIMAL_PARAMS["x_shift"]
    y_shift = OPTIMAL_PARAMS["y_shift"]
    d_param = OPTIMAL_PARAMS["d_param"]
    omega = OPTIMAL_PARAMS["omega"]
    a_vx = OPTIMAL_PARAMS["a_vx"]
    a_va = OPTIMAL_PARAMS["a_va"]
    
    # Create a grid of points
    x = np.linspace(-10, 10, 1000)
    
    # Calculate potentials
    vx = 0.5 * a_vx * (x - x_shift)**2
    va = 0.5 * a_va * (x - y_shift)**2
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot potentials
    plt.plot(x, vx, 'b-', label=f'VX (x_shift={x_shift})')
    plt.plot(x, va, 'r-', label=f'VA (y_shift={y_shift})')
    
    # Add labels and title
    plt.xlabel('Position (x)')
    plt.ylabel('Potential Energy')
    plt.title('VX and VA Potentials (Optimal Configuration)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/potentials.png", dpi=300)
    plt.close()

def create_comprehensive_infographic(results, eigenstate_data=None):
    """Create a comprehensive infographic with all relevant information."""
    print("Creating comprehensive infographic...")
    
    # Extract data
    berry_phases = results["berry_phases"]
    parity_flips = results["parity_flips"]
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 15))  # Increased height to accommodate eigenstate-theta plot
    
    if eigenstate_data:
        # With eigenstate-theta data, use a 4-row layout
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.8])
    else:
        # Without eigenstate-theta data, use the original 3-row layout
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])
    
    # 1. Plot Berry phases
    ax1 = fig.add_subplot(gs[0, 0])
    eigenstates = sorted(berry_phases.keys())
    phases = [berry_phases[e] for e in eigenstates]
    ax1.bar(eigenstates, phases, color='blue', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Eigenstate')
    ax1.set_ylabel('Berry Phase')
    ax1.set_title('Berry Phases')
    ax1.set_xticks(eigenstates)
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot parity flips
    ax2 = fig.add_subplot(gs[0, 1])
    flips = [parity_flips[e] for e in eigenstates]
    colors = ['blue' if e != 3 else 'green' for e in eigenstates]
    ax2.bar(eigenstates, flips, color=colors, alpha=0.7)
    ax2.set_xlabel('Eigenstate')
    ax2.set_ylabel('Number of Parity Flips')
    ax2.set_title('Parity Flips')
    ax2.set_xticks(eigenstates)
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(flips):
        ax2.text(eigenstates[i], v + 0.5, str(v), ha='center')
    
    # 3. Plot potentials
    ax3 = fig.add_subplot(gs[1, :])
    x = np.linspace(-10, 10, 1000)
    vx = 0.5 * OPTIMAL_PARAMS["a_vx"] * (x - OPTIMAL_PARAMS["x_shift"])**2
    va = 0.5 * OPTIMAL_PARAMS["a_va"] * (x - OPTIMAL_PARAMS["y_shift"])**2
    ax3.plot(x, vx, 'b-', label=f'VX (x_shift={OPTIMAL_PARAMS["x_shift"]})')
    ax3.plot(x, va, 'r-', label=f'VA (y_shift={OPTIMAL_PARAMS["y_shift"]})')
    ax3.set_xlabel('Position (x)')
    ax3.set_ylabel('Potential Energy')
    ax3.set_title('VX and VA Potentials')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Plot eigenstate vs theta (if available)
    if eigenstate_data:
        # Create normalized data
        normalized_data = {}
        all_values = np.concatenate([data[:, 1] for data in eigenstate_data.values()])
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        global_range = global_max - global_min
        
        for eigenstate, data in eigenstate_data.items():
            theta = data[:, 0]
            values = data[:, 1]
            normalized_values = (values - global_min) / global_range
            normalized_data[eigenstate] = np.column_stack((theta, normalized_values))
        
        # Plot normalized values instead of original
        ax4 = fig.add_subplot(gs[2, :])
        for eigenstate, data in normalized_data.items():
            theta = data[:, 0]
            values = data[:, 1]
            ax4.plot(theta, values, label=f'Eigenstate {eigenstate}')
        ax4.set_xlabel('Theta (degrees)')
        ax4.set_ylabel('Eigenstate Value (normalized 0-1)')
        ax4.set_title('Eigenstate vs Theta (Normalized to 0-1 Range)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Add parameter information (moved to row 3 if eigenstate data available)
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')
    else:
        # 4. Add parameter information (stays at row 2 if no eigenstate data)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
    
    # Parameter information text
    param_text = (
        f"Optimal Parameters:\n"
        f"x_shift = {OPTIMAL_PARAMS['x_shift']}\n"
        f"y_shift = {OPTIMAL_PARAMS['y_shift']}\n"
        f"d_param = {OPTIMAL_PARAMS['d_param']}\n"
        f"omega = {OPTIMAL_PARAMS['omega']}\n"
        f"a_vx = {OPTIMAL_PARAMS['a_vx']}\n"
        f"a_va = {OPTIMAL_PARAMS['a_va']}\n\n"
        f"Results:\n"
        f"Eigenstate 3 Parity Flips: {parity_flips.get(3, 'N/A')} (Target: 0)\n"
        f"Total Parity Flips: {sum(parity_flips.values())}\n"
    )
    ax5.text(0.5, 0.5, param_text, ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Add main title
    fig.suptitle('Optimal Configuration Analysis (0 Parity Flips in Eigenstate 3)', 
                 fontsize=16, y=0.98)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/plots/comprehensive_infographic.png", dpi=300)
    plt.close()

def create_summary_file(results, degeneracy_data=None):
    """Create a summary file with all the results."""
    print("Creating summary file...")
    
    summary_filename = f"{OUTPUT_DIR}/summary.txt"
    
    # Get normalization parameters if available
    norm_params = {}
    
    # First try to get normalization parameters from degeneracy data
    if degeneracy_data and "1-2" in degeneracy_data and "normalization" in degeneracy_data["1-2"]:
        norm_data = degeneracy_data["1-2"]["normalization"]
        norm_params['global_min'] = norm_data["global_min"]
        norm_params['global_max'] = norm_data["global_max"]
        norm_params['global_range'] = norm_data["global_range"]
    else:
        # Otherwise try to read from file
        norm_params_file = f"{OUTPUT_DIR}/plots/normalization_params.txt"
        if os.path.exists(norm_params_file):
            with open(norm_params_file, 'r') as f_norm:
                for line in f_norm:
                    if line.startswith("Global minimum eigenvalue:"):
                        norm_params['global_min'] = float(line.split(":")[1].strip())
                    elif line.startswith("Global maximum eigenvalue:"):
                        norm_params['global_max'] = float(line.split(":")[1].strip())
                    elif line.startswith("Global range:"):
                        norm_params['global_range'] = float(line.split(":")[1].strip())
    
    with open(summary_filename, 'w') as f:
        f.write("Optimal Configuration Analysis (0 Parity Flips in Eigenstate 3)\n")
        f.write("==========================================================\n\n")
        
        f.write("Parameters:\n")
        for param, value in OPTIMAL_PARAMS.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        f.write("Berry Phases:\n")
        for eigenstate, phase in sorted(results["berry_phases"].items()):
            f.write(f"  Eigenstate {eigenstate}: {phase}\n")
        f.write("\n")
        
        f.write("Parity Flips:\n")
        for eigenstate, flips in sorted(results["parity_flips"].items()):
            f.write(f"  Eigenstate {eigenstate}: {flips}\n")
        f.write("\n")
        
        f.write(f"Total Parity Flips: {sum(results['parity_flips'].values())}\n")
        f.write(f"Eigenstate 3 Parity Flips: {results['parity_flips'].get(3, 'N/A')} (Target: 0)\n\n")
        
        # Add eigenvalue normalization information if available
        if norm_params:
            f.write("Eigenvalue Normalization:\n")
            f.write(f"  Global Minimum: {norm_params['global_min']:.6f}\n")
            f.write(f"  Global Maximum: {norm_params['global_max']:.6f}\n")
            f.write(f"  Global Range: {norm_params['global_range']:.6f}\n")
            f.write(f"  Normalization Formula: normalized = (original - {norm_params['global_min']:.6f}) / {norm_params['global_range']:.6f}\n\n")
            f.write("  Note: All eigenstate plots and degeneracy analyses use normalized (0-1 range) values.\n\n")
        
        # Add degeneracy information if available
        if degeneracy_data and "1-2" in degeneracy_data:
            f.write("Eigenstate Degeneracy Analysis:\n")
            f.write("  Eigenstates 1-2 (Should be degenerate):\n")
            data = degeneracy_data["1-2"]
            f.write(f"    Mean Difference: {data['mean']:.6f}\n")
            f.write(f"    Min Difference: {data['min']:.6f}\n")
            f.write(f"    Max Difference: {data['max']:.6f}\n")
            f.write(f"    Std Deviation: {data['std']:.6f}\n")
            
            # Add more detailed analysis for the 1-2 pair (always using normalized values)
            if data['mean'] < 0.0005:  # 0.05% of the full range
                f.write(f"    Degeneracy Status: EXCELLENT - Mean difference is less than 0.0005 (normalized scale)\n")
            elif data['mean'] < 0.001:  # 0.1% of the full range
                f.write(f"    Degeneracy Status: GOOD - Mean difference is less than 0.001 (normalized scale)\n")
            elif data['mean'] < 0.005:  # 0.5% of the full range
                f.write(f"    Degeneracy Status: ACCEPTABLE - Mean difference is less than 0.005 (normalized scale)\n")
            else:
                f.write(f"    Degeneracy Status: POOR - Mean difference is greater than 0.005 (normalized scale)\n")
                
            # Add percentage of points with difference less than a threshold
            if 'differences' in data:
                # Always use normalized threshold
                threshold = 0.0002  # 0.02% of range for normalized values
                degenerate_points = sum(1 for diff in data['differences'] if diff < threshold)
                total_points = len(data['differences'])
                percentage = (degenerate_points / total_points) * 100 if total_points > 0 else 0
                f.write(f"    Points with difference < {threshold}: {degenerate_points}/{total_points} ({percentage:.2f}%)\n")
                
                # Add information about where degeneracy is strongest/weakest
                if 'theta_values' in data and len(data['theta_values']) == len(data['differences']):
                    min_diff_idx = np.argmin(data['differences'])
                    max_diff_idx = np.argmax(data['differences'])
                    f.write(f"    Strongest Degeneracy: At theta = {data['theta_values'][min_diff_idx]:.1f}° (diff = {data['differences'][min_diff_idx]:.6f})\n")
                    f.write(f"    Weakest Degeneracy: At theta = {data['theta_values'][max_diff_idx]:.1f}° (diff = {data['differences'][max_diff_idx]:.6f})\n")
            f.write("\n")
            
            # Add information about other pairs for comparison
            f.write("  Other Eigenstate Pairs (Should NOT be degenerate):\n")
            for pair, data in degeneracy_data.items():
                if pair != "1-2":
                    f.write(f"    Eigenstates {pair}:\n")
                    f.write(f"      Mean Difference: {data['mean']:.6f}\n")
                    f.write(f"      Min Difference: {data['min']:.6f}\n")
                    f.write(f"      Max Difference: {data['max']:.6f}\n")
                    f.write(f"      Std Deviation: {data['std']:.6f}\n")
                    
                    # Add degeneracy status for comparison (always using normalized values)
                    if data['mean'] > 0.5:  # 50% of the full range
                        f.write(f"      Degeneracy Status: GOOD - Mean difference is large (> 0.5, normalized scale)\n")
                    elif data['mean'] > 0.1:  # 10% of the full range
                        f.write(f"      Degeneracy Status: ACCEPTABLE - Mean difference is moderate (> 0.1, normalized scale)\n")
                    else:
                        f.write(f"      Degeneracy Status: CONCERN - Mean difference is small (< 0.1, normalized scale)\n")
            f.write("\n")
        
        f.write("Files:\n")
        f.write(f"  Results: {results['filename']}\n")
        f.write(f"  Plots: {OUTPUT_DIR}/plots/\n")
        f.write(f"  Summary: {summary_filename}\n")
        if norm_params:
            f.write(f"  Normalized Data: {OUTPUT_DIR}/plots/eigenstate*_vs_theta_normalized.txt\n")
    
    print(f"Summary file created: {summary_filename}")

def extract_eigenstate_theta_data():
    """Extract eigenstate vs theta data from the output files."""
    print("Extracting eigenstate vs theta data...")
    
    # Find the eigenstate vs theta data files
    eigenstate_files = []
    plots_dir = f"{OUTPUT_DIR}/plots/plots_x{OPTIMAL_PARAMS['x_shift']}_y{OPTIMAL_PARAMS['y_shift']}_d{OPTIMAL_PARAMS['d_param']}_w{OPTIMAL_PARAMS['omega']}_avx{OPTIMAL_PARAMS['a_vx']}_ava{OPTIMAL_PARAMS['a_va']}_step1"
    
    # Also look in the improved berry phase plots directory
    improved_plots_dir = f"improved_berry_phase_plots_x{OPTIMAL_PARAMS['x_shift']}_y{OPTIMAL_PARAMS['y_shift']}_d{OPTIMAL_PARAMS['d_param']}_w{OPTIMAL_PARAMS['omega']}_avx{OPTIMAL_PARAMS['a_vx']}_ava{OPTIMAL_PARAMS['a_va']}"
    
    # Check if either directory exists
    if not os.path.exists(plots_dir) and not os.path.exists(improved_plots_dir):
        print(f"Plots directories not found: {plots_dir} or {improved_plots_dir}")
        # Try to find the plots directory with a pattern match
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for d in dirs:
                if d.startswith("plots_"):
                    plots_dir = os.path.join(root, d)
                    print(f"Using alternative plots directory: {plots_dir}")
                    break
            if plots_dir != f"{OUTPUT_DIR}/plots/plots_x{OPTIMAL_PARAMS['x_shift']}_y{OPTIMAL_PARAMS['y_shift']}_d{OPTIMAL_PARAMS['d_param']}_w{OPTIMAL_PARAMS['omega']}_avx{OPTIMAL_PARAMS['a_vx']}_ava{OPTIMAL_PARAMS['a_va']}_step1":
                break
    elif os.path.exists(improved_plots_dir):
        plots_dir = improved_plots_dir
        print(f"Using improved berry phase plots directory: {plots_dir}")
    
    # Look for eigenstate data files
    eigenstate_data = {}
    
    # First try to find text files with data
    for i in range(4):  # Assuming 4 eigenstates (0, 1, 2, 3)
        data_file = f"{plots_dir}/eigenstate{i}_vs_theta.txt"
        if os.path.exists(data_file):
            eigenstate_data[i] = np.loadtxt(data_file)
            eigenstate_files.append(data_file)
    
    # If no text files found, we'll need to generate the data from the eigenvector files
    if not eigenstate_data:
        print("No eigenstate vs theta data files found. Attempting to generate from eigenvector files...")
        
        # Find the eigenvector files
        eigenvector_dir = "/home/zoli/arrowhead/Arrowhead/generalized/example_use/arrowhead_matrix/results"
        if os.path.exists(eigenvector_dir):
            # Get the list of eigenvector files
            eigenvector_files = sorted(glob.glob(os.path.join(eigenvector_dir, "eigenvectors_theta_*.npy")))
            eigenvalue_files = sorted(glob.glob(os.path.join(eigenvector_dir, "eigenvalues_theta_*.npy")))
            
            if eigenvector_files and eigenvalue_files:
                print(f"Found {len(eigenvector_files)} eigenvector files and {len(eigenvalue_files)} eigenvalue files")
                
                # Create theta values array (0 to 360 degrees)
                theta_values = np.linspace(0, 360, len(eigenvector_files))
                
                # Load eigenvalues
                eigenvalues = []
                for file in eigenvalue_files:
                    eigenvalues.append(np.load(file))
                eigenvalues = np.array(eigenvalues)
                
                # Normalize eigenvalues for better visualization
                # First, calculate the mean of each eigenstate
                eigenvalue_means = np.mean(eigenvalues, axis=0)
                
                # Subtract the mean to center around zero
                normalized_eigenvalues = eigenvalues - eigenvalue_means[np.newaxis, :]
                
                # Also save the original eigenvalues
                original_eigenstate_data = {}
                
                # Create eigenstate data with normalized values
                for i in range(eigenvalues.shape[1]):  # For each eigenstate
                    eigenstate_data[i] = np.column_stack((theta_values, normalized_eigenvalues[:, i]))
                    original_eigenstate_data[i] = np.column_stack((theta_values, eigenvalues[:, i]))
                
                # Save the data to text files for future use
                os.makedirs(plots_dir, exist_ok=True)
                for i, data in eigenstate_data.items():
                    np.savetxt(f"{plots_dir}/eigenstate{i}_vs_theta_normalized.txt", data)
                    np.savetxt(f"{plots_dir}/eigenstate{i}_vs_theta.txt", original_eigenstate_data[i])
                    print(f"Saved eigenstate {i} data to {plots_dir}/eigenstate{i}_vs_theta.txt and normalized version")
    
    if not eigenstate_data:
        print("No eigenstate vs theta data could be found or generated.")
        return None
    
    print(f"Found eigenstate vs theta data for {len(eigenstate_data)} eigenstates")
    return eigenstate_data

def analyze_eigenstate_degeneracy(eigenstate_data):
    """Analyze the degeneracy between eigenstates, particularly between 1 and 2."""
    if not eigenstate_data or 1 not in eigenstate_data or 2 not in eigenstate_data:
        print("Cannot analyze degeneracy: missing eigenstate data for states 1 and 2.")
        return None
    
    print("Analyzing eigenstate degeneracy...")
    
    # Find global min and max across all eigenstates for normalization
    all_values = np.concatenate([data[:, 1] for data in eigenstate_data.values()])
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    global_range = global_max - global_min
    
    # Calculate the difference between eigenstate 1 and 2 at each theta
    data1 = eigenstate_data[1]
    data2 = eigenstate_data[2]
    
    # Ensure the theta values match
    if len(data1) != len(data2) or not np.allclose(data1[:, 0], data2[:, 0]):
        print("Warning: Theta values for eigenstates 1 and 2 do not match.")
        return None
    
    theta = data1[:, 0]
    values1 = data1[:, 1]
    values2 = data2[:, 1]
    
    # Calculate the absolute difference
    diff_12 = np.abs(values1 - values2)
    
    # Normalize the difference to 0-1 range
    normalized_diff_12 = diff_12 / global_range
    
    # Calculate statistics on the normalized differences
    mean_diff = np.mean(normalized_diff_12)
    max_diff = np.max(normalized_diff_12)
    min_diff = np.min(normalized_diff_12)
    std_diff = np.std(normalized_diff_12)
    
    # Calculate differences for other eigenstate pairs for comparison
    degeneracy_data = {
        "1-2": {
            "mean": mean_diff,
            "max": max_diff,
            "min": min_diff,
            "std": std_diff,
            "theta_values": theta,  # Store theta values for detailed analysis
            "differences": normalized_diff_12,  # Store normalized differences for detailed analysis
            "normalization": {
                "global_min": global_min,
                "global_max": global_max,
                "global_range": global_range
            }
        }
    }
    
    # Calculate differences for other pairs if data is available
    pairs = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
    for i, j in pairs:
        if i in eigenstate_data and j in eigenstate_data:
            data_i = eigenstate_data[i]
            data_j = eigenstate_data[j]
            
            # Ensure the theta values match
            if len(data_i) == len(data_j) and np.allclose(data_i[:, 0], data_j[:, 0]):
                values_i = data_i[:, 1]
                values_j = data_j[:, 1]
                diff = np.abs(values_i - values_j)
                
                # Normalize the difference using the same global range
                normalized_diff = diff / global_range
                
                degeneracy_data[f"{i}-{j}"] = {
                    "mean": np.mean(normalized_diff),
                    "max": np.max(normalized_diff),
                    "min": np.min(normalized_diff),
                    "std": np.std(normalized_diff),
                    "theta_values": data_i[:, 0],
                    "differences": normalized_diff,
                    "normalization": {
                        "global_min": global_min,
                        "global_max": global_max,
                        "global_range": global_range
                    }
                }
    
    return degeneracy_data

def plot_eigenstate_theta(eigenstate_data):
    """Plot eigenstate vs theta for all eigenstates."""
    if not eigenstate_data:
        return
    
    print("Plotting eigenstate vs theta...")
    
    # Create normalized data by scaling to 0-1 range
    normalized_data = {}
    
    # First find global min and max across all eigenstates for consistent scaling
    all_values = np.concatenate([data[:, 1] for data in eigenstate_data.values()])
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    global_range = global_max - global_min
    
    # Save the normalization parameters to a file for reference
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    with open(f"{OUTPUT_DIR}/plots/normalization_params.txt", 'w') as f:
        f.write(f"Global minimum eigenvalue: {global_min}\n")
        f.write(f"Global maximum eigenvalue: {global_max}\n")
        f.write(f"Global range: {global_range}\n")
        f.write(f"Normalization formula: normalized = (original - {global_min}) / {global_range}\n")
    
    for eigenstate, data in eigenstate_data.items():
        theta = data[:, 0]
        values = data[:, 1]
        
        # Normalize the values to 0-1 range
        normalized_values = (values - global_min) / global_range
        
        normalized_data[eigenstate] = np.column_stack((theta, normalized_values))
        
        # Save the normalized data to files
        np.savetxt(f"{OUTPUT_DIR}/plots/eigenstate{eigenstate}_vs_theta_normalized.txt", normalized_data[eigenstate])
    
    # Create original plot
    plt.figure(figsize=(12, 8))
    
    # Plot each eigenstate (original values)
    for eigenstate, data in eigenstate_data.items():
        theta = data[:, 0]
        values = data[:, 1]
        plt.plot(theta, values, label=f'Eigenstate {eigenstate}')
    
    # Add labels and title
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Eigenstate Value')
    plt.title('Eigenstate vs Theta (Optimal Configuration)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate_vs_theta.png", dpi=300)
    plt.close()
    
    # Create normalized plot
    plt.figure(figsize=(12, 8))
    
    # Plot each eigenstate (normalized values)
    for eigenstate, data in normalized_data.items():
        theta = data[:, 0]
        values = data[:, 1]
        plt.plot(theta, values, label=f'Eigenstate {eigenstate}')
    
    # Add labels and title
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Eigenstate Value (normalized 0-1)')
    plt.title('Eigenstate vs Theta (Normalized to 0-1 Range)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add info text
    info_text = (
        "This plot shows the eigenvalues of the system\n"
        "as a function of the parameter θ (theta).\n"
        "Values are normalized to a 0-1 range for better visualization.\n"
        "Original values are around 60,000.\n\n"
        "Key features to observe:\n"
        "- Crossing/avoided crossing points\n"
        "- Periodicity of eigenvalues\n"
        "- Symmetry around specific θ values"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate_vs_theta_normalized.png", dpi=300)
    plt.close()
    
    # Also create individual plots for each eigenstate (original values)
    for eigenstate, data in eigenstate_data.items():
        plt.figure(figsize=(10, 6))
        
        theta = data[:, 0]
        values = data[:, 1]
        plt.plot(theta, values, label=f'Eigenstate {eigenstate}')
        
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Eigenstate Value')
        plt.title(f'Eigenstate {eigenstate} vs Theta (Optimal Configuration)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate{eigenstate}_vs_theta.png", dpi=300)
        plt.close()
    
    # Also create individual plots for each eigenstate (normalized values)
    for eigenstate, data in normalized_data.items():
        plt.figure(figsize=(10, 6))
        
        theta = data[:, 0]
        values = data[:, 1]
        plt.plot(theta, values, label=f'Eigenstate {eigenstate}')
        
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Eigenstate Value (normalized 0-1)')
        plt.title(f'Eigenstate {eigenstate} vs Theta (Normalized to 0-1 Range)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate{eigenstate}_vs_theta_normalized.png", dpi=300)
        plt.close()

def plot_eigenstate_degeneracy(degeneracy_data):
    """Plot the degeneracy between eigenstates."""
    if not degeneracy_data:
        return
    
    print("Plotting eigenstate degeneracy...")
    
    # Create a figure for all pairs
    plt.figure(figsize=(12, 8))
    
    # Plot the difference for each pair
    for pair, data in degeneracy_data.items():
        plt.plot(data["theta_values"], data["differences"], label=f'Eigenstates {pair}')
    
    # Add labels and title
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Normalized Absolute Difference (0-1 scale)')
    plt.title('Eigenstate Degeneracy Analysis - Normalized Differences (Optimal Configuration)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate_degeneracy_normalized.png", dpi=300)
    plt.close()
    
    # Create a special figure focusing on the 1-2 pair
    if "1-2" in degeneracy_data:
        plt.figure(figsize=(10, 6))
        
        data = degeneracy_data["1-2"]
        plt.plot(data["theta_values"], data["differences"], 'g-', linewidth=2)
        
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Normalized Absolute Difference (0-1 scale)')
        plt.title('Degeneracy Between Eigenstates 1 and 2 (Should be Degenerate)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Mean Difference: {data['mean']:.6f}\n"
            f"Min Difference: {data['min']:.6f}\n"
            f"Max Difference: {data['max']:.6f}\n"
            f"Std Deviation: {data['std']:.6f}"
        )
        plt.figtext(0.15, 0.15, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate1_2_degeneracy.png", dpi=300)
        plt.close()
    
    # Create a bar chart of mean differences for all pairs
    plt.figure(figsize=(10, 6))
    
    pairs = list(degeneracy_data.keys())
    means = [degeneracy_data[pair]["mean"] for pair in pairs]
    
    # Use a different color for the 1-2 pair
    colors = ['green' if pair == '1-2' else 'blue' for pair in pairs]
    
    plt.bar(pairs, means, color=colors, alpha=0.7)
    plt.xlabel('Eigenstate Pairs')
    plt.ylabel('Mean Absolute Difference')
    plt.title('Mean Eigenstate Differences (Lower = More Degenerate)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations for each bar
    for i, v in enumerate(means):
        plt.text(i, v + 0.0001, f'{v:.6f}', ha='center', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/eigenstate_degeneracy_means.png", dpi=300)
    plt.close()

def copy_logs_to_output_dir():
    """Copy relevant log files to the output directory."""
    print("Copying log files to output directory...")
    
    # Create logs directory in the output directory
    logs_dir = f"{OUTPUT_DIR}/logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Find all relevant log files
    source_logs_dir = "logs"
    if os.path.exists(source_logs_dir):
        # Get the exact parameter string for current run
        current_param_str = f'x{OPTIMAL_PARAMS["x_shift"]}_y{OPTIMAL_PARAMS["y_shift"]}_d{OPTIMAL_PARAMS["d_param"]}_w{OPTIMAL_PARAMS["omega"]}_avx{OPTIMAL_PARAMS["a_vx"]}_ava{OPTIMAL_PARAMS["a_va"]}'
        simple_param_str = f'x{OPTIMAL_PARAMS["x_shift"]}_y{OPTIMAL_PARAMS["y_shift"]}'
        
        # Copy all relevant log files
        copied_files = 0
        
        # First, copy the exact current run logs (highest priority)
        current_run_files = []
        for filename in os.listdir(source_logs_dir):
            if filename.endswith(".txt") and current_param_str in filename:
                source_path = os.path.join(source_logs_dir, filename)
                dest_path = os.path.join(logs_dir, filename)
                shutil.copy2(source_path, dest_path)
                copied_files += 1
                current_run_files.append(filename)
                print(f"Copied current run log file: {filename}")
        
        # Then copy other logs with the same x,y coordinates
        for filename in os.listdir(source_logs_dir):
            if filename.endswith(".txt") and simple_param_str in filename and filename not in current_run_files:
                source_path = os.path.join(source_logs_dir, filename)
                dest_path = os.path.join(logs_dir, filename)
                shutil.copy2(source_path, dest_path)
                copied_files += 1
                print(f"Copied related log file: {filename}")
        
        # Finally, copy a limited number of other berry phase logs for reference
        other_files = []
        for filename in os.listdir(source_logs_dir):
            if filename.endswith(".txt") and (
                ("berry" in filename.lower() or "improved_berry" in filename.lower()) and 
                filename not in current_run_files and 
                simple_param_str not in filename
            ):
                other_files.append(filename)
        
        # Sort by modification time (newest first) and take the 10 most recent
        other_files.sort(key=lambda f: os.path.getmtime(os.path.join(source_logs_dir, f)), reverse=True)
        for filename in other_files[:10]:  # Only copy the 10 most recent files
            source_path = os.path.join(source_logs_dir, filename)
            dest_path = os.path.join(logs_dir, filename)
            shutil.copy2(source_path, dest_path)
            copied_files += 1
            print(f"Copied reference log file: {filename}")
        
        print(f"Copied {copied_files} log files to {logs_dir}")
    else:
        print(f"Source logs directory not found: {source_logs_dir}")

def main():
    """Main function to run the optimal visualization."""
    start_time = time.time()
    
    print(f"=== Running optimal visualization with parameters ===")
    for param, value in OPTIMAL_PARAMS.items():
        print(f"{param}: {value}")
    print()
    
    # Run Berry phase calculation
    success = run_berry_phase_calculation()
    if not success:
        print("Failed to run Berry phase calculation.")
        return
    
    # Parse results
    results = parse_results_file()
    if not results:
        print("Failed to parse results file.")
        return
    
    # Extract eigenstate vs theta data
    eigenstate_data = extract_eigenstate_theta_data()
    
    # Analyze eigenstate degeneracy
    degeneracy_data = None
    if eigenstate_data:
        degeneracy_data = analyze_eigenstate_degeneracy(eigenstate_data)
    
    # Generate plots
    plot_berry_phases(results)
    plot_parity_flips(results)
    plot_potentials()
    if eigenstate_data:
        plot_eigenstate_theta(eigenstate_data)
    if degeneracy_data:
        plot_eigenstate_degeneracy(degeneracy_data)
    create_comprehensive_infographic(results, eigenstate_data)
    
    # Create summary file
    create_summary_file(results, degeneracy_data)
    
    # Copy log files to output directory
    copy_logs_to_output_dir()
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimal visualization completed in {elapsed_time:.2f} seconds.")
    print(f"All outputs saved to {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main()
