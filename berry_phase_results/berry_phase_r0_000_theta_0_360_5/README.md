# Berry Phase Calculation for Perfect Circle

This script generates a perfect circle with 72 points between 0 and 360 degrees, visualizes it, calculates the eigenproblem for a 4x4 arrowhead matrix for each point, and calculates the Berry phase.

## Features

- Generates a perfect circle orthogonal to the x=y=z line using basis vectors [1, -1/2, -1/2] and [0, -1/2, 1/2]
- Visualizes the circle with enhanced visualization
- Calculates the eigenproblem for a 4x4 arrowhead matrix for each point
- Calculates and visualizes the Berry phase
- Generates comprehensive logs and visualizations

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default parameters
python berry_phase_r0_000_theta_0_360_5.py

# Run with custom parameters
python berry_phase_r0_000_theta_0_360_5.py --r0 0 0 0 --d 1.0 --theta-steps 72 --coupling 0.1 --omega 1.0 --matrix-size 4 --output-dir berry_phase_logs
```

## Parameters

- `--r0`: Origin vector (x, y, z), default: [0, 0, 0]
- `--d`: Distance parameter, default: 1.0
- `--theta-steps`: Number of theta values to generate matrices for, default: 72
- `--coupling`: Coupling constant for off-diagonal elements, default: 0.1
- `--omega`: Angular frequency for the energy term h*ω, default: 1.0
- `--matrix-size`: Size of the matrix to generate, default: 4
- `--output-dir`: Directory to save results, default: berry_phase_logs

## Output

The script generates the following outputs in the specified output directory:

- `plots/`: Contains visualizations of the circle, eigenvalues, and Berry phase
- `text/`: Contains detailed logs of the results
- `numpy/`: Contains saved NumPy arrays of eigenvalues, eigenvectors, and Berry phases
- `csv/`: Contains CSV files with data for further analysis

## Berry Phase Calculation

The Berry phase is calculated for each eigenstate by:

1. Computing the overlap between eigenvectors at adjacent points on the circle
2. Calculating the phase difference between these overlaps
3. Summing these phase differences around the entire circle
4. Normalizing the result to be in the range [-π, π]

The results are visualized and saved to the output directory.
