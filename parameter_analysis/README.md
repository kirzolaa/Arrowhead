# Parameter Analysis for Arrowhead Berry Phase Calculation

This directory contains a systematic analysis of parameter variations for the Arrowhead Berry Phase calculation, with a focus on minimizing parity flips in eigenstate 3.

## Directory Structure

- **logs/**: Contains log files from simulation runs
- **matrix_generation/**: Contains logs from the matrix generation process
- **plots/**: Contains visualization plots for each parameter configuration
- **results/**: Contains the Berry phase calculation results for each configuration

## Optimal Parameters

### Best Configuration (0 parity flips in eigenstate 3)

- x_shift: 22.5
- y_shift: 547.7222222222222
- d_param: 0.005
- omega: 0.025
- a_vx: 0.018
- a_va: 0.42

### Alternative Zero-Flip Configuration

- x_shift: 22.5
- y_shift: 542.1666666666666
- d_param: 0.005
- omega: 0.025
- a_vx: 0.018
- a_va: 0.42

### Previous Good Configuration (1 parity flip in eigenstate 3)

- x_shift: 22.5
- y_shift: 525.5
- d_param: 0.005
- omega: 0.025
- a_vx: 0.018
- a_va: 0.42

### Initial Configuration (2 parity flips in eigenstate 3)

- x_shift: 25.0
- y_shift: 550.5
- d_param: 0.005
- omega: 0.025
- a_vx: 0.018
- a_va: 0.42

## Analysis Approach

The parameter analysis explores variations of x_shift and y_shift around the optimal values to find configurations that further minimize parity flips in eigenstate 3. The script systematically tests multiple combinations and ranks them based on the number of parity flips.

## Results

The analysis results are summarized in the `analysis_summary.txt` file, which ranks all tested configurations based on their performance in minimizing parity flips in eigenstate 3.

## Scripts

Two Python scripts were created for this analysis:

1. **run_optimal_simulation.py**: Runs a single simulation with the known optimal parameters and saves all outputs to this directory.

2. **parameter_analysis.py**: Systematically explores variations of x_shift and y_shift around the optimal values to find the best configuration.

## Interpretation

The parity flips in eigenstate 3 are particularly important as they represent unwanted transitions in the quantum system. By minimizing these flips, we can achieve a more stable and predictable quantum behavior, which is crucial for applications in quantum computing and quantum information processing.
