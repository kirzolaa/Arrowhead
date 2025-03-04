# Generalized Orthogonal Vectors Generator and Visualizer

This is a generalized implementation of the orthogonal vectors generator and visualizer. It provides a modular and configurable approach to creating and visualizing three orthogonal vectors from a given origin point.

## Features

- Modular architecture with separate components for vector calculations, visualization, and configuration
- Command-line interface with various options for customization
- Configuration management with JSON file support
- Ability to save plots to files
- Example scripts demonstrating different use cases
- Enhanced R_0 plane projections with improved axis handling and visualization
- Combined 3D and R_0 plane projection views

## Usage

### Basic Usage

```bash
python main.py
```

This will generate orthogonal vectors with default parameters and display the plots.

### Command-line Options

```bash
python main.py --origin 1 1 1 --distance 2 --angle 1.047
```

This will generate orthogonal vectors with the specified origin (1,1,1), distance (2), and angle (π/3 radians).

### Saving Plots

```bash
python main.py --save-plots --output-dir my_plots
```

This will save the plots to the `my_plots` directory instead of displaying them.

### Configuration Files

```bash
# Save configuration to a file
python main.py --save-config my_config.json

# Load configuration from a file
python main.py --config my_config.json
```

## Examples

See `example.py` for examples of how to use the package programmatically.

## Package Structure

- `__init__.py`: Package initialization and exports
- `vector_utils.py`: Vector calculation utilities
- `visualization.py`: Visualization functions
- `config.py`: Configuration management
- `main.py`: Command-line interface
- `example.py`: Example usage

## API Reference

### Vector Utilities

- `create_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0)`: Create three orthogonal vectors from a given origin
- `check_orthogonality(R_0, R_1, R_2, R_3)`: Check if the vectors are orthogonal

### Visualization

- `plot_vectors_3d(R_0, R_1, R_2, R_3, ...)`: Plot the vectors in 3D
- `plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='xy', ...)`: Plot a 2D projection of the vectors
- `plot_all_projections(R_0, R_1, R_2, R_3, ...)`: Plot all projections of the vectors
- `plot_r0_projections(R_0, R_1, R_2, R_3, ...)`: Plot the R_0 plane projections with enhanced visualization
- `plot_combined_with_r0(R_0, R_1, R_2, R_3, ...)`: Plot both 3D vectors and R_0 plane projections in a single figure

### Configuration

- `VectorConfig`: Configuration class for vector generation and visualization
- `default_config`: Default configuration instance

## R_0 Plane Projection Scripts

The package includes specialized scripts for generating R_0 plane projections, which provide a clear view of the orthogonality of the vectors in the plane perpendicular to the origin direction.

### generate_r0_projections.py

```bash
python docs/generate_r0_projections.py
```

Generates R_0 plane projections for various configurations with improved axis handling, symmetric axis limits, and better legend placement.

### generate_combined_views.py

```bash
python docs/generate_combined_views.py
```

Generates combined 3D and R_0 plane projection figures, showing both perspectives side by side.

### generate_specific_r0_projections.py

```bash
python docs/generate_specific_r0_projections.py
```

Generates R_0 plane projections specifically for the three combined effect configurations (origins at (0,0,0), (1,1,1), and (0,0,2)).
