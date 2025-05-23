# Circle Examples for Orthogonal Vector Visualization

This document explains the circle examples created to visualize orthogonal vectors in different ways.

## Overview

Three different circle examples have been created:

1. `example_circle.py` - Generates points using the orthogonal vector formulas, creating a sphere-like pattern
2. `example_circle_xy.py` - Generates a traditional circle in the XY plane
3. `example_orthogonal_circle.py` - Similar to example_circle.py but with improved visualization

## Running the Examples

To run any of the examples, first set up the Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run the desired example:

```bash
# Run the original circle example
python generalized/example_circle.py

# Run the XY plane circle example
python generalized/example_circle_xy.py

# Run the orthogonal circle example
python generalized/example_orthogonal_circle.py
```

## Example Details

### example_circle.py

This example generates 73 points (0° to 360° in 5° increments) using the orthogonal vector formulas with:
- R_0 = (0, 0, 0)
- d = 0.1
- theta = 0° to 360° in 5° increments

The resulting pattern forms a sphere-like shape because the orthogonal vector formula generates points that move in all three dimensions as theta changes.

### example_circle_xy.py

This example creates a traditional circle in the XY plane with:
- R_0 = (0, 0, 0)
- radius = 0.1
- theta = 0° to 360° in 5° increments

The points are calculated using the standard circle formula:
- x = radius * cos(theta)
- y = radius * sin(theta)
- z = 0

This creates a perfect circle in the XY plane.

### example_orthogonal_circle.py

This example is similar to example_circle.py but uses the visualization module's plot_multiple_vectors function to show the endpoints of the vectors in a more organized way.

## Output Files

All examples save their plots to the `circle_plots` directory:

- Original circle example:
  - `circle_3d.png` - 3D view of the points
  - `circle_xy.png`, `circle_xz.png`, `circle_yz.png` - 2D projections
  - `circle_r0.png` - R_0 plane projection

- XY plane circle example:
  - `3d_xy_circle.png` - 3D view of the circle
  - `xy_circle.png` - 2D view of the circle in the XY plane

- Orthogonal circle example:
  - `orthogonal_3d.png` - 3D view of the points
  - `orthogonal_xy.png`, `orthogonal_xz.png`, `orthogonal_yz.png` - 2D projections
  - `orthogonal_r0.png` - R_0 plane projection

## Key Observations

1. The orthogonal vector formula (`vector_utils.py`) generates points that form a sphere-like pattern when theta is varied from 0° to 360°.

2. All points generated by the orthogonal vector formula lie on a sphere centered at the origin, with the radius determined by the distance parameter `d`.

3. The traditional circle example demonstrates how to create a perfect circle in the XY plane for comparison.

4. The endpoints-only plotting option provides a clearer visualization of the point patterns without the clutter of arrows.
