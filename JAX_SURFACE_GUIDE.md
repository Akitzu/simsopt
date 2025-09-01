# JAX-based Parametric Surfaces in SIMSOPT

This document describes the new `SurfaceXYZ` class, which allows you to define surfaces using JAX functions for the x, y, and z coordinates. This approach provides automatic differentiation and is much more flexible than Fourier-based representations.

## Overview

The `SurfaceXYZ` class takes three JAX functions that define the x, y, and z coordinates as functions of:
- `dofs`: Array of degrees of freedom (parameters)
- `phi`: Toroidal parameter (0 to 1)
- `theta`: Poloidal parameter (0 to 1)

## Basic Usage

### Creating a Custom Surface

```python
import jax.numpy as jnp
from simsopt.geo import SurfaceXYZ

def x_func(dofs, phi, theta):
    # Your custom x coordinate function
    return dofs[0] + dofs[1] * jnp.cos(2 * jnp.pi * phi)

def y_func(dofs, phi, theta):
    # Your custom y coordinate function
    return dofs[2] + dofs[1] * jnp.sin(2 * jnp.pi * phi)

def z_func(dofs, phi, theta):
    # Your custom z coordinate function
    return dofs[3] * theta

initial_dofs = [0.0, 1.0, 0.0, 2.0]  # [x_offset, radius, y_offset, height]
dof_names = ['x_offset', 'radius', 'y_offset', 'height']

surface = SurfaceXYZ(x_func, y_func, z_func, initial_dofs, dof_names=dof_names)
```

### Creating a Cylinder (Helper Function)

For common shapes like cylinders, there's a helper function:

```python
from simsopt.geo import create_cylinder_surface

cylinder = create_cylinder_surface(
    center=[0.0, 0.0, 0.0],      # Center position [x0, y0, z0]
    orientation=[0.0, 0.0],       # Euler angles [theta, phi] in radians
    radius=1.0,                   # Cylinder radius
    height=2.0,                   # Cylinder height
    nfp=1                         # Number of field periods
)
```

The cylinder has 7 DOFs: `[x0, y0, z0, theta, phi, radius, height]`
- `x0, y0, z0`: Center position
- `theta`: Rotation around z-axis
- `phi`: Rotation around y-axis (after theta rotation)
- `radius`: Cylinder radius
- `height`: Cylinder height

## Surface Evaluation

Once you have a surface, you can evaluate it and its derivatives:

```python
# Get surface coordinates
gamma = surface.gamma()  # Shape: (nphi, ntheta, 3)

# Get derivatives
dgamma_dphi = surface.gammadash1()    # Derivative w.r.t. phi
dgamma_dtheta = surface.gammadash2()  # Derivative w.r.t. theta

# Get surface normal (automatically computed)
normal = surface.normal()

# Get area and volume (if applicable)
area = surface.area()
volume = surface.volume()
```

## Automatic Differentiation

The JAX-based approach provides automatic derivatives with respect to the DOFs:

```python
# Derivatives of coordinates w.r.t. DOFs
dgamma_ddofs = surface.dgamma_by_dcoeff()  # Shape: (nphi, ntheta, 3, ndofs)

# Derivatives of phi-derivatives w.r.t. DOFs
dgammadash1_ddofs = surface.dgammadash1_by_dcoeff()

# Derivatives of theta-derivatives w.r.t. DOFs
dgammadash2_ddofs = surface.dgammadash2_by_dcoeff()
```

## Modifying Parameters

You can modify the surface by changing the DOFs:

```python
# Get current DOFs
current_dofs = surface.get_dofs()

# Modify them
new_dofs = current_dofs.copy()
new_dofs[5] = 2.0  # Change radius to 2.0
new_dofs[6] = 3.0  # Change height to 3.0

# Set new DOFs
surface.x = new_dofs

# The surface is automatically updated
new_gamma = surface.gamma()
```

## Conversion to RZFourier

You can convert your parametric surface to a Fourier representation:

```python
rz_surface = surface.to_RZFourier(mpol=6, ntor=6)
```

This creates a `SurfaceRZFourier` object that approximates your parametric surface using Fourier modes.

## Example: Custom Torus

Here's how to create a torus surface:

```python
def x_func(dofs, phi, theta):
    R, r = dofs[0], dofs[1]  # Major radius, minor radius
    phi_angle = 2 * jnp.pi * phi
    theta_angle = 2 * jnp.pi * theta
    return (R + r * jnp.cos(theta_angle)) * jnp.cos(phi_angle)

def y_func(dofs, phi, theta):
    R, r = dofs[0], dofs[1]
    phi_angle = 2 * jnp.pi * phi
    theta_angle = 2 * jnp.pi * theta
    return (R + r * jnp.cos(theta_angle)) * jnp.sin(phi_angle)

def z_func(dofs, phi, theta):
    R, r = dofs[0], dofs[1]
    theta_angle = 2 * jnp.pi * theta
    return r * jnp.sin(theta_angle)

initial_dofs = [2.0, 0.5]  # Major radius 2.0, minor radius 0.5
dof_names = ['major_radius', 'minor_radius']

torus = SurfaceXYZ(x_func, y_func, z_func, initial_dofs, dof_names=dof_names)
```

## Advantages

1. **Flexibility**: Define any parametric surface using mathematical functions
2. **Automatic Differentiation**: JAX provides exact derivatives automatically
3. **Performance**: JIT compilation makes evaluation fast
4. **Easy Parameterization**: Direct control over geometric parameters
5. **Integration**: Works with all existing SIMSOPT optimization routines

## Performance Notes

- Functions are automatically JIT-compiled for fast evaluation
- Use JAX functions (`jnp.sin`, `jnp.cos`, etc.) instead of NumPy for compatibility
- The surface is evaluated on a fixed grid defined by `quadpoints_phi` and `quadpoints_theta`

## Advanced Usage

For optimization, you can define objectives that depend on your surface:

```python
from simsopt.geo import SurfaceObjective

def my_objective(surface):
    return surface.area()  # Minimize surface area

# Use in optimization...
```

The automatic differentiation will provide exact gradients for efficient optimization.
