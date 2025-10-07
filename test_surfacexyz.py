#!/usr/bin/env python3

"""
Example script demonstrating the JAX-based parametric surface SurfaceXYZ.
This shows how to create a cylinder surface using JAX functions.
"""

import numpy as np
import jax.numpy as jnp
import sys
import os

# Add the simsopt path
sys.path.insert(0, '/misc/CENV/simsopt/src')

try:
    from simsopt.geo import SurfaceXYZ, create_cylinder_surface
    print("✓ Successfully imported SurfaceXYZ and cylinder helper")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def example_simple_cylinder():
    """Example 1: Create a simple cylinder using the helper function."""
    print("\n=== Example 1: Simple Cylinder ===")
    
    # Create a cylinder using the helper function
    cylinder = create_cylinder_surface(
        center=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0],  # No rotation
        radius=1.0,
        height=2.0,
        nfp=1
    )
    
    print(f"Cylinder DOFs: {cylinder.get_dofs()}")
    print(f"DOF names: {cylinder.names}")
    
    # Evaluate the surface
    gamma = cylinder.gamma()
    print(f"Surface shape: {gamma.shape}")
    
    # Check that all points are at the correct radius
    x, y, z = gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2]
    r_computed = np.sqrt(x**2 + y**2)
    
    print(f"Radius range: {np.min(r_computed):.6f} to {np.max(r_computed):.6f}")
    print(f"Height range: {np.min(z):.6f} to {np.max(z):.6f}")
    
    # Test derivatives
    try:
        gammadash1 = cylinder.gammadash1()  # d/dphi
        gammadash2 = cylinder.gammadash2()  # d/dtheta
        print(f"✓ Derivative computation successful")
        print(f"  Phi derivative shape: {gammadash1.shape}")
        print(f"  Theta derivative shape: {gammadash2.shape}")
    except Exception as e:
        print(f"✗ Derivative computation failed: {e}")
    
    # Test area and volume if available
    try:
        area = cylinder.area()
        volume = cylinder.volume()
        print(f"Surface area: {area:.6f}")
        print(f"Volume: {volume:.6f}")
        
        # Analytical values for comparison
        expected_area = 2 * np.pi * 1.0 * 2.0 + 2 * np.pi * 1.0**2  # lateral + caps
        expected_volume = np.pi * 1.0**2 * 2.0
        print(f"Expected area: {expected_area:.6f}")
        print(f"Expected volume: {expected_volume:.6f}")
    except Exception as e:
        print(f"Note: Area/volume computation not available: {e}")
    
    return cylinder


def example_rotated_cylinder():
    """Example 2: Create a rotated cylinder."""
    print("\n=== Example 2: Rotated Cylinder ===")
    
    # Create cylinder rotated 45 degrees around z-axis
    cylinder = create_cylinder_surface(
        center=[1.0, 2.0, 3.0],
        orientation=[np.pi/4, np.pi/6],  # 45° around z, 30° around y
        radius=0.5,
        height=1.5
    )
    
    gamma = cylinder.gamma()
    print(f"Rotated cylinder center: {np.mean(gamma, axis=(0,1))}")
    print(f"Expected center: [1.0, 2.0, 3.0]")
    
    return cylinder


def example_custom_surface():
    """Example 3: Create a custom parametric surface using SurfaceXYZ directly."""
    print("\n=== Example 3: Custom Torus Surface ===")
    
    def x_func(dofs, phi, theta):
        """X coordinate for a torus."""
        R, r = dofs[0], dofs[1]  # Major radius, minor radius
        phi_angle = 2 * jnp.pi * phi
        theta_angle = 2 * jnp.pi * theta
        return (R + r * jnp.cos(theta_angle)) * jnp.cos(phi_angle)
    
    def y_func(dofs, phi, theta):
        """Y coordinate for a torus.""" 
        R, r = dofs[0], dofs[1]
        phi_angle = 2 * jnp.pi * phi
        theta_angle = 2 * jnp.pi * theta
        return (R + r * jnp.cos(theta_angle)) * jnp.sin(phi_angle)
    
    def z_func(dofs, phi, theta):
        """Z coordinate for a torus."""
        R, r = dofs[0], dofs[1]
        theta_angle = 2 * jnp.pi * theta
        return r * jnp.sin(theta_angle)
    
    # Create torus with major radius 2.0, minor radius 0.5
    initial_dofs = [2.0, 0.5]
    dof_names = ['major_radius', 'minor_radius']
    
    torus = SurfaceXYZ(x_func, y_func, z_func, initial_dofs, 
                       dof_names=dof_names)
    
    print(f"Torus DOFs: {torus.get_dofs()}")
    print(f"DOF names: {torus.names}")
    
    # Evaluate the torus
    gamma = torus.gamma()
    print(f"Torus surface shape: {gamma.shape}")
    
    # Check torus properties
    x, y, z = gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2]
    
    # Distance from z-axis
    r_from_z = np.sqrt(x**2 + y**2)
    print(f"Distance from z-axis range: {np.min(r_from_z):.6f} to {np.max(r_from_z):.6f}")
    print(f"Expected range: {2.0 - 0.5:.1f} to {2.0 + 0.5:.1f}")
    
    print(f"Z-coordinate range: {np.min(z):.6f} to {np.max(z):.6f}")
    print(f"Expected range: {-0.5:.1f} to {0.5:.1f}")
    
    return torus


def example_dof_modification():
    """Example 4: Modify surface by changing DOFs."""
    print("\n=== Example 4: DOF Modification ===")
    
    # Start with a cylinder
    cylinder = create_cylinder_surface(radius=1.0, height=2.0)
    
    print("Original cylinder:")
    gamma1 = cylinder.gamma()
    print(f"  Original radius range: {np.min(np.sqrt(gamma1[:,:,0]**2 + gamma1[:,:,1]**2)):.3f} to {np.max(np.sqrt(gamma1[:,:,0]**2 + gamma1[:,:,1]**2)):.3f}")
    
    # Modify the radius and height
    new_dofs = cylinder.get_dofs().copy()
    new_dofs[5] = 2.0  # radius
    new_dofs[6] = 4.0  # height
    cylinder.x = new_dofs
    
    print("Modified cylinder:")
    gamma2 = cylinder.gamma()
    print(f"  New radius range: {np.min(np.sqrt(gamma2[:,:,0]**2 + gamma2[:,:,1]**2)):.3f} to {np.max(np.sqrt(gamma2[:,:,0]**2 + gamma2[:,:,1]**2)):.3f}")
    print(f"  New height range: {np.min(gamma2[:,:,2]):.3f} to {np.max(gamma2[:,:,2]):.3f}")
    
    return cylinder


def example_conversion_to_rzfourier():
    """Example 5: Convert to SurfaceRZFourier."""
    print("\n=== Example 5: Conversion to RZFourier ===")
    
    # Create cylinder
    cylinder = create_cylinder_surface(radius=1.0, height=2.0)
    
    try:
        # Convert to RZFourier representation
        rz_surface = cylinder.to_RZFourier(mpol=4, ntor=4)
        
        print(f"✓ Conversion successful")
        print(f"Original surface type: {type(cylinder)}")
        print(f"RZFourier surface type: {type(rz_surface)}")
        
        # Compare areas if possible
        try:
            area1 = cylinder.area()
            area2 = rz_surface.area()
            print(f"Original area: {area1:.6f}")
            print(f"RZFourier area: {area2:.6f}")
            print(f"Area difference: {abs(area1-area2):.6f}")
        except:
            print("Area comparison not available")
            
    except Exception as e:
        print(f"✗ Conversion failed: {e}")


def main():
    """Run all examples."""
    print("Testing JAX-based parametric surfaces (SurfaceXYZ)...")
    
    examples = [
        example_simple_cylinder,
        example_rotated_cylinder,
        example_custom_surface,
        example_dof_modification,
        example_conversion_to_rzfourier,
    ]
    
    results = []
    for example in examples:
        try:
            result = example()
            results.append(result)
            print("✓ Example completed successfully")
        except Exception as e:
            print(f"✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)
    
    print(f"\n=== Summary ===")
    print(f"Completed {len([r for r in results if r is not None])}/{len(examples)} examples successfully")
    
    if all(r is not None for r in results):
        print("🎉 All examples completed successfully!")
        return True
    else:
        print("❌ Some examples failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
