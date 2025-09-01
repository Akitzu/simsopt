import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, vjp, jvp

import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

__all__ = ['SurfaceXYZ', 'create_cylinder_surface']


class SurfaceXYZ(sopp.Surface, Surface):
    """
    A parametric surface class that uses JAX functions to define x, y, z coordinates
    as functions of phi and theta parameters.
    
    This class allows you to define a surface using three JAX functions:
    - x_func(dofs, phi, theta) -> x coordinates  
    - y_func(dofs, phi, theta) -> y coordinates
    - z_func(dofs, phi, theta) -> z coordinates
    
    The functions should be JAX-compatible pure functions that take:
    - dofs: array of degrees of freedom (parameters)
    - phi: toroidal angle parameter (0 to 1) 
    - theta: poloidal angle parameter (0 to 1)
    
    And return the corresponding coordinate values.
    
    Example for a cylinder:
        def x_func(dofs, phi, theta):
            x0, y0, z0, theta_rot, phi_rot, radius, height = dofs
            # Local cylinder coordinates
            phi_angle = 2 * jnp.pi * phi
            x_local = radius * jnp.cos(phi_angle)
            y_local = radius * jnp.sin(phi_angle) 
            z_local = height * (theta - 0.5)
            # Apply rotation and translation
            return rotated_and_translated_coordinates...
    """
    
    def __init__(self, x_func, y_func, z_func, initial_dofs, 
                 dof_names=None, quadpoints_phi=None, quadpoints_theta=None, 
                 **kwargs):
        """
        Initialize the parametric surface.
        
        Args:
            x_func: JAX function for x coordinates: x_func(dofs, phi, theta)
            y_func: JAX function for y coordinates: y_func(dofs, phi, theta)  
            z_func: JAX function for z coordinates: z_func(dofs, phi, theta)
            initial_dofs: Initial values for the degrees of freedom
            dof_names: Optional list of names for the DOFs
            quadpoints_phi: Phi quadrature points (if None, uses default)
            quadpoints_theta: Theta quadrature points (if None, uses default)
            nfp: Number of field periods
            **kwargs: Additional arguments for the Surface base class
        """
        
        # Set up quadrature points
        if quadpoints_theta is None:
            quadpoints_theta = np.linspace(0, 1, 100)
        if quadpoints_phi is None:
            quadpoints_phi = np.linspace(0, 1, 100)
        
        # Convert to lists if numpy arrays
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
        if isinstance(quadpoints_theta, np.ndarray):
            quadpoints_theta = list(quadpoints_theta)
            
        # Initialize C++ Surface base class
        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        
        # Store the coordinate functions
        self.x_func = x_func
        self.y_func = y_func
        self.z_func = z_func
        self.nfp = 1
        
        # Set up dof names
        if dof_names is None:
            dof_names = [f'dof_{i}' for i in range(len(initial_dofs))]
        
        # Initialize Surface base class with external dof setter
        if "external_dof_setter" not in kwargs:
            kwargs["external_dof_setter"] = sopp.Surface.set_dofs_impl
            
        Surface.__init__(self, x0=np.array(initial_dofs), names=dof_names, **kwargs)
        
        # Create JAX-compiled functions for efficient evaluation
        self._setup_jax_functions()
    
    def _setup_jax_functions(self):
        """Set up JIT-compiled JAX functions for surface evaluation and derivatives."""
        
        # Convert quadrature points to JAX arrays
        phi_points = jnp.array(self.quadpoints_phi)
        theta_points = jnp.array(self.quadpoints_theta)
        
        # Create meshgrid for evaluation
        phi_grid, theta_grid = jnp.meshgrid(phi_points, theta_points, indexing='ij')
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()
        
        # Function to evaluate all three coordinates
        def gamma_func(dofs):
            x_vals = self.x_func(dofs, phi_flat, theta_flat)
            y_vals = self.y_func(dofs, phi_flat, theta_flat)
            z_vals = self.z_func(dofs, phi_flat, theta_flat)
            
            # Reshape to (nphi, ntheta, 3)
            nphi = len(self.quadpoints_phi)
            ntheta = len(self.quadpoints_theta)
            x_grid = x_vals.reshape(nphi, ntheta)
            y_grid = y_vals.reshape(nphi, ntheta)
            z_grid = z_vals.reshape(nphi, ntheta)
            
            return jnp.stack([x_grid, y_grid, z_grid], axis=2)
        
        # Function for specific phi, theta values (for gamma_lin)
        def gamma_lin_func(dofs, phi_vals, theta_vals):
            x_vals = self.x_func(dofs, phi_vals, theta_vals)
            y_vals = self.y_func(dofs, phi_vals, theta_vals)
            z_vals = self.z_func(dofs, phi_vals, theta_vals)
            return jnp.stack([x_vals, y_vals, z_vals], axis=1)
        
        # Functions for derivatives with respect to phi and theta
        def gammadash1_func(dofs):  # derivative w.r.t. phi
            ones_phi = jnp.ones_like(phi_flat)
            zeros_theta = jnp.zeros_like(theta_flat)
            
            dx_dphi = jvp(lambda p: self.x_func(dofs, p, theta_flat), (phi_flat,), (ones_phi,))[1]
            dy_dphi = jvp(lambda p: self.y_func(dofs, p, theta_flat), (phi_flat,), (ones_phi,))[1]
            dz_dphi = jvp(lambda p: self.z_func(dofs, p, theta_flat), (phi_flat,), (ones_phi,))[1]
            
            nphi = len(self.quadpoints_phi)
            ntheta = len(self.quadpoints_theta)
            dx_grid = dx_dphi.reshape(nphi, ntheta)
            dy_grid = dy_dphi.reshape(nphi, ntheta)
            dz_grid = dz_dphi.reshape(nphi, ntheta)
            
            return jnp.stack([dx_grid, dy_grid, dz_grid], axis=2)
        
        def gammadash2_func(dofs):  # derivative w.r.t. theta
            zeros_phi = jnp.zeros_like(phi_flat)
            ones_theta = jnp.ones_like(theta_flat)
            
            dx_dtheta = jvp(lambda t: self.x_func(dofs, phi_flat, t), (theta_flat,), (ones_theta,))[1]
            dy_dtheta = jvp(lambda t: self.y_func(dofs, phi_flat, t), (theta_flat,), (ones_theta,))[1]
            dz_dtheta = jvp(lambda t: self.z_func(dofs, phi_flat, t), (theta_flat,), (ones_theta,))[1]
            
            nphi = len(self.quadpoints_phi)
            ntheta = len(self.quadpoints_theta)
            dx_grid = dx_dtheta.reshape(nphi, ntheta)
            dy_grid = dy_dtheta.reshape(nphi, ntheta)
            dz_grid = dz_dtheta.reshape(nphi, ntheta)
            
            return jnp.stack([dx_grid, dy_grid, dz_grid], axis=2)
        
        # JIT compile the functions
        self.gamma_jax = jit(gamma_func)
        self.gamma_lin_jax = jit(gamma_lin_func)
        self.gammadash1_jax = jit(gammadash1_func)
        self.gammadash2_jax = jit(gammadash2_func)
        
        # Functions for derivatives with respect to DOFs
        self.dgamma_by_dcoeff_jax = jit(jacfwd(gamma_func))
        self.dgammadash1_by_dcoeff_jax = jit(jacfwd(gammadash1_func))
        self.dgammadash2_by_dcoeff_jax = jit(jacfwd(gammadash2_func))
        
        # VJP functions for efficient gradient computation
        self.dgamma_by_dcoeff_vjp_jax = jit(lambda dofs, v: vjp(gamma_func, dofs)[1](v)[0])
        self.dgammadash1_by_dcoeff_vjp_jax = jit(lambda dofs, v: vjp(gammadash1_func, dofs)[1](v)[0])
        self.dgammadash2_by_dcoeff_vjp_jax = jit(lambda dofs, v: vjp(gammadash2_func, dofs)[1](v)[0])
        
    def get_dofs(self):
        """Get the current degrees of freedom."""
        return self.x
    
    def set_dofs_impl(self, dofs):
        """Set the degrees of freedom."""
        self.x = np.array(dofs)
    
    def num_dofs(self):
        """Return the number of degrees of freedom."""
        return len(self.x)
    
    def gamma_impl(self, data, quadpoints_phi, quadpoints_theta):
        """Compute surface coordinates."""
        data[:, :, :] = self.gamma_jax(self.full_x)
    
    def gamma_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute surface coordinates for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        result = self.gamma_lin_jax(self.full_x, phi_vals, theta_vals)
        data[:, :] = result
    
    def gammadash1_impl(self, data):
        """Compute derivatives with respect to phi."""
        data[:, :, :] = self.gammadash1_jax(self.full_x)
    
    def gammadash2_impl(self, data):
        """Compute derivatives with respect to theta."""
        data[:, :, :] = self.gammadash2_jax(self.full_x)
    
    def gammadash1_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute derivatives with respect to phi for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        # Create vectors for JVP
        ones_phi = jnp.ones_like(phi_vals)
        
        # Compute derivatives using JVP
        dx_dphi = jvp(lambda p: self.x_func(self.full_x, p, theta_vals), (phi_vals,), (ones_phi,))[1]
        dy_dphi = jvp(lambda p: self.y_func(self.full_x, p, theta_vals), (phi_vals,), (ones_phi,))[1]
        dz_dphi = jvp(lambda p: self.z_func(self.full_x, p, theta_vals), (phi_vals,), (ones_phi,))[1]
        
        # Stack the results
        result = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=1)
        data[:, :] = result
    
    def gammadash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute derivatives with respect to theta for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        # Create vectors for JVP
        ones_theta = jnp.ones_like(theta_vals)
        
        # Compute derivatives using JVP
        dx_dtheta = jvp(lambda t: self.x_func(self.full_x, phi_vals, t), (theta_vals,), (ones_theta,))[1]
        dy_dtheta = jvp(lambda t: self.y_func(self.full_x, phi_vals, t), (theta_vals,), (ones_theta,))[1]
        dz_dtheta = jvp(lambda t: self.z_func(self.full_x, phi_vals, t), (theta_vals,), (ones_theta,))[1]
        
        # Stack the results
        result = jnp.stack([dx_dtheta, dy_dtheta, dz_dtheta], axis=1)
        data[:, :] = result
    
    def gammadash1dash1_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute second derivatives with respect to phi (d²/dphi²) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        # Create vectors for second-order JVP
        ones_phi = jnp.ones_like(phi_vals)
        
        # Compute second derivatives using nested JVP (d²/dphi²)
        def d_dphi_x(p): 
            return jvp(lambda pp: self.x_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        def d_dphi_y(p): 
            return jvp(lambda pp: self.y_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        def d_dphi_z(p): 
            return jvp(lambda pp: self.z_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        
        d2x_dphi2 = jvp(d_dphi_x, (phi_vals,), (ones_phi,))[1]
        d2y_dphi2 = jvp(d_dphi_y, (phi_vals,), (ones_phi,))[1]
        d2z_dphi2 = jvp(d_dphi_z, (phi_vals,), (ones_phi,))[1]
        
        # Stack the results
        result = jnp.stack([d2x_dphi2, d2y_dphi2, d2z_dphi2], axis=1)
        data[:, :] = result
    
    def gammadash2dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute second derivatives with respect to theta (d²/dtheta²) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        # Create vectors for second-order JVP
        ones_theta = jnp.ones_like(theta_vals)
        
        # Compute second derivatives using nested JVP (d²/dtheta²)
        def d_dtheta_x(t): 
            return jvp(lambda tt: self.x_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_y(t): 
            return jvp(lambda tt: self.y_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_z(t): 
            return jvp(lambda tt: self.z_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        
        d2x_dtheta2 = jvp(d_dtheta_x, (theta_vals,), (ones_theta,))[1]
        d2y_dtheta2 = jvp(d_dtheta_y, (theta_vals,), (ones_theta,))[1]
        d2z_dtheta2 = jvp(d_dtheta_z, (theta_vals,), (ones_theta,))[1]
        
        # Stack the results
        result = jnp.stack([d2x_dtheta2, d2y_dtheta2, d2z_dtheta2], axis=1)
        data[:, :] = result
    
    def gammadash1dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute mixed second derivatives (d²/dphi∂theta) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        # Create vectors for mixed second-order JVP
        ones_phi = jnp.ones_like(phi_vals)
        ones_theta = jnp.ones_like(theta_vals)
        
        # Compute mixed derivatives using nested JVP (d²/dphi∂theta)
        def d_dtheta_x(t): 
            return jvp(lambda tt: self.x_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_y(t): 
            return jvp(lambda tt: self.y_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_z(t): 
            return jvp(lambda tt: self.z_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        
        # Now take derivative with respect to phi
        d2x_dphidtheta = jvp(lambda p: d_dtheta_x(theta_vals), (phi_vals,), (ones_phi,))[1]
        d2y_dphidtheta = jvp(lambda p: d_dtheta_y(theta_vals), (phi_vals,), (ones_phi,))[1]
        d2z_dphidtheta = jvp(lambda p: d_dtheta_z(theta_vals), (phi_vals,), (ones_phi,))[1]
        
        # Stack the results
        result = jnp.stack([d2x_dphidtheta, d2y_dphidtheta, d2z_dphidtheta], axis=1)
        data[:, :] = result
    
    def gammadash1dash1dash1_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute third derivatives with respect to phi (d³/dphi³) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        ones_phi = jnp.ones_like(phi_vals)
        
        # Triple nested JVP for third derivative
        def d_dphi_x(p): 
            return jvp(lambda pp: self.x_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        def d_dphi_y(p): 
            return jvp(lambda pp: self.y_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        def d_dphi_z(p): 
            return jvp(lambda pp: self.z_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        
        def d2_dphi2_x(p):
            return jvp(d_dphi_x, (p,), (ones_phi,))[1]
        def d2_dphi2_y(p):
            return jvp(d_dphi_y, (p,), (ones_phi,))[1]
        def d2_dphi2_z(p):
            return jvp(d_dphi_z, (p,), (ones_phi,))[1]
        
        d3x_dphi3 = jvp(d2_dphi2_x, (phi_vals,), (ones_phi,))[1]
        d3y_dphi3 = jvp(d2_dphi2_y, (phi_vals,), (ones_phi,))[1]
        d3z_dphi3 = jvp(d2_dphi2_z, (phi_vals,), (ones_phi,))[1]
        
        result = jnp.stack([d3x_dphi3, d3y_dphi3, d3z_dphi3], axis=1)
        data[:, :] = result
    
    def gammadash2dash2dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute third derivatives with respect to theta (d³/dtheta³) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        ones_theta = jnp.ones_like(theta_vals)
        
        # Triple nested JVP for third derivative
        def d_dtheta_x(t): 
            return jvp(lambda tt: self.x_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_y(t): 
            return jvp(lambda tt: self.y_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_z(t): 
            return jvp(lambda tt: self.z_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        
        def d2_dtheta2_x(t):
            return jvp(d_dtheta_x, (t,), (ones_theta,))[1]
        def d2_dtheta2_y(t):
            return jvp(d_dtheta_y, (t,), (ones_theta,))[1]
        def d2_dtheta2_z(t):
            return jvp(d_dtheta_z, (t,), (ones_theta,))[1]
        
        d3x_dtheta3 = jvp(d2_dtheta2_x, (theta_vals,), (ones_theta,))[1]
        d3y_dtheta3 = jvp(d2_dtheta2_y, (theta_vals,), (ones_theta,))[1]
        d3z_dtheta3 = jvp(d2_dtheta2_z, (theta_vals,), (ones_theta,))[1]
        
        result = jnp.stack([d3x_dtheta3, d3y_dtheta3, d3z_dtheta3], axis=1)
        data[:, :] = result
    
    def gammadash1dash1dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute mixed third derivatives (d³/dphi²∂theta) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        ones_phi = jnp.ones_like(phi_vals)
        ones_theta = jnp.ones_like(theta_vals)
        
        # Mixed third derivative: d²/dphi² then d/dtheta
        def d_dphi_x(p): 
            return jvp(lambda pp: self.x_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        def d_dphi_y(p): 
            return jvp(lambda pp: self.y_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        def d_dphi_z(p): 
            return jvp(lambda pp: self.z_func(self.full_x, pp, theta_vals), (p,), (ones_phi,))[1]
        
        def d2_dphi2_x(p):
            return jvp(d_dphi_x, (p,), (ones_phi,))[1]
        def d2_dphi2_y(p):
            return jvp(d_dphi_y, (p,), (ones_phi,))[1]
        def d2_dphi2_z(p):
            return jvp(d_dphi_z, (p,), (ones_phi,))[1]
        
        # Now take derivative with respect to theta
        d3x_dphi2dtheta = jvp(lambda t: d2_dphi2_x(phi_vals), (theta_vals,), (ones_theta,))[1]
        d3y_dphi2dtheta = jvp(lambda t: d2_dphi2_y(phi_vals), (theta_vals,), (ones_theta,))[1]
        d3z_dphi2dtheta = jvp(lambda t: d2_dphi2_z(phi_vals), (theta_vals,), (ones_theta,))[1]
        
        result = jnp.stack([d3x_dphi2dtheta, d3y_dphi2dtheta, d3z_dphi2dtheta], axis=1)
        data[:, :] = result
    
    def gammadash1dash2dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        """Compute mixed third derivatives (d³/dphi∂theta²) for arbitrary quadrature points."""
        phi_vals = jnp.array(quadpoints_phi, dtype=jnp.float64)
        theta_vals = jnp.array(quadpoints_theta, dtype=jnp.float64)
        
        ones_phi = jnp.ones_like(phi_vals)
        ones_theta = jnp.ones_like(theta_vals)
        
        # Mixed third derivative: d²/dtheta² then d/dphi
        def d_dtheta_x(t): 
            return jvp(lambda tt: self.x_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_y(t): 
            return jvp(lambda tt: self.y_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        def d_dtheta_z(t): 
            return jvp(lambda tt: self.z_func(self.full_x, phi_vals, tt), (t,), (ones_theta,))[1]
        
        def d2_dtheta2_x(t):
            return jvp(d_dtheta_x, (t,), (ones_theta,))[1]
        def d2_dtheta2_y(t):
            return jvp(d_dtheta_y, (t,), (ones_theta,))[1]
        def d2_dtheta2_z(t):
            return jvp(d_dtheta_z, (t,), (ones_theta,))[1]
        
        # Now take derivative with respect to phi
        d3x_dphidtheta2 = jvp(lambda p: d2_dtheta2_x(theta_vals), (phi_vals,), (ones_phi,))[1]
        d3y_dphidtheta2 = jvp(lambda p: d2_dtheta2_y(theta_vals), (phi_vals,), (ones_phi,))[1]
        d3z_dphidtheta2 = jvp(lambda p: d2_dtheta2_z(theta_vals), (phi_vals,), (ones_phi,))[1]
        
        result = jnp.stack([d3x_dphidtheta2, d3y_dphidtheta2, d3z_dphidtheta2], axis=1)
        data[:, :] = result
    
    def dgamma_by_dcoeff_impl(self, data):
        """Compute derivatives of coordinates with respect to DOFs."""
        data[:, :, :, :] = self.dgamma_by_dcoeff_jax(self.full_x)
    
    def dgammadash1_by_dcoeff_impl(self, data):
        """Compute derivatives of phi-derivatives with respect to DOFs."""
        data[:, :, :, :] = self.dgammadash1_by_dcoeff_jax(self.full_x)
    
    def dgammadash2_by_dcoeff_impl(self, data):
        """Compute derivatives of theta-derivatives with respect to DOFs."""
        data[:, :, :, :] = self.dgammadash2_by_dcoeff_jax(self.full_x)
    
    def dgamma_by_dcoeff_vjp_impl(self, v):
        """Vector-Jacobian product for gamma derivatives."""
        return self.dgamma_by_dcoeff_vjp_jax(self.full_x, v)
    
    def dgammadash1_by_dcoeff_vjp_impl(self, v):
        """Vector-Jacobian product for gammadash1 derivatives."""
        return self.dgammadash1_by_dcoeff_vjp_jax(self.full_x, v)
    
    def dgammadash2_by_dcoeff_vjp_impl(self, v):
        """Vector-Jacobian product for gammadash2 derivatives."""
        return self.dgammadash2_by_dcoeff_vjp_jax(self.full_x, v)
    
    def to_RZFourier(self, mpol=6, ntor=6):
        """
        Convert to SurfaceRZFourier representation by fitting Fourier modes.
        
        Args:
            mpol: Maximum poloidal mode number
            ntor: Maximum toroidal mode number
            
        Returns:
            SurfaceRZFourier object approximating this surface
        """
        # Create a new SurfaceRZFourier
        surf = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=self.nfp,
                               stellsym=True,  # Assume stellarator symmetry
                               quadpoints_phi=self.quadpoints_phi,
                               quadpoints_theta=self.quadpoints_theta)
        
        # Evaluate this surface on the grid
        gamma_data = self.gamma()
        
        # Convert Cartesian to cylindrical coordinates
        x = gamma_data[:, :, 0]
        y = gamma_data[:, :, 1]
        z = gamma_data[:, :, 2]
        
        r = np.sqrt(x**2 + y**2)
        
        # Simple Fourier fitting using FFT-like approach
        nphi, ntheta = r.shape
        phi_grid = np.array(self.quadpoints_phi)
        theta_grid = np.array(self.quadpoints_theta)
        
        # Fit Fourier coefficients
        for m in range(mpol + 1):
            for n in range(-ntor, ntor + 1):
                if m == 0 and n < 0:
                    continue
                
                rc_val = 0.0
                zs_val = 0.0
                
                for i in range(nphi):
                    for j in range(ntheta):
                        phi_val = 2 * np.pi * phi_grid[i]
                        theta_val = 2 * np.pi * theta_grid[j]
                        arg = m * theta_val - n * self.nfp * phi_val
                        
                        rc_val += r[i, j] * np.cos(arg)
                        if not (m == 0 and n == 0):
                            zs_val += z[i, j] * np.sin(arg)
                
                rc_val *= 2.0 / (nphi * ntheta)
                zs_val *= 2.0 / (nphi * ntheta)
                
                if m == 0 and n == 0:
                    rc_val *= 0.5
                
                surf.rc[m, n + ntor] = rc_val
                if not (m == 0 and n == 0):
                    surf.zs[m, n + ntor] = zs_val
        
        surf.local_full_x = surf.get_dofs()
        return surf


def create_cylinder_surface(center=[0.0, 0.0, 0.0], orientation=[0.0, 0.0], 
                           radius=1.0, height=2.0, **kwargs):
    """
    Create a cylinder surface using JAX functions.
    
    Args:
        center: [x0, y0, z0] center coordinates
        orientation: [theta, phi] Euler angles for rotation
        radius: Cylinder radius
        height: Cylinder height
        **kwargs: Additional arguments for SurfaceXYZ
        
    Returns:
        SurfaceXYZ representing a cylinder
    """
    
    def x_func(dofs, phi, theta):
        x0, y0, z0, theta_rot, phi_rot, r, h = dofs
        
        # Local cylinder coordinates (axis along z)
        phi_angle = 2 * jnp.pi * phi
        x_local = r * jnp.cos(phi_angle)
        y_local = r * jnp.sin(phi_angle)
        z_local = h * (theta - 0.5)
        
        # Rotation matrices
        # Rotation by theta_rot around z-axis
        cos_t, sin_t = jnp.cos(theta_rot), jnp.sin(theta_rot)
        # Rotation by phi_rot around y-axis
        cos_p, sin_p = jnp.cos(phi_rot), jnp.sin(phi_rot)
        
        # Apply rotations: first Rz, then Ry
        x_rot1 = cos_t * x_local - sin_t * y_local
        y_rot1 = sin_t * x_local + cos_t * y_local
        z_rot1 = z_local
        
        x_rot2 = cos_p * x_rot1 + sin_p * z_rot1
        y_rot2 = y_rot1
        z_rot2 = -sin_p * x_rot1 + cos_p * z_rot1
        
        return x_rot2 + x0

    def y_func(dofs, phi, theta):
        x0, y0, z0, theta_rot, phi_rot, r, h = dofs
        
        phi_angle = 2 * jnp.pi * phi
        x_local = r * jnp.cos(phi_angle)
        y_local = r * jnp.sin(phi_angle)
        z_local = h * (theta - 0.5)
        
        cos_t, sin_t = jnp.cos(theta_rot), jnp.sin(theta_rot)
        cos_p, sin_p = jnp.cos(phi_rot), jnp.sin(phi_rot)
        
        x_rot1 = cos_t * x_local - sin_t * y_local
        y_rot1 = sin_t * x_local + cos_t * y_local
        z_rot1 = z_local
        
        x_rot2 = cos_p * x_rot1 + sin_p * z_rot1
        y_rot2 = y_rot1
        z_rot2 = -sin_p * x_rot1 + cos_p * z_rot1
        
        return y_rot2 + y0

    def z_func(dofs, phi, theta):
        x0, y0, z0, theta_rot, phi_rot, r, h = dofs
        
        phi_angle = 2 * jnp.pi * phi
        x_local = r * jnp.cos(phi_angle)
        y_local = r * jnp.sin(phi_angle)
        z_local = h * (theta - 0.5)
        
        cos_t, sin_t = jnp.cos(theta_rot), jnp.sin(theta_rot)
        cos_p, sin_p = jnp.cos(phi_rot), jnp.sin(phi_rot)
        
        x_rot1 = cos_t * x_local - sin_t * y_local
        y_rot1 = sin_t * x_local + cos_t * y_local
        z_rot1 = z_local
        
        x_rot2 = cos_p * x_rot1 + sin_p * z_rot1
        y_rot2 = y_rot1
        z_rot2 = -sin_p * x_rot1 + cos_p * z_rot1
        
        return z_rot2 + z0
    
    # Set up initial DOFs
    initial_dofs = [center[0], center[1], center[2], 
                    orientation[0], orientation[1], 
                    radius, height]
    
    dof_names = ['x0', 'y0', 'z0', 'theta', 'phi', 'radius', 'height']
    
    return SurfaceXYZ(x_func, y_func, z_func, initial_dofs, 
                      dof_names=dof_names, **kwargs)
