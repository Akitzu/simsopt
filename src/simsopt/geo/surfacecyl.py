import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, vjp, jvp

import simsoptpp as sopp
from .surface import Surface

__all__ = ["Cylinder"]

class Cylinder(sopp.Surface, Surface):
    def __init__(self, quadpoints_phi=None, quadpoints_theta=None, dofs=None):

        self._R = 1.0
        self._H = 1.0

        # Set up quadrature points
        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=1)
        
        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        if dofs is None:
            Surface.__init__(self, x0=self.get_dofs(), names=self._make_names(),
                             external_dof_setter=Cylinder.set_dofs_impl)
        else:
            Surface.__init__(self, dofs=dofs,
                             external_dof_setter=Cylinder.set_dofs_impl)
    
    def gamma_lin(self, data, quadpoints_phi, quadpoints_theta):
        x = self._R * jnp.cos(2*np.pi*quadpoints_phi)
        y = self._R * jnp.sin(2*np.pi*quadpoints_phi)
        z = self._H * (quadpoints_theta - 0.5)
        data[:, :] = jnp.stack((x, y, z), axis=-1)

    def gamma_impl(self, data, quadpoints_phi, quadpoints_theta):
        phi, theta = jnp.meshgrid(quadpoints_phi, quadpoints_theta, indexing='ij')
        x = self._R * jnp.cos(2*np.pi*phi)
        y = self._R * jnp.sin(2*np.pi*phi)
        z = self._H * (theta - 0.5)
        data[:, :, :] = jnp.stack((x, y, z), axis=-1)

    def _make_names(self):
        return ["R", "H"]
    
    def set_dofs_impl(self, dofs):
        self._R = dofs[0]
        self._H = dofs[1]

    def set_dofs(self, dofs):
        self.local_x = dofs

    def get_dofs(self):
        return np.array([self._R, self._H])

    def num_dofs(self):
        return 2
    
    def gammadash1_impl(self, data):
        phi, theta = jnp.meshgrid(self.quadpoints_phi, self.quadpoints_theta, indexing='ij')
        dx_dphi = -self._R * 2 * np.pi * jnp.sin(2 * np.pi * phi)
        dy_dphi = self._R * 2 * np.pi * jnp.cos(2 * np.pi * phi)
        dz_dphi = self._H * jnp.ones_like(phi)
        
        data[:, :, 0] = dx_dphi
        data[:, :, 1] = dy_dphi
        data[:, :, 2] = dz_dphi

    def gammadash2_impl(self, data):
        phi, theta = jnp.meshgrid(self.quadpoints_phi, self.quadpoints_theta, indexing='ij')
        dx_dtheta = jnp.zeros_like(theta)
        dy_dtheta = jnp.zeros_like(theta)
        dz_dtheta = self._H * jnp.ones_like(theta)

        data[:, :, 0] = dx_dtheta
        data[:, :, 1] = dy_dtheta
        data[:, :, 2] = dz_dtheta

    def gammadash1_lin(self, data, quadpoints_phi, quadpoints_theta):
        dx_dphi = -self._R * 2 * np.pi * jnp.sin(2 * np.pi * quadpoints_phi)
        dy_dphi = self._R * 2 * np.pi * jnp.cos(2 * np.pi * quadpoints_phi)
        dz_dphi = jnp.zeros_like(quadpoints_phi)
        
        data[:, 0] = dx_dphi
        data[:, 1] = dy_dphi
        data[:, 2] = dz_dphi

    def gammadash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        dx_dtheta = jnp.zeros_like(quadpoints_theta)
        dy_dtheta = jnp.zeros_like(quadpoints_theta)
        dz_dtheta = self._H * jnp.ones_like(quadpoints_theta)

        data[:, 0] = dx_dtheta
        data[:, 1] = dy_dtheta
        data[:, 2] = dz_dtheta

    def gammadash1dash1_lin(self, data, quadpoints_phi, quadpoints_theta):
        dx_dphi2 = -self._R * (2 * np.pi)**2 * jnp.cos(2 * np.pi * quadpoints_phi)
        dy_dphi2 = -self._R * (2 * np.pi)**2 * jnp.sin(2 * np.pi * quadpoints_phi)
        dz_dphi2 = jnp.zeros_like(quadpoints_phi)

        data[:, 0] = dx_dphi2
        data[:, 1] = dy_dphi2
        data[:, 2] = dz_dphi2

    def gammadash1dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        dx_dphidtheta = jnp.zeros_like(quadpoints_phi)
        dy_dphidtheta = jnp.zeros_like(quadpoints_phi)
        dz_dphidtheta = jnp.zeros_like(quadpoints_phi)

        data[:, 0] = dx_dphidtheta
        data[:, 1] = dy_dphidtheta
        data[:, 2] = dz_dphidtheta

    def gammadash2dash2_lin(self, data, quadpoints_phi, quadpoints_theta):
        dx_dtheta2 = jnp.zeros_like(quadpoints_theta)
        dy_dtheta2 = jnp.zeros_like(quadpoints_theta)
        dz_dtheta2 = jnp.zeros_like(quadpoints_theta)

        data[:, 0] = dx_dtheta2
        data[:, 1] = dy_dtheta2
        data[:, 2] = dz_dtheta2