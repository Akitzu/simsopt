import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, vjp, jvp

import simsoptpp as sopp
from .surface import Surface

__all__ = ["Cylinder"]

class Cylinder(sopp.Surface, Surface):
    def __init__(self, R=1.0, Z=1.0, quadpoints_phi=None, quadpoints_theta=None):
        self._R = R
        self._Z = Z
        
        # Set up quadrature points
        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=1)
        
        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        
    def gamma_impl(self, _, phi, theta):
        x = self._R * jnp.cos(theta)
        y = self._R * jnp.sin(theta)
        z = self._Z * (phi - 0.5)
        return jnp.array([x, y, z])