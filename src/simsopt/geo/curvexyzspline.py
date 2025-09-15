from .curve import JaxCurve
from numpy.typing import ArrayLike
from typing import Union
import jax.numpy as jnp

try:
    from interpax import Interpolator1D
except ImportError:
    raise ImportError(
        "The interpax package is required to use CurveXYZSpline. "
        "Please install it with `pip install interpax`."
    )
    

__all__ = ['CurveXYZSpline']

class CurveXYZSpline(JaxCurve):
    """
    
    """

    def __init__(self, coords : ArrayLike, knots : ArrayLike = None, method : str =  "cubic2", quadpoints : Union[ArrayLike,int] = None):
        """
        coords are xyz
        knots are in [0,1)
        method is one of the possible provided by interpax
        """
        
        if knots is None:
            knots = jnp.linspace(0, 1, len(coords), endpoint=False)
        elif len(knots) != len(coords):
            raise ValueError("Length of knots must match length of coords")
        elif jnp.any(jnp.diff(knots) <= 0):
            raise ValueError("Knots must be in strictly increasing order")
        elif knots[0] < 0 or knots[-1] >= 1:
            raise ValueError("Knots must be in the interval [0, 1)")

        self._knots = jnp.asarray(knots)
        self._method = method
        self._n = len(self._knots)
               
        self._interpolator = Interpolator1D(self._knots, coords, method=self._method, extrap=False, period=1)
    
        if quadpoints is None:
            quadpoints = jnp.linspace(0, 1, 100, endpoint=False)
        elif isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)
    
        self._dof_list = jnp.concatenate((jnp.asarray(knots), jnp.asarray(coords).flatten()))

        super().__init__(quadpoints, self.jaxcurve_pure, dofs=self._dof_list,
                        names=self._make_names(),
                        external_dof_setter=self.set_dofs_impl)

    def jaxcurve_pure(self, dofs, quadpoints):
        self.dof_list = dofs
        self._knots = dofs[:self._n]
        coords = dofs[self._n:].reshape((self._n, 3))
        self._interpolator = Interpolator1D(self._knots, coords, method=self._method, extrap=False, period=1)
        return self._interpolator(quadpoints)

    def _make_names(self):
        phi_names = [f'phi({i})' for i in range(self._n)]
        x_names = [f'x({i})' for i in range(self._n)]
        y_names = [f'y({i})' for i in range(self._n)]
        z_names = [f'z({i})' for i in range(self._n)]
        return phi_names + x_names + y_names + z_names

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 4 * self.self_n

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return jnp.array(self.dof_list)

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.dof_list = jnp.array(dofs)

    @property
    def knots(self):
        return self._knots
    
    @knots.setter
    def knots(self, knots : ArrayLike):
        pass
