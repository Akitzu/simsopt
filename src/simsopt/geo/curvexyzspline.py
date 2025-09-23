from .._core.descriptor import PositiveInteger
from numpy.typing import ArrayLike
from typing import Union

from .._core.optimizable import DOFs
from .curve import JaxCurve
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    Warning("Matplotlib not found. The plot method will not work.")

try:
    from interpax import interp1d
except ImportError:
    raise ImportError(
        "The interpax package is required to use CurveXYZSpline. "
        "Please install it with `pip install interpax`."
    )
    

__all__ = ['CurveXYZSpline']

import logging
logger = logging.getLogger(__name__)
  

class CurveXYZSpline(JaxCurve):
    """
    Class representing a 3D curve defined by a set of knots and interpolated using various spline methods.
    """

    INTERPAX_METHODS_1D = (
    "cubic",
    "cubic2",
    "cardinal",
    "catmull-rom",
    "akima",
    "monotonic",
    "monotonic-0",
    "linear")

    def __init__(self, quadpoints : Union[ArrayLike,int] = None, dofs : DOFs = None, **kwargs):
        """

        """

        # default example with a triangle
        knots = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [1.0, 1.0, 0.0]])
        arclengths = np.linspace(0, 1, len(knots), endpoint=False)
        x0_example = np.hstack((arclengths, np.array(knots).flatten()))
        
        # initialize the method and use cubic2 as default
        self.method = kwargs.pop("method", "cubic2")

        # sets up the quadpoints 
        if quadpoints is None:
            quadpoints = np.linspace(0, 1, 100, endpoint=False)
        elif isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        # call to parent constructor
        if dofs is None:
            if "x0" not in kwargs:
                x0 = x0_example
                self._knots_size = len(knots)
            else:
                x0 = kwargs.pop("x0")
                if len(x0) % 4 != 0:
                    raise ValueError("Length of x0 must be a multiple of 4")
                self._knots_size = len(x0) // 4

            JaxCurve.__init__(self, quadpoints, self.jaxcurve_pure, 
                              names=self._make_names(self._knots_size), 
                              x0=x0, **kwargs)
        else:
            JaxCurve.__init__(self, quadpoints, self.jaxcurve_pure,
                              dofs=dofs, **kwargs)

    def jaxcurve_pure(self, dofs, quadpoints):
        n = dofs.shape[0] // 4
        arclengths = dofs[:n]
        knots = dofs[n:].reshape((n, 3))
        return interp1d(quadpoints, arclengths, knots, method=self._method, extrap=False, period=1)

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 4 * self._knots_size

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return self.local_full_x

    def set_dofs_impl(self, x : ArrayLike):
        """
        There are no local dofs to set as they are all enclosed in the DOFs object.
        """
        pass

    def fix_arclengths(self):
        """
        This function fixes the arclengths dofs, so that only the knots can be modified.
        """
        for i in range(self._knots_size):
            self.fix(f'arclength({i})')
        
    # arclengths property : arclengths on the curve at the knots
    @property
    def arclengths(self):
        return self.local_full_x[:self._knots_size]
    
    @arclengths.setter
    def arclengths(self, arclengths : ArrayLike):
        if len(arclengths) != self._knots_size:
            raise ValueError("Length of arclengths must match the number of knots")
        elif not all(isinstance(a, float) or isinstance(a, np.floating) for a in arclengths):
            raise ValueError("All elements of arclengths must be floats")
        elif np.any(np.diff(arclengths) <= 0):
            raise ValueError("Arclengths must be in strictly increasing order")
        elif arclengths[0] < 0 or arclengths[-1] >= 1:
            raise ValueError("Arclengths must be in the interval [0, 1)")
        elif self.local_full_dof_size != self.local_dof_size:
            raise ValueError("Cannot set arclengths when  of DOFs has changed. Create a new CurveXYZSpline instance instead.")

        # update the DOFS object
        self.x = np.hstack((arclengths, self.x[self._knots_size:]))

    # knots property : provided points where the curve values are specified 
    @property
    def knots(self):
        return self.local_full_x[self._knots_size:].reshape((self._knots_size, 3))
    
    @knots.setter
    def knots(self, knots : ArrayLike):
        if len(knots) != self._knots_size:
            raise ValueError("Length of new knots must match length of old knots")
        elif not all(isinstance(k, (list, np.ndarray)) and len(k) == 3 for k in knots):
            raise ValueError("All elements of knots must be array-like of length 3")

        # update the DOFS object
        self.x = np.hstack((self.x[:self._knots_size], np.array(knots).flatten()))
    
    # method of interpolation
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, method : str):
        if method not in self.INTERPAX_METHODS_1D:
            raise ValueError(f"Method must be one of {self.INTERPAX_METHODS_1D}")
        self._method = method
   
    @staticmethod
    def _make_names(nknots : PositiveInteger):
        """
        Static method to create names for the dofs of the CurveXYZSpline object.
        """
        phi_names = [f'arclength({i})' for i in range(nknots)]
        x_names = [f'x({i})' for i in range(nknots)]
        y_names = [f'y({i})' for i in range(nknots)]
        z_names = [f'z({i})' for i in range(nknots)]
        return phi_names + x_names + y_names + z_names

    @classmethod
    def from_knots(cls, knots : ArrayLike, method : str = "cubic2", arclenghts : ArrayLike = None, quadpoints : Union[ArrayLike,int] = None):
        """
        
        """

        if arclenghts is None:
            arclenghts = np.linspace(0, 1, len(knots), endpoint=False)

        return cls(quadpoints=quadpoints, x0=np.hstack((arclenghts, np.array(knots).flatten())), method=method)
    
    def plot_knots(self, **kwargs):
        """
        Plots the knots (known points) used to  in 3D space.
        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            fig = ax.get_figure()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot the knots as points in 3D space
        knots = self.knots
        ax.plot(*knots.T, 'o', label='Knots')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid()

        if 'show' not in kwargs or kwargs.pop('show'):
            plt.show()
        return ax