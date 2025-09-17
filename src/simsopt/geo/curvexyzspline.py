from .._core.descriptor import PositiveInteger
from numpy.typing import ArrayLike
from typing import Union

from .curve import JaxCurve
import jax.numpy as jnp
import numpy as np

try:
    from interpax import interp1d
except ImportError:
    raise ImportError(
        "The interpax package is required to use CurveXYZSpline. "
        "Please install it with `pip install interpax`."
    )
    

__all__ = ['CurveXYZSpline']

  

class CurveXYZSpline(JaxCurve):
    
    INTERPAX_METHODS_1D = (
    "cubic",
    "cubic2",
    "cardinal",
    "catmull-rom",
    "akima",
    "monotonic",
    "monotonic-0", 
    "nearest", 
    "linear")

    def __init__(self, quadpoints : Union[ArrayLike,int] = None, dofs : ArrayLike = None):
        """
        coords are xyz
        knots are in [0,1)
        method is one of the possible provided by interpax
        """

        # default example with 3 knots
        self._knots = np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0]])
        self._arclengths = np.linspace(0, 1, len(self._knots), endpoint=False)
        self._method = "cubic2"
        self._n = len(self._knots)

        if quadpoints is None:
            quadpoints = np.linspace(0, 1, 100, endpoint=False)
        elif isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
    
        if dofs is None:
            JaxCurve.__init__(self, quadpoints, self.jaxcurve_pure, external_dof_setter=CurveXYZSpline.set_dofs_impl,
                           names=self._make_names(self.get_dofs()),
                           x0=self.get_dofs())
        else:
            JaxCurve.__init__(self, quadpoints, self.jaxcurve_pure, external_dof_setter=CurveXYZSpline.set_dofs_impl,
                           dofs=dofs,
                           names=self._make_names(dofs))

    def jaxcurve_pure(self, dofs, quadpoints):
        n = dofs.shape[0] // 4
        knots = dofs[:n]
        coords = dofs[n:].reshape((n, 3))
        return interp1d(quadpoints, knots, coords, method=self._method, extrap=False, period=1)

    def _make_names(self, dofs):
        n = dofs.shape[0] // 4
        phi_names = [f'arclength({i})' for i in range(n)]
        x_names = [f'x({i})' for i in range(n)]
        y_names = [f'y({i})' for i in range(n)]
        z_names = [f'z({i})' for i in range(n)]
        return phi_names + x_names + y_names + z_names

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 4 * self._n

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return jnp.array(np.hstack((self._arclengths, self._knots.flatten())))

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        n = dofs.shape[0] // 4
        self._n = n
        self._arclengths = np.array(dofs[:n])
        self._knots = np.array(dofs[n:]).reshape((n, 3))

    def plot_knots(self, **kwargs):
        import matplotlib.pyplot as plt

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            fig = ax.get_figure()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot the knots as points in 3D space
        ax.plot(self._knots[:, 0], self._knots[:, 1], self._knots[:, 2], 'o-', label='Knots')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Knots in 3D')
        ax.grid()

        if 'show' in kwargs and kwargs.pop('show'):
            plt.show()
        return fig, ax

    # arclengths property : arclengths on the curve at the knots
    @property
    def arclengths(self):
        return self._arclengths
    @arclengths.setter
    def arclengths(self, arclengths : ArrayLike):
        if arclengths is None:
            raise ValueError("arclengths cannot be None")
        elif len(arclengths) != len(self._arclengths):
            raise ValueError("Length of arclengths must match length of coords")
        elif not all(isinstance(a, float) or isinstance(a, np.floating) for a in arclengths):
            raise ValueError("All elements of arclengths must be floats")
        elif jnp.any(jnp.diff(arclengths) <= 0):
            raise ValueError("Arclengths must be in strictly increasing order")
        elif arclengths[0] < 0 or arclengths[-1] >= 1:
            raise ValueError("Arclengths must be in the interval [0, 1)")
        self._arclengths = arclengths
        self.dof_list[:self._n] = arclengths

    # knots property : provided points where the curve values are specified 
    @property
    def knots(self):
        return self._knots
    @knots.setter
    def knots(self, knots : ArrayLike):
        if knots is None:
            raise ValueError("knots cannot be None")
        elif len(knots) != len(self._knots):
            raise ValueError("Length of knots must match length of coords")
        elif not all(isinstance(k, (list, np.ndarray)) and len(k) == 3 for k in knots):
            raise ValueError("All elements of knots must be array-like of length 3")
        self._knots = np.array(knots)
        self.dof_list[self._n:] = self._knots.flatten()
    
    # method of interpolation
    @property
    def method(self):
        return self._method
    @method.setter
    def method(self, method : str):
        if method not in self.INTERPAX_METHODS_1D:
            raise ValueError(f"Method must be one of {self.INTERPAX_METHODS_1D}")
        self._method = method

    @classmethod
    def from_knots(cls, knots : ArrayLike, method : str = "cubic2", arclenghts : ArrayLike = None, quadpoints : Union[ArrayLike,int] = None):
        
        if arclenghts is None:
            arclenghts = np.linspace(0, 1, len(knots), endpoint=False)
 
        dofs = np.hstack((arclenghts, np.array(knots).flatten()))
        instance = cls(quadpoints=quadpoints, dofs=dofs)
        instance.method = method
        return instance