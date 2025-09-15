from .curve import JaxCurve
from numpy.typing import ArrayLike
import jax.numpy as jnp

import warnings

try:
    from interpax import interp1d
except ImportError:
    interp1d = None

try:
    from scipy.interpolate import CubicSpline
except:
    CubicSpline = None

__all__ = []

class CurveXYZSpline(JaxCurve):
    """
    
    """

    def __init__(self, points : ArrayLike):

        pass
    

