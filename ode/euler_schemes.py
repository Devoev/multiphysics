import numpy as np
import scipy as sp

from ode.types import RHSFunction


def explicit_euler(i: int, t: np.ndarray, y: np.ndarray, h: np.ndarray, f: RHSFunction) -> float:
    """Explicit Euler update scheme."""
    return y[i] + h[i]*f(i, t[i], y[i])


def implicit_euler(i: int, t: np.ndarray, y: np.ndarray, h: np.ndarray, f: RHSFunction) -> float:
    """Implicit Euler update scheme."""
    return sp.optimize.fsolve(lambda yn: yn - y[i] - h[i] * f(i, t[i + 1], yn), y[i])[0]