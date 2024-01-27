import numpy as np

from ode.euler_schemes import implicit_euler
from ode.types import RHSFunction, UpdateScheme


def solve_ivp(t: np.ndarray, y0: float, f: RHSFunction, update: UpdateScheme = implicit_euler):
    """
    Solves the initial value problem ``y' = f(t,y)``.
    :param t: Discrete time steps. Array of size ``(n)``.
    :param y0: Initial value.
    :param f: Right hand side function.
    :param update: Update scheme. Implicit Euler by default.
    """

    n = t.size
    h = np.diff(t)
    y = np.empty((n,))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = update(i, t, y, h, f)

    return y
