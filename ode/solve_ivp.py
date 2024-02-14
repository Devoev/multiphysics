import numpy as np

from ode.UpdateScheme import UpdateScheme

# TODO: This only works for explicit schemes. Fix for implicit ones!


def solve_ivp(t: np.ndarray, y0: np.ndarray, update: UpdateScheme) -> np.ndarray:
    """
    Solves the initial value problem ``y' = f(t,y)`` with vector valued unknown.
    :param t: Discrete time steps. Array of size ``(n)``.
    :param y0: Initial value.
    :param update: Update scheme.
    """

    n = t.size      # Number of time steps
    m = y0.size     # Number of unknowns
    h = np.diff(t)
    y = np.empty((n,m))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = y[i] + h[i]*update(t[i], [i+1], y[i], y[i+1], h[i])

    return y
