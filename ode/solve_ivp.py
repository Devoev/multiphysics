import numpy as np

from ode.update_scheme import UpdateScheme
from ode.solve_nonlinear import NonLinearSolver


def solve_ivp_explicit(t: np.ndarray, y0: np.ndarray, update: UpdateScheme) -> np.ndarray:
    """
    Solves the initial value problem ``y' = f(t,y)`` with a vector valued unknown by an explicit update scheme.
    :param t: Discrete time steps. Array of size ``(n)``.
    :param y0: Initial value.
    :param update: Explicit update scheme.
    """

    n = t.size      # Number of time steps
    m = y0.size     # Number of unknowns
    h = np.diff(t)
    y = np.empty((n,m))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = y[i] + h[i]*update(t[i], 0, y[i], y[i], h[i])    # TODO: remove calls to i+1 variables

    return y


def solve_ivp_implicit(t: np.ndarray, y0: np.ndarray, solver: NonLinearSolver) -> np.ndarray:
    """
    Solves the initial value problem ``y' = f(t,y)`` with a vector valued unknown by an implicit update scheme.
    :param t: Discrete time steps. Array of size ``(n)``.
    :param y0: Initial value.
    :param solver: Nonlinear solver.
    """

    n = t.size      # Number of time steps
    m = y0.size     # Number of unknowns
    h = np.diff(t)
    y = np.empty((n,m))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = solver(t[i], t[i+1], y[i], h[i])

    return y
