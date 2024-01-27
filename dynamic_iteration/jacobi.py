from typing import Callable

import numpy as np

from ode.solve_ivp import solve_ivp


def jacobi(t: np.ndarray, y0: float, kmax: int, f1: Callable[[float, float], float],
           f2: Callable[[float, float], float]) -> [np.ndarray, np.ndarray]:
    n = t.size
    y1 = np.empty((n,))
    y2 = np.empty((n,))
    y1[0] = y2[0] = y0

    for k in range(kmax):
        y1_new = solve_ivp(t, y1[0], lambda i, ti, yi: f1(yi, y2[i]))
        y2_new = solve_ivp(t, y2[0], lambda i, ti, yi: f2(y1[i], yi))
        y1 = y1_new
        y2 = y2_new

    return y1, y2