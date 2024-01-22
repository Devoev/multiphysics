from typing import Callable

import numpy as np
import scipy as sp


def euler(t: np.ndarray, y0: float, f: Callable[[int, float, float], float]) -> np.ndarray:
    n = t.size
    h = np.diff(t)
    y = np.empty((n,))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = sp.optimize.fsolve(lambda x: x - y[i] - h[i] * f(i, t[i + 1], x), y[i])[0]

    return y