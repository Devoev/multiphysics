from typing import Callable

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


# %% Define problem
def f1(y1: float, y2: float) -> np.array:
    return y1 * y2


def f2(y1: float, y2: float) -> np.array:
    return np.sin(y1) - np.cos(y2)


kmin = 3    # Inclusive
kmax = 8    # Exclusive
n = 100
T = 5
t = np.linspace(0, T, n)


# %% Define solvers
def euler(t: np.ndarray, y0: float, f: Callable[[int, float, float], float]) -> np.ndarray:
    n = t.size
    h = np.diff(t)
    y = np.empty((n,))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = sp.optimize.fsolve(lambda x: x - y[i] - h[i] * f(i, t[i + 1], x), y[i])[0]

    return y


def jacobi(t: np.ndarray, y0: float, kmax: int, f1: Callable[[float, float], float],
           f2: Callable[[float, float], float]) -> [np.ndarray, np.ndarray]:
    n = t.size
    y1 = np.empty((n,))
    y2 = np.empty((n,))
    y1[0] = y2[0] = y0

    for k in range(kmax):
        y1_new = euler(t, y1[0], lambda i, ti, yi: f1(yi, y2[i]))
        y2_new = euler(t, y2[0], lambda i, ti, yi: f2(y1[i], yi))
        y1 = y1_new
        y2 = y2_new

    return y1, y2


def gauss_seidel(t: np.ndarray, y0: float, kmax: int, f1: Callable[[float, float], float],
                 f2: Callable[[float, float], float]) -> [np.ndarray, np.ndarray]:
    n = t.size
    y1 = np.empty((n,))
    y2 = np.empty((n,))
    y1[0] = y2[0] = y0

    for k in range(kmax):
        y1_new = euler(t, y1[0], lambda i, ti, yi: f1(yi, y2[i]))
        y2_new = euler(t, y2[0], lambda i, ti, yi: f2(y1_new[i], yi))
        y1 = y1_new
        y2 = y2_new

    return y1, y2


# %% Solve
y1 = np.empty((n, kmax - kmin))
y2 = np.empty((n, kmax - kmin))

for k in range(kmin, kmax):
    i = k - kmin
    y1[:, i], y2[:, i] = gauss_seidel(t, 1, k, f1, f2)

# %% Plot
for i in range(kmax - kmin):
    style = 'r' if i == kmax-kmin-1 else 'r--'
    plt.plot(t, y1[:, i], style)
    plt.draw()
plt.show()

for i in range(kmax - kmin):
    style = 'b' if i == kmax-kmin-1 else 'b--'
    plt.plot(t, y2[:, i], style)
    plt.draw()
plt.show()
