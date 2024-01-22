import numpy as np
from matplotlib import pyplot as plt

from dynamic_iteration.gauss_seidel import gauss_seidel

# %% Simulation options
kmin = 3    # Inclusive
kmax = 15   # Exclusive
n = 100     # Time steps
T = 5       # Time interval
t = np.linspace(0, T, n)
dyn_iter_solver = gauss_seidel


# %% Define problem
def f1(y1: float, y2: float) -> np.array:
    return y1 * y2


def f2(y1: float, y2: float) -> np.array:
    return np.sin(y1) - np.cos(y2)


# %% Solve
y1 = np.empty((n, kmax - kmin))
y2 = np.empty((n, kmax - kmin))

for k in range(kmin, kmax):
    i = k - kmin
    y1[:, i], y2[:, i] = dyn_iter_solver(t, 1, k, f1, f2)

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
