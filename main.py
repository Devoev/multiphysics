import numpy as np

from nonlinear.gauss_seidel import GaussSeidel
from nonlinear.jacobi import Jacobi
from ode.euler_schemes import ImplicitEuler
from ode.solve_ivp import solve_ivp_implicit

# %% Simulation options
n = 100  # Time steps
T = 5  # Time interval
t = np.linspace(0, T, n)


# %% Define problem
def f(t: float, y: np.ndarray) -> np.array:
    return -y


def y_ana(t: np.ndarray) -> np.ndarray:
    return np.outer(y0, np.exp(-t)).T


y0 = np.array([1, 1, 1, 1])

# %% Solve
# y = solve_ivp_explicit(t, y0, ExplicitEuler(f))
y = solve_ivp_implicit(t, y0, ImplicitEuler(f), Jacobi(2))

# %% Plot
# plt.plot(t, y)
# plt.show()

# %% Error
err = np.linalg.norm(y - y_ana(t), axis=0) / np.linalg.norm(y_ana(t), axis=0)
print(f"Relative error for each component = {err*100} %")
