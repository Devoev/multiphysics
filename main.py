import numpy as np
from matplotlib import pyplot as plt

from ode.euler_schemes import ExplicitEuler
from ode.solve_ivp import solve_ivp

# %% Simulation options
n = 100  # Time steps
T = 5  # Time interval
t = np.linspace(0, T, n)


# %% Define problem
def f(t: float, y: np.ndarray) -> np.array:
    return y


update = ExplicitEuler(f)

# %% Solve
y = solve_ivp(t, np.array([1,2,3,4]), update)

# %% Plot
plt.plot(t, y)
plt.show()
