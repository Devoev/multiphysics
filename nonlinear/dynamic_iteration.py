from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy as sp

from nonlinear.solve_nonlinear import NonLinearSolver, EqFunction


@dataclass
class DynamicIteration(NonLinearSolver, ABC):
    """
    Solves the nonlinear equation by a dynamic iteration.

    Uses ``sp.optimize.fsolve`` to solve for each component.
    """

    kmax: int
    """Number of dynamic iterations."""

    def __call__(self, f: EqFunction, y0: np.ndarray) -> np.ndarray:
        m = y0.size
        y = np.empty((m, self.kmax))  # TODO Don't save all old iterates
        y[:, 0] = y0

        for k in range(self.kmax - 1):
            yk = y[:, k]  # Values from previous iteration
            for l in range(m):
                y[l, k+1] = sp.optimize.fsolve(lambda x: self.eval(f, l, x, y[:, k+1], yk), yk[l])

        return y[:, self.kmax - 1]

    @abstractmethod
    def eval(self, f: EqFunction, l: int, yl: float, yk: np.ndarray, ys: np.ndarray) -> float:
        """
        Evaluates the ``l``-th component of ``f`` at variables depending on the unknown ``yl`` and already known ones.
        :param f: Equation function.
        :param l: Index of variable ``yl`` to solve for.
        :param yl: Unknown variable
        :param yk: Values of ``y`` at previous iteration ``k``. Array of size ``(m)``.
        :param ys: Values of ``y`` at current iteration ``s=k+1``. Array of size ``(m)``. Values at indices ``i>l`` are uninitialized.
        :return: The value of the ``l``-th component of the equation function.
        """
