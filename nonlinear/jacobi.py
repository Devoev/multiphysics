from dataclasses import dataclass

import numpy as np

from nonlinear.dynamic_iteration import DynamicIteration
from nonlinear.solve_nonlinear import EqFunction


@dataclass
class Jacobi(DynamicIteration):
    """Solves the nonlinear equation by a dynamic iteration with the Jacobi method."""

    def eval(self, f: EqFunction, l: int, yl: float, yk: np.ndarray, ys: np.ndarray) -> float:
        inputs = np.copy(yk)
        inputs[l] = yl
        return f(inputs)[l]
