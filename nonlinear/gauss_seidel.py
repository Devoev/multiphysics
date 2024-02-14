from dataclasses import dataclass

import numpy as np

from nonlinear.dynamic_iteration import DynamicIteration
from nonlinear.solve_nonlinear import EqFunction


@dataclass
class GaussSeidel(DynamicIteration):
    """Solves the nonlinear equation by a dynamic iteration with the Gauss-Seidel method."""

    def eval(self, yl: float, l: int, f: EqFunction, yk: np.ndarray, ys: np.ndarray) -> float:
        inputs = np.copy(yk)
        inputs[0:l] = np.copy(ys[0:l])
        inputs[l] = yl
        return f(inputs)[l]