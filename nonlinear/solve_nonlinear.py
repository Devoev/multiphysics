from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy as sp


class NonLinearSolver(ABC):
    """
    Solver for nonlinear equations ``f(y) = 0`` with vector valued ``y``.
    """

    @abstractmethod
    def __call__(self, f: Callable[[np.ndarray], np.ndarray], y0: np.ndarray) -> np.ndarray:
        """Solves ``f(y) = 0`` with initial guess ``y0``."""


@dataclass
class FSolveUpdate(NonLinearSolver):
    """
    Solves the nonlinear equation using ``scipy.optimize.fsolve``.
    """

    def __call__(self, f: Callable[[np.ndarray], np.ndarray], y0: np.ndarray) -> np.ndarray:
        return sp.optimize.fsolve(lambda y: f(y), y0)[0]
