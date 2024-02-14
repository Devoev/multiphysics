from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from ode.UpdateScheme import UpdateScheme


class NonLinearSolver(ABC):
    """
    Solver for nonlinear update equations

    ``yj = yi + hi*P(ti,tj,yi,yj,hi)``

    with an implicit update scheme function ``P`` and ``j=i+1``.
    """

    @abstractmethod
    def __call__(self, ti: float, tj: float, yi: np.ndarray, hi: float) -> np.ndarray:
        """Solves the nonlinear update equation at ``(ti,tj,yi,yj,hi)`` for ``yj`` with ``j=i+1``."""


@dataclass
class FSolveUpdate(NonLinearSolver):
    """
    Solves the nonlinear update equation using ``scipy.optimize.fsolve``.
    """

    update: UpdateScheme
    """Implicit update scheme."""

    def __call__(self, ti: float, tj: float, yi: np.ndarray, hi: float) -> np.ndarray:
        return sp.optimize.fsolve(lambda x: x - yi - hi*self.update(ti, tj, yi, x, hi), yi)[0]
