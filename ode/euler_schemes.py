from dataclasses import dataclass

import numpy as np

from ode.UpdateScheme import UpdateScheme
from ode.types import RHSFunction


@dataclass
class ExplicitEuler(UpdateScheme):
    """
    Explicit euler scheme with the explicit update function

    ``P(ti,yi,hi) := f(ti, yi)``.
    """

    f: RHSFunction
    """Right hand side function this scheme approximates."""

    def __call__(self, ti: float, tj: float, yi: np.ndarray, yj: np.ndarray, hi: float) -> float:
        return self.f(ti, yi)


@dataclass
class ImplicitEuler(UpdateScheme):
    """
    Implicit euler scheme with the implicit update function

    ``P(tj,yj,hi) := f(tj, yj)``.
    """

    f: RHSFunction
    """Right hand side function this scheme approximates."""

    def __call__(self, ti: float, tj: float, yi: np.ndarray, yj: np.ndarray, hi: float) -> float:
        return self.f(tj, yj)



