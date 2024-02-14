from abc import abstractmethod, ABC

import numpy as np


class UpdateScheme(ABC):
    """
    An update scheme for one-step schemes. The update is defined as

    ``yj = yi + hi*P(ti,tj,yi,yj,hi)``

    with ``P`` being the update scheme function and ``j=i+1``.
    """

    @abstractmethod
    def __call__(self, ti: float, tj: float, yi: np.ndarray, yj: np.ndarray, hi: float) -> np.ndarray:
        """Evaluates this update scheme at ``(ti,tj,yi,yj,hi)`` with ``j=i+1``."""
