from abc import abstractmethod, ABC


class UpdateScheme(ABC):
    """
    An update scheme for one-step schemes. The update is defined as

    ``yj = yi + hi*P(ti,tj,yi,yj,hi)``

    with ``P`` being the update scheme function and ``j=i+1``.
    """

    @abstractmethod
    def __call__(self, ti: float, tj: float, yi, yj, hi: float) -> float:
        """Evaluates this update scheme at ``(ti,tj,yi,yj,hi)`` with ``j=i+1``."""
