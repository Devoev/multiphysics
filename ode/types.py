from typing import Callable

import numpy as np

RHSFunction = Callable[[float, np.ndarray], float]
"""Signature of right hands side function for initial value problems ``f(t,y)`` with vector valued unknown."""
