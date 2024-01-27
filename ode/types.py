from typing import Callable

import numpy as np

RHSFunction = Callable[[int, float, float], float]
"""Signature of right hands side function for initial value problems. ``f(i,ti,yi)``"""

UpdateScheme = Callable[[int, np.ndarray, np.ndarray, np.ndarray, RHSFunction], float]
"""Signature of an update scheme for initial value problems. ``update(i,t,y,h,f)``"""
