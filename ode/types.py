from typing import Callable

import numpy as np

RHSFunction = Callable[[float, float], float]
"""Signature of right hands side function for initial value problems. ``f(t,y)``"""
