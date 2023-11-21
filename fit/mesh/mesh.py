from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """A rectangular mesh in 3D."""

    xmesh: np.ndarray
    ymesh: np.ndarray
    zmesh: np.ndarray

    def __post_init__(self):
        self.nx = self.xmesh.size
        self.ny = self.ymesh.size
        self.nz = self.zmesh.size
        self.np = self.nx * self.ny * self.nz
        self.mx = 1
        self.my = self.nx
        self.mz = self.nx * self.nz
