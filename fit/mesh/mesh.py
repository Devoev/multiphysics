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

    def idx(self, i: int, j: int, k: int) -> int:
        """Calculates the canonical index."""
        return 1 + (i-1)*self.mx + (j-1)*self.my + (k-1)*self.mz
