from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """A rectangular mesh in 3D."""

    xmesh: np.ndarray
    ymesh: np.ndarray
    zmesh: np.ndarray

    def __post_init__(self):
        self.nx: int = self.xmesh.size
        self.ny: int = self.ymesh.size
        self.nz: int = self.zmesh.size
        self.np: int = self.nx * self.ny * self.nz
        self.mx: int = 1
        self.my: int = self.nx
        self.mz: int = self.nx * self.nz

    def idx(self, i: int, j: int, k: int) -> int:
        """Calculates the canonical index."""
        return 1 + (i-1)*self.mx + (j-1)*self.my + (k-1)*self.mz

    def __iter__(self):
        return iter((self.xmesh, self.ymesh, self.zmesh, self.nx, self.ny, self.nz, self.np, self.mx, self.my, self.mz))
