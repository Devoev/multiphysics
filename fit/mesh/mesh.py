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
        self.mz: int = self.nx * self.ny

    def idx(self, i: int, j: int, k: int) -> int:
        """Calculates the canonical index."""
        return i*self.mx + j*self.my + k*self.mz

    def find_idx(self, x: float, y: float, z: float) -> float:
        """Finds the canonical index nearest to the given physical coordinates."""
        return self.idx(
            np.argmin(np.abs(self.xmesh - x)),
            np.argmin(np.abs(self.ymesh - y)),
            np.argmin(np.abs(self.zmesh - z))
        )

    def __iter__(self):
        return iter((self.xmesh, self.ymesh, self.zmesh, self.nx, self.ny, self.nz, self.np, self.mx, self.my, self.mz))
