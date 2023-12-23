from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from fit.mesh.mesh import Mesh


@dataclass
class Box:
    """A rectangular box with a value."""

    value: float
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]

    def indices(self, msh: Mesh) -> np.ndarray:
        """
        Computes the canonical indices of this box.
        :param msh: Mesh object.
        :return: Index array.
        """

        xmin, xmax = self.x_range
        ymin, ymax = self.y_range
        zmin, zmax = self.z_range

        idx = []
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                for k in range(zmin, zmax):
                    idx.append(msh.idx(i,j,k))

        return np.array(idx)


def mesh_boxes(msh: Mesh, boxes: List[Box], mat: float) -> np.ndarray:
    """
    Discretizes a boxed region of a material distribution.
    :param msh: Mesh object.
    :param boxes: Boxed regions.
    :param mat: Default material value.
    :return: Material array of size ``(msh.np)``.
    """

    mat = np.full(msh.np, mat, dtype=float)
    for box in boxes:
        mat[box.indices(msh)] = box.value

    return mat
