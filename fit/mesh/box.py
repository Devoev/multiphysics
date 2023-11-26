from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from fit.mesh.mesh import Mesh


@dataclass
class Box:
    """A rectangular box with a value."""

    value: float
    x_idx: Tuple[int, int]
    y_idx: Tuple[int, int]
    z_idx: Tuple[int, int]


def mesh_boxes(msh: Mesh, boxes: List[Box], mat: float) -> np.ndarray:
    """
    Discretizes a boxed region of a material distribution.
    :param msh: Mesh object.
    :param boxes: Boxed regions.
    :param mat: Default material value.
    :return: Material array of size (msh.np).
    """

    mat = mat * np.ones(msh.np,)
    for box in boxes:
        xmin, xmax = box.x_idx
        ymin, ymax = box.y_idx
        zmin, zmax = box.z_idx

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        x_idx = np.arange(xmin, xmax)
        y_idx = np.arange(ymin, ymax)
        z_idx = np.arange(zmin, zmax)

        idx = np.tile(x_idx, (dy,)) + np.tile(y_idx * msh.my, (dx,))
        idx = np.reshape(idx, (dx*dy))
        idx = np.tile(idx, (dz,)) + np.tile(z_idx * msh.mz, (dx*dy,))
        mat[idx] = box.value

    return mat
