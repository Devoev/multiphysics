from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, spmatrix, bmat

from fit.mesh.mesh import Mesh


def create_top_mats(msh: Mesh) -> Tuple[spmatrix, spmatrix, spmatrix]:
    """Creates the topological matrices for the given ``msh``.
    :return: The ``g``, ``c`` and ``d`` matrices.
    """

    n = msh.np
    px = create_p_mat(n, msh.nx)
    py = create_p_mat(n, msh.ny)
    pz = create_p_mat(n, msh.nz)

    g = bmat([[-px.T, -py.T, -pz.T]]).T
    c = bmat([[None, -pz, py], [pz, None, -px], [-py, pz, None]])
    d = bmat([[px, py, pz]])
    return g, c, d


def create_p_mat(n: int, m: int) -> spmatrix:
    """Creates a discrete differentiation matrix.

    :param n: Number of grid points.
    :param m: Increment in either ``x``, ``y`` or ``z`` direction.
    """

    rows = np.concatenate((np.arange(n), np.arange(n - m)))
    cols = np.concatenate((np.arange(n), np.arange(m, n)))
    vals = np.concatenate((-np.ones((n, 1)), np.ones((n - m, 1))))[:, 0]
    return csr_matrix((vals, (rows, cols)), shape=(n, n))
