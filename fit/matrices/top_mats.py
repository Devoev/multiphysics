from typing import Tuple

import numpy as np
import scipy.sparse as sp

from fit.mesh.mesh import Mesh


def create_top_mats(msh: Mesh) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
    """Creates the topological matrices for the given ``msh``.
    :return: The ``g``, ``c`` and ``d`` matrices.
    """

    n = msh.np
    px = create_p_mat(n, msh.mx)
    py = create_p_mat(n, msh.my)
    pz = create_p_mat(n, msh.mz)

    g = sp.bmat([[px.T, py.T, pz.T]]).T
    c = sp.bmat([[None, -pz, py], [pz, None, -px], [-py, px, None]])
    d = sp.bmat([[px, py, pz]])
    return g, c, d


def create_p_mat(n: int, m: int) -> sp.coo_matrix:
    """Creates a discrete differentiation matrix.

    :param n: Number of grid points.
    :param m: Increment in either ``x``, ``y`` or ``z`` direction.
    """

    rows = np.concatenate((np.arange(n), np.arange(n - m)))
    cols = np.concatenate((np.arange(n), np.arange(m, n)))
    vals = np.concatenate((-np.ones((n,)), np.ones((n - m,))))
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
