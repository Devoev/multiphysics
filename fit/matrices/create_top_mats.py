import numpy as np
from scipy.sparse import csr_array, sparray

from fit.mesh.mesh import Mesh


def create_top_mats(msh: Mesh):
    """Creates the topological matrices for the given ``msh``."""
    ...


def create_px(n: int) -> sparray:
    """Creates the discrete differentiation matrix in ``x`` direction."""

    rows = np.concatenate((np.arange(1, n), np.arange(1, n-1)))
    cols = np.concatenate((np.arange(1, n), np.arange(2, n)))
    vals = np.concatenate((-np.ones((1, n)), np.ones((1, n-1))))
    return csr_array((vals, (rows, cols)), shape=(n, n))
