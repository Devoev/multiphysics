import numpy as np
from scipy.sparse.linalg import spsolve

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.top_mats import create_top_mats
from fit.mesh.mesh import Mesh


def solve_poisson(msh: Mesh, f: np.ndarray, mat: float, bc: np.ndarray) -> np.ndarray:
    """
    Solves the `Poisson` problem.
    :param msh: Mesh object.
    :param f: Right hand side vector of size ``np``.
    :param mat: Material inside the domain.
    :param bc: Boundary conditions vector of size ``np``. ``NaN`` at DOF indices.
    :return: Solution vector.
    """

    # Assemble matrix
    g, _, d = create_top_mats(msh)
    const_mat = create_p2d_mat(msh, mat)
    A = -d @ const_mat @ g

    # Deflate system
    idx_dof = np.isnan(bc)
    idx_bc = ~idx_dof
    bc = bc[idx_bc]

    b = f[idx_dof] - A[idx_dof,:][:,idx_bc] @ bc
    A = A[idx_dof,:][:,idx_dof]

    # Solve system
    x = np.empty((msh.np,))
    x[idx_dof] = spsolve(A, b)
    x[idx_bc] = bc
    return x
