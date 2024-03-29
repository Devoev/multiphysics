import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

from fit.matrices.top_mats import create_top_mats
from fit.mesh.mesh import Mesh


def solve_poisson(msh: Mesh, f: np.ndarray, const_mat: sp.spmatrix, bc: np.ndarray) -> np.ndarray:
    """
    Solves the `Poisson` problem.
    :param msh: Mesh object.
    :param f: Right hand side vector of size ``np``.
    :param const_mat: Constitutive matrix.
    :param bc: Boundary conditions vector of size ``np``. ``NaN`` at DOF indices.
    :return: Solution vector.
    """

    # Assemble matrix
    g, _, _ = create_top_mats(msh)
    A = g.T @ const_mat @ g

    # Deflate system
    idx_dof = np.isnan(bc)
    idx_bc = ~idx_dof
    bc = bc[idx_bc]

    b = f[idx_dof] - A[idx_dof,:][:,idx_bc] @ bc
    A = A[idx_dof,:][:,idx_dof]

    # Solve system
    x = np.empty((msh.np,))
    x[idx_dof], info = sp.linalg.gmres(A, b, tol=1e-13, restart=20, maxiter=msh.np)
    print(f"gmres terminated with exit code {info}")
    x[idx_bc] = bc
    return x
