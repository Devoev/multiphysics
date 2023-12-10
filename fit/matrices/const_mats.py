import numbers

import numpy as np
import scipy.sparse as sp

from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.util import pinv
from fit.mesh.mesh import Mesh


def create_p2d_mat(msh: Mesh, mat: float | np.ndarray) -> sp.spmatrix:
    """Creates the constitutive matrix from the primary to dual grid.
    :param msh: Mesh object.
    :param mat: Material parameter for the constitutive relation. Either a single value or an array of size (msh.np).
    """

    _, _, _, nx, ny, nz, n, mx, my, mz = msh
    ds, _, da, dat = create_geo_mats(msh)

    if isinstance(mat, numbers.Number):
        dmat = sp.eye(3*n) * mat
    elif isinstance(mat, np.ndarray):
        diag = np.ones((4,n))

        x_av = sp.diags(diag, (0, -my, -mz, -my-mz), (n,n))
        y_av = sp.diags(diag, (0, -mz, -mx, -mz-mx), (n,n))
        z_av = sp.diags(diag, (0, -mx, -my, -mx-my), (n,n))
        av = sp.block_diag([x_av, y_av, z_av])

        mat = np.concatenate([mat, mat, mat])
        dmat = sp.diags(0.25 * pinv(dat) @ av @ da @ mat, 0, (3 * n, 3 * n))
    else:
        raise Exception("not implemented")

    return dat @ dmat @ pinv(ds)


def create_d2p_mat(msh: Mesh, mat: float) -> sp.spmatrix:
    """Creates the constitutive matrix from the dual to primary grid.
    :param msh: Mesh object.
    :param mat: Material parameter for the constitutive relation.
    """

    _, dst, da, _ = create_geo_mats(msh)
    const_mat = mat * sp.eye(3 * msh.np)
    return dst @ const_mat @ pinv(da)
