from scipy.linalg import pinv
from scipy.sparse import spmatrix, eye

from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.util import spdiag_pinv
from fit.mesh.mesh import Mesh


def create_p2d_mat(msh: Mesh, mat: float) -> spmatrix:
    """Creates the constitutive matrix from the primary to dual grid.
    :param msh: Mesh object.
    :param mat: Material parameter for the constitutive relation.
    """

    ds, _, _, dat = create_geo_mats(msh)
    const_mat = mat * eye(3 * msh.np)
    return dat * const_mat * spdiag_pinv(ds)


def create_d2p_mat(msh: Mesh, mat: float) -> spmatrix:
    """Creates the constitutive matrix from the dual to primary grid.
    :param msh: Mesh object.
    :param mat: Material parameter for the constitutive relation.
    """

    _, dst, da, _ = create_geo_mats(msh)
    const_mat = mat * eye(3 * msh.np)
    return dst * const_mat * spdiag_pinv(da)
