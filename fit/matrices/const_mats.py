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
    dmat = mat * eye(3*msh.np)
    return dat*dmat*spdiag_pinv(ds)
