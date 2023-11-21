from scipy.linalg import pinv
from scipy.sparse import spmatrix, eye

from fit.matrices.geo_mats import create_geo_mats
from fit.mesh.mesh import Mesh


def create_p2d_mat(msh: Mesh, mat: float) -> spmatrix:

    ds, _, _, dat = create_geo_mats(msh)
    dmat = mat * eye(3*msh.np)
    return dat*dmat*ds  # todo: sparse pinv of ds
