import numpy as np

from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.util import pinv
from fit.mesh.mesh import Mesh


def interpolate_field(msh: Mesh, vec: np.ndarray) -> np.ndarray:
    """
    Interpolates the given field strength ``vec``.
    :param msh: Mesh object.
    :param vec: Field array of size ``(3*msh.np)``.
    :return: Interpolated field array ``(3*msh.np)``.
    """

    _, _, _, nx, ny, nz, n, mx, my, mz = msh
    ds, *_ = create_geo_mats(msh)

    vec = pinv(ds) @ vec
    return vec


def interpolate_flux(msh: Mesh, vec: np.ndarray) -> np.ndarray:
    """
    Interpolates the given flux density ``vec``.
    :param msh: Mesh object.
    :param vec: Flux array of size ``(3*msh.np)``.
    :return: Interpolated flux array ``(3*msh.np)``.
    """

    _,_,da,_ = create_geo_mats(msh)

    vec = pinv(da) @ vec
    return vec

# eEdge = eBow[:, 0] * nullInvVector(DSDiag[0:3 * n_p]);
# eEdgeX = eEdge[0:n_p];
# eEdgeY = eEdge[n_p:2 * n_p];
# eEdgeZ = eEdge[2 * n_p:3 * n_p];
#
# # Interpolation des E-Feldes
# eX = zeros(n_p);
# eY = zeros(n_p);
# eZ = zeros(n_p);
# for k in range(n_z):
#     for j in range(n_y):
#         for i in range(n_x):
#             # Berechnung des kanonischen Index
#             n = i * M_x + j * M_y + k * M_z;
#
#             # X-Richtung
#             if i != 0 and i != n_x - 1:
#                 eX[n] = (eEdgeX[n - M_x] * DSDiag[n] + eEdgeX[n] * DSDiag[n - M_x]) / (
#                         DSDiag[n] + DSDiag[n - M_x]);
#
#             # Y-Richtung
#             if j != 0 and j != n_y - 1:
#                 eY[n] = (eEdgeY[n - M_y] * DSDiag[n_p + n] + eEdgeY[n] * DSDiag[n_p + n - M_y]) / (
#                         DSDiag[n_p + n] + DSDiag[n_p + n - M_y]);
#
#             # Z-Richtung
#             if k != 0 and k != n_z - 1:
#                 eZ[n] = (eEdgeZ[n - M_z] * DSDiag[2 * n_p + n] + eEdgeZ[n] * DSDiag[2 * n_p + n - M_z]) / (
#                         DSDiag[2 * n_p + n] + DSDiag[2 * n_p + n - M_z]);
#
# eField = hstack([eX, eY, eZ]);
#
# # meshP ist das reduzierte Gitter, das heißt ohne 1. und letzten Punkt
# meshP = mesh(self.xmesh[2:-2], self.ymesh[2:-2], self.zmesh[2:-2]);

# return meshP, eField
