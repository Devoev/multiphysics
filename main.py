import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import norm

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.top_mats import create_p_mat, create_top_mats
from fit.matrices.util import spdiag_pinv
from fit.mesh.mesh import Mesh

xmesh = np.linspace(0, 10, 5)
ymesh = np.linspace(0, 10, 5)
zmesh = np.linspace(0, 10, 5)
msh = Mesh(xmesh, ymesh, zmesh)

c, s, st = create_top_mats(msh)

deps = create_p2d_mat(msh, 1)
