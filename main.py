import numpy as np
from matplotlib import pyplot as plt

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.top_mats import create_top_mats
from fit.mesh.mesh import Mesh

xmesh = np.linspace(0, 10, 3)
ymesh = np.linspace(0, 10, 3)
zmesh = np.linspace(0, 10, 3)
msh = Mesh(xmesh, ymesh, zmesh)

c, s, st = create_top_mats(msh)

meps = create_p2d_mat(msh, 8.854e-12)
mmui = create_p2d_mat(msh, 1 / 1.256e-6)

plt.spy(st, markersize=.1)
plt.show()
