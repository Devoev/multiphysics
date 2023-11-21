import numpy as np
from matplotlib import pyplot as plt

from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.top_mats import create_p_mat, create_top_mats
from fit.mesh.mesh import Mesh

xmesh = np.linspace(0, 10, 5)
ymesh = np.linspace(0, 10, 5)
zmesh = np.linspace(0, 10, 5)
msh = Mesh(xmesh, ymesh, zmesh)

ds, dst, da, dat = create_geo_mats(msh)

plt.spy(dst)
plt.show()