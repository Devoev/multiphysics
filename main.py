import numpy as np
from matplotlib import pyplot as plt

from fit.matrices.create_top_mats import create_px
from fit.mesh.mesh import Mesh

xmesh = np.linspace(0, 10, 10)
ymesh = np.linspace(0, 10, 10)
zmesh = np.linspace(0, 10, 10)
msh = Mesh(xmesh, ymesh, zmesh)

px = create_px(msh.np)
plt.spy(px)
plt.show()
