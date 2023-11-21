import numpy as np
from matplotlib import pyplot as plt

from fit.matrices.create_top_mats import create_p_mat, create_top_mats
from fit.mesh.mesh import Mesh

xmesh = np.linspace(0, 10, 5)
ymesh = np.linspace(0, 10, 5)
zmesh = np.linspace(0, 10, 5)
msh = Mesh(xmesh, ymesh, zmesh)

# px2 = create_px(msh)
# px = create_p_mat(msh.np, msh.mx)
# py = create_p_mat(msh.np, msh.my)
# pz = create_p_mat(msh.np, msh.mz)
# plt.figure()
# plt.spy(px)
# plt.show()
# plt.figure()
# plt.spy(py)
# plt.show()
# plt.figure()
# plt.spy(pz)
# plt.show()

c,s,st = create_top_mats(msh)
plt.spy(st)
plt.show()
