import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import norm

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.top_mats import create_top_mats, create_p_mat
from fit.mesh.mesh import Mesh
from fit.solver.solve_poisson import solve_poisson

xmesh = np.linspace(0, 10, 10)
ymesh = np.linspace(0, 10, 10)
zmesh = np.linspace(0, 10, 2)
msh = Mesh(xmesh, ymesh, zmesh)


#%%
n = msh.np
px = create_p_mat(n, msh.mx)
py = create_p_mat(n, msh.my)
pz = create_p_mat(n, msh.mz)

#%%
g, c, d = create_top_mats(msh)
print(norm(c @ g))
print(norm(d @ c))
print(norm(g.T @ c.T))

#%%
plt.spy(c, markersize=0.1)
plt.show()

# #%%
# bc = np.full(msh.np, np.nan)
# idx_bc = [msh.idx(10, j, 1) for j in range(10)]
# bc[idx_bc] = 10
#
# #%%
# eps = 8.854e-12
# phi = solve_poisson(msh, np.zeros(msh.np), eps, bc)
