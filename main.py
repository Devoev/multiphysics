import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import norm

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.top_mats import create_top_mats, create_p_mat
from fit.mesh.box import Box, mesh_boxes
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
V0 = 1

bc = np.full(msh.np, np.nan)
idx_bc_1 = []
idx_bc_2 = []

for i in range(10):
    for k in range(2):
        idx_bc_1.append(msh.idx(0, i, k))
        idx_bc_2.append(msh.idx(10-1, i, k))

bc[idx_bc_1] = 0
bc[idx_bc_2] = V0

box_bc_1 = Box(0, (0, 1), (0, msh.ny), (0, msh.nz))
box_bc_2 = Box(V0, (10-1, 10), (0, msh.ny), (0, msh.nz))
bc_new = mesh_boxes(msh, [box_bc_1, box_bc_2], np.nan)

print(bc)
print(bc_new)
print(np.array_equal(bc, bc_new, equal_nan=True))
print(bc - bc_new)

# #%%
# bc = np.full(msh.np, np.nan)
# idx_bc = [msh.idx(10, j, 1) for j in range(10)]
# bc[idx_bc] = 10
#
# #%%
# eps = 8.854e-12
# phi = solve_poisson(msh, np.zeros(msh.np), eps, bc)
