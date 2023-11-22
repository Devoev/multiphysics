import numpy as np
from matplotlib import pyplot as plt

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.top_mats import create_top_mats
from fit.mesh.mesh import Mesh
from fit.solver.solve_poisson import solve_poisson

xmesh = np.linspace(0, 10, 10)
ymesh = np.linspace(0, 10, 10)
zmesh = np.linspace(0, 10, 1)
msh = Mesh(xmesh, ymesh, zmesh)

#%%
bc = np.full(msh.np, np.nan)
idx_bc = [msh.idx(10, j, 1) for j in range(10)]
bc[idx_bc] = 10

#%%
eps = 8.854e-12
phi = solve_poisson(msh, np.zeros(msh.np), eps, bc)
