import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags, block_diag, coo_matrix
from scipy.sparse.linalg import norm

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.top_mats import create_top_mats, create_p_mat
from fit.matrices.util import spdiag_pinv
from fit.mesh.box import Box, mesh_boxes
from fit.mesh.mesh import Mesh
from fit.solver.solve_poisson import solve_poisson

xmesh = np.linspace(0, 1, 2)
ymesh = np.linspace(0, 1, 2)
zmesh = np.linspace(0, 1, 2)
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
ds, dst, da, dat = create_geo_mats(msh)
ds = ds.toarray()
dst = dst.toarray()
da = da.toarray()
dat = dat.toarray()

#%%
_, _, _, nx, ny, nz, n, mx, my, mz = msh
diag = np.ones((4,n))
x_av = diags(diag, (0, -my, -mz, -my-mz), (n,n))
y_av = diags(diag, (0, -mz, -mx, -mz-mx), (n,n))
z_av = diags(diag, (0, -mx, -my, -mx-my), (n,n))
av = block_diag([x_av, y_av, z_av]).toarray()

plt.spy(av, markersize=.1)
plt.show()

#%%
sig = 1
sig_arr = np.full(msh.np, sig)
m1 = create_p2d_mat(msh, sig_arr).toarray()
m2 = create_p2d_mat(msh, sig).toarray()

plt.figure()
plt.spy(m1, markersize=.1)
plt.title("Array")
plt.show()

plt.figure()
plt.spy(m2, markersize=.1)
plt.title("Scalar")
plt.show()

print(np.linalg.norm(m1 - m2))
