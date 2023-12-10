import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from fit.matrices.const_mats import create_p2d_mat
from fit.matrices.geo_mats import create_geo_mats
from fit.matrices.util import pinv
from fit.mesh.mesh import Mesh
from fit.plot.plot_pot import plot_pot
from fit.solver.solve_poisson import solve_poisson
from fit.util.interpolate import interpolate_field

xmesh = np.linspace(0, 1, 20)
ymesh = np.linspace(0, 1, 20)
zmesh = np.linspace(0, 1, 5)
msh = Mesh(xmesh, ymesh, zmesh)

# ds, dst, da, dat = create_geo_mats(msh)
# inv = pinv(ds)
#
# ds = ds.toarray()
# dst = dst.toarray()
# da = da.toarray()
# dat = dat.toarray()
#
# one = ds @ inv
# print(np.diag(one))

#
# print(np.linalg.norm(ds, ord=2))
# print(np.linalg.norm(dst, ord=2))
# print(np.linalg.norm(da, ord=2))
# print(np.linalg.norm(dat, ord=2))

eps0 = 8.85e-12
meps = create_p2d_mat(msh, eps0)
# print(sla.norm(meps, ord=2))
# meps = np.eye(3*msh.np) * eps0
# print(np.linalg.norm(meps, ord=2))
# print(np.linalg.matrix_rank(meps.toarray()))

q = np.zeros((msh.np,))
q[msh.find_idx(0.5, 0.5, 0.5)] = 1
bc = np.full((msh.np,), np.nan)
# bc[msh.find_idx(0.5, 0.5, 0.5)] = 1

# bc[msh.find_idx(0,0,0.5)] = 0
# bc[msh.find_idx(1,0,0.5)] = 0
# bc[msh.find_idx(0,1,0.5)] = 0
# bc[msh.find_idx(1,1,0.5)] = 0
pot = solve_poisson(msh, q, meps, bc)

plot_pot(msh, pot, nz=2, levels=20)
