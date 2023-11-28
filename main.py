import numpy as np

from fit.mesh.mesh import Mesh
from fit.util.interpolate import interpolate_field

xmesh = np.linspace(0, 1, 7)
ymesh = np.linspace(0, 1, 7)
zmesh = np.linspace(0, 1, 2)
msh = Mesh(xmesh, ymesh, zmesh)


E0 = 1
ebow = E0 * np.ones((3*msh.np,))
print(ebow)
print(interpolate_field(msh, ebow))
