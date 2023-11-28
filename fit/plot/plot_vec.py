import numpy as np
from matplotlib import pyplot as plt

from fit.mesh.mesh import Mesh
from fit.util.array import vec_to_3d_arr


def plot_vec(msh: Mesh, vec: np.ndarray, nz: int, xlabel='$x$ (m)', ylabel='$y$ (m)', title='', **kwargs):
    """
    Plots the given vector field ``vec`` on a ``z``-surface as a ``quiver`` plot.
    :param msh: Mesh object.
    :param vec: Vector field array of size (3*msh.np).
    :param nz: Number of ``z``-surface.
    :param xlabel: Label of x-axis.
    :param ylabel: Label of y-axis.
    :param title: Title of the plot.
    """

    x, y = np.meshgrid(msh.xmesh, msh.ymesh)
    vec_x, vec_y, _ = vec_to_3d_arr(msh, vec)
    ux = vec_x[:,:,nz]
    uy = vec_y[:,:,nz]

    # Normalize
    u_abs = np.sqrt(ux**2 + uy**2)

    plt.quiver(x, y, ux, uy, u_abs, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.show()
