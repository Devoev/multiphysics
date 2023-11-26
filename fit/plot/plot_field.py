import numpy as np
from matplotlib import pyplot as plt

from fit.mesh.mesh import Mesh


def plot_field(msh: Mesh, field: np.ndarray, nz: int, xlabel='$x$ (m)', ylabel='$y$ (m)', title='', **kwargs):
    """
    Plots the given vector field ``field`` on a ``z``-surface as a ``quiver`` plot.
    :param msh: Mesh object.
    :param field: Field strength array of size (3*msh.np).
    :param nz: Number of ``z``-surface.
    :param xlabel: Label of x-axis.
    :param ylabel: Label of y-axis.
    :param title: Title of the plot.
    :return:
    """

    x, y = np.meshgrid(msh.xmesh, msh.ymesh)
    ux = field[0:msh.np]
    ux = np.reshape(ux, (msh.nz, msh.ny, msh.nx))
    ux = ux[nz, :, :]
    uy = field[msh.np:2 * msh.np]
    uy = np.reshape(uy, (msh.nz, msh.ny, msh.nx))
    uy = uy[nz, :, :]

    # Normalize
    u_abs = np.sqrt(ux**2 + uy**2)
    # ux = ux / magnitude
    # uy = uy / magnitude

    plt.quiver(x, y, ux, uy, u_abs, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
