import numpy as np
from matplotlib import pyplot as plt

from fit.mesh.mesh import Mesh


def plot_pot(msh: Mesh, pot: np.ndarray, nz: int, xlabel='$x$ (m)', ylabel='$y$ (m)', title='', **kwargs):
    """
    Plots the given scalar potential ``pot`` on a ``z``-surface as a ``contourf`` plot.
    :param msh: Mesh struct
    :param pot: Potential array of size (msh.np).
    :param nz: Number of ``z``-surface.
    :param xlabel: Label of x-axis.
    :param ylabel: Label of y-axis.
    :param title: Title of the plot.
    """

    pot_z = np.reshape(pot, (msh.nz, msh.ny, msh.nx))
    pot_z = pot_z[nz, :, :]
    plt.rcParams['text.usetex'] = True

    [x, y] = np.meshgrid(msh.xmesh, msh.ymesh)
    plt.contourf(x, y, pot_z, **kwargs)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.show()
