import numpy as np

from fit.mesh.mesh import Mesh


def pot_to_3d_arr(msh: Mesh, pot: np.ndarray) -> np.ndarray:
    """
    Reshapes the given FIT potential ``pot`` to an array of size ``(msh.nx, msh.ny, msh.nz)``.
    :param msh: Mesh object.
    :param pot: Potential array.
    :return: Reshaped potential array.
    """
    return np.reshape(pot, (msh.nx, msh.ny, msh.nz), order='F').transpose((1, 0, 2))


def vec_to_xyz(msh: Mesh, vec: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the ``x``, ``y`` and ``z`` component arrays of the given vector ``vec``.
    :param msh: Mesh object.
    :param vec: Vector array.
    :return: Component arrays ``vec_x``, ``vec_y``, ``vec_z``.
    """
    return vec[0:msh.np], vec[msh.np:2 * msh.np], vec[2 * msh.np:3 * msh.np]


def vec_to_3d_arr(msh: Mesh, vec: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshapes the given FIT vector ``vec`` to 3 arrays of size ``(msh.nx, msh.ny, msh.nz)``.
    :param msh: Mesh object.
    :param vec: Vector array.
    :return: Reshaped vectors. ``vec_x``, ``vec_y``, ``vec_z``.
    """
    vec_x, vec_y, vec_z = vec_to_xyz(msh, vec)
    return pot_to_3d_arr(msh, vec_x), pot_to_3d_arr(msh, vec_y), pot_to_3d_arr(msh, vec_z)
