from dataclasses import astuple
from typing import Tuple

import numpy as np
from scipy.sparse import diags, spmatrix

from fit.mesh.mesh import Mesh


def create_geo_mats(msh: Mesh) -> Tuple[spmatrix, spmatrix, spmatrix, spmatrix]:
    """Creates the geometric matrices for the given ``msh``.

    :return: ``ds``, ``dst``, ``da`` and ``dat``.
    """

    xmesh, ymesh, zmesh, nx, ny, nz, n, *_ = msh

    # Primary grid
    dx = np.append(np.diff(xmesh), 0)
    dy = np.append(np.diff(ymesh), 0)
    dz = np.append(np.diff(zmesh), 0)

    dsx, dsy, dsz = create_ds_vecs(msh, dx, dy, dz)
    ds, da = create_geo_mats_from_ds_vecs(dsx, dsy, dsz)

    # Dual grid
    dxt = (np.concatenate([[0], dx[:-1]]) + dx) / 2
    dyt = (np.concatenate([[0], dy[:-1]]) + dy) / 2
    dzt = (np.concatenate([[0], dz[:-1]]) + dz) / 2

    dsxt, dsyt, dszt = create_ds_vecs(msh, dxt, dyt, dzt)
    dst, dat = create_geo_mats_from_ds_vecs(dsxt, dsyt, dszt)

    return ds, dst, da, dat


def create_ds_vecs(msh: Mesh, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates the ds vectors for the given ``msh``.

    :return: ``dsx``, ``dsy`` and ``dsz``.
    """

    _, _, _, nx, ny, nz, n, *_ = msh
    dsx = np.tile(dx, (ny * nz,))
    dsy = np.reshape(np.tile(dy, (nx, nz)), (n,))
    dsz = np.reshape(np.tile(dz, (nx * ny, 1)), (n,))
    return dsx, dsy, dsz


def create_geo_mats_from_ds_vecs(dsx: np.ndarray, dsy: np.ndarray, dsz: np.ndarray) -> Tuple[spmatrix, spmatrix]:
    """ Creates the geometrical matrices from the given ``ds`` vectors.
    :return: ``ds`` and ``da``.
    """
    ds: spmatrix = diags(np.concatenate([dsx, dsy, dsz]))
    da: spmatrix = diags(np.concatenate([dsy * dsz, dsz * dsx, dsx * dsy]))
    return ds, da
