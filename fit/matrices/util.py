import scipy.sparse as sp


def pinv(mat: sp.spmatrix) -> sp.spmatrix:
    """Computes the pseudo inverse of the given sparse diagonal matrix ``mat``."""

    mat = mat.tocoo()
    return sp.coo_matrix((1 / mat.data, (mat.row, mat.col)), shape=mat.shape)
