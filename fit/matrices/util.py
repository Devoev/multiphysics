import scipy.sparse as sp


def pinv(mat: sp.dia_matrix) -> sp.coo_matrix:
    """Computes the pseudo inverse of the given sparse diagonal matrix ``mat``."""

    # TODO: Change to dia_matrix
    mat = mat.tocoo()
    return sp.coo_matrix((1 / mat.data, (mat.row, mat.col)), shape=mat.shape)
