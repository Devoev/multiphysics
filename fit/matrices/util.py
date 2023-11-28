from scipy.sparse import spmatrix, csr_matrix


def pinv(mat: spmatrix) -> spmatrix:
    """Computes the pseudo inverse of the given sparse diagonal matrix ``mat``."""

    mat = mat.tocoo()
    return csr_matrix((1 / mat.data, (mat.row, mat.col)), shape=mat.shape)
