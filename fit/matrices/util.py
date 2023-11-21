from scipy.sparse import spmatrix, csr_matrix


def spdiag_pinv(mat: spmatrix) -> spmatrix:
    """Computes the pseude inverse of the given sparse diagonal matrix ``mat``."""

    mat = mat.tocoo()
    return csr_matrix((1 / mat.data, (mat.row, mat.col)), shape=mat.shape)
