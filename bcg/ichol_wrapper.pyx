from eigen_sparse_wrapper cimport *
from scipy import sparse


cdef extern from "ichol.hpp":
    EigenSparseWrapper ichol_raw "ichol"(EigenSparseWrapper A);


cpdef ichol(A):
    """
    Compute the incomplete Cholesky decomposition of the sparse matrix A.

    Args:
        A (sparse matrix):  A sparse symmetric positive-definite matrix.

    Returns:

    """
    assert sparse.issparse(A)

    cdef EigenSparseWrapper ret = ichol_raw(make_sparse(A))
    return get_sparse(ret)