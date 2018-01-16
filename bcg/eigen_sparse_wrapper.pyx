
import scipy.sparse

cdef EigenSparseWrapper make_sparse(sparse_matrix):
	"""
	Create an Eigen sparse matrix from a scipy sparse matrix.
	"""
	cx = sparse_matrix.tocoo()    
	cdef EigenSparseWrapper ctor = EigenSparseWrapper(sparse_matrix.shape[0], sparse_matrix.shape[1])
	cdef int i,j
	cdef double v
	for i,j,v in zip(cx.row, cx.col, cx.data):
		ctor.insert(i, j, v)
	return ctor

cdef get_sparse(EigenSparseWrapper sparse_wrapper):
	"""
	Get a scipy sparse matrix from an Eigen sparse matrix.
	"""
	cdef EigenSparseTriples triples = sparse_wrapper.to_triples()
	return scipy.sparse.coo_matrix((triples.values, (triples.rows, triples.cols)), shape=(sparse_wrapper.rows(), sparse_wrapper.cols()))
