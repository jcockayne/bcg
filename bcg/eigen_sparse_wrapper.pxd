from libcpp.vector cimport vector

cdef extern from "eigen_sparse_wrapper.hpp":
	cdef cppclass EigenSparseWrapper:
		EigenSparseWrapper()
		EigenSparseWrapper(int, int)
		void reserve(int)
		void insert(int, int, double)
		int rows()
		int cols()
		EigenSparseTriples to_triples()

	cdef cppclass EigenSparseTriples:
		vector[int] rows
		vector[int] cols
		vector[double] values
		EigenSparseTriples()

cdef EigenSparseWrapper make_sparse(sparse_matrix)
cdef get_sparse(EigenSparseWrapper wrapper)
