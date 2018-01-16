#include "eigen_sparse_wrapper.hpp"

void EigenSparseWrapper::insert(int row, int col, double value) {
	_matrix.insert(row, col) = value;
}
void EigenSparseWrapper::reserve(int nnz) {
	_matrix.reserve(nnz);
}
Eigen::SparseMatrix<double> EigenSparseWrapper::get_matrix() {
	return _matrix;
}