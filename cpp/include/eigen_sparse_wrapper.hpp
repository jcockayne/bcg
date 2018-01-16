#include <Eigen/SparseCore>
#include<vector>

#ifndef EigenSparseWrapper_H

class EigenSparseTriples {
public:
	std::vector<int> rows;
	std::vector<int> cols;
	std::vector<double> values;

	EigenSparseTriples() {};
	EigenSparseTriples(Eigen::SparseMatrix<double> matrix) {
		for (int k=0; k<matrix.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(matrix,k); it; ++it)
			{
				rows.push_back(it.row());
				cols.push_back(it.col());
				values.push_back(it.value());
			}
		}
	};
};

class EigenSparseWrapper {
public:
	EigenSparseWrapper() {};
	EigenSparseWrapper(const Eigen::SparseMatrix<double> &matrix) : _matrix(matrix) { };
	EigenSparseWrapper(int nrows, int ncols) : _matrix(Eigen::SparseMatrix<double>(nrows, ncols)) { };
	void reserve(int nnz);
	void insert(int row, int col, double value);
	int rows() { return _matrix.rows(); }
	int cols() { return _matrix.cols(); }
	EigenSparseTriples to_triples() {
		return EigenSparseTriples(_matrix);
	}
	Eigen::SparseMatrix<double> get_matrix();
	operator Eigen::SparseMatrix<double>() { return _matrix; }
private:
	Eigen::SparseMatrix<double> _matrix;
};

#define EigenSparseWrapper_H
#endif