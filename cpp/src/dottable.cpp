#include "dottable.hpp"

Eigen::VectorXd DenseMatrix::dot(const Eigen::VectorXd &rhs, bool transpose) const {
	if(transpose && !_symmetric)
		return _matrix.transpose() * rhs;
	return _matrix*rhs;
}

Eigen::VectorXd SparseMatrix::dot(const Eigen::VectorXd &rhs, bool transpose) const {
	if(transpose && !_symmetric)
		return _matrix.transpose() * rhs;
	return _matrix*rhs;
}

Eigen::VectorXd DensePrecisionMatrix::dot(const Eigen::VectorXd &rhs, bool transpose) const {
	// todo: how do I do a transpose solve in Eigen??
	if(transpose)
		return _solver.transpose().solve(rhs);
	return _solver.solve(rhs);
}

Eigen::VectorXd SparsePrecisionMatrix::dot(const Eigen::VectorXd &rhs, bool transpose) const {
	// todo: how do I do a transpose solve in Eigen??
	/*if(transpose)
		return _solver.transpose().solve(rhs);*/
	return _solver.solve(rhs);
}

Eigen::VectorXd IncompleteLUMatrix::dot(const Eigen::VectorXd &rhs, bool transpose) const {
	if(transpose)
		throw "Transpose solve not supported for IncompleteLUMatrix!";
	Eigen::VectorXd tmp = _solver.solve(rhs);
	if(_sym) {
		return _solver.solve(tmp);
	}
	else
		return _transpose_solver.solve(tmp);
}

Eigen::VectorXd IncompleteCholeskyMatrix::dot(const Eigen::VectorXd &rhs, bool transpose) const {
	// NB transpose is irrelevant here
	Eigen::VectorXd tmp = _L.triangularView<Eigen::Lower>().solve(rhs);
	_L.triangularView<Eigen::Lower>().transpose().solveInPlace(tmp);
	_L.triangularView<Eigen::Lower>().solveInPlace(tmp);
	_L.triangularView<Eigen::Lower>().transpose().solveInPlace(tmp);
	return tmp;
}