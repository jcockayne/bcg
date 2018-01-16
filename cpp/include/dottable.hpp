#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>

#include <iostream>
#ifndef DOTTABLE_H

class Dottable {
public:
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const = 0;
	virtual ~Dottable() { };
};

class DenseMatrix : public Dottable {
public:
	DenseMatrix(const Eigen::MatrixXd &matrix, bool symmetric=false) :
		_matrix(matrix), _symmetric(symmetric) { };
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const;
	virtual ~DenseMatrix() { };
private:
	const Eigen::MatrixXd _matrix;
	const bool _symmetric;
};

class SparseMatrix : public Dottable {
public:
	SparseMatrix(Eigen::SparseMatrix<double> matrix, bool symmetric=false) :
		_matrix(matrix), _symmetric(symmetric) { };
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const;
	virtual ~SparseMatrix() { };
private:
	const Eigen::SparseMatrix<double> _matrix;
	const bool _symmetric;
};

class DensePrecisionMatrix : public Dottable {
public:
	DensePrecisionMatrix(const Eigen::MatrixXd &matrix)
	{
		_solver.compute(matrix);
	};
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const;
	virtual ~DensePrecisionMatrix() { };
private:
	Eigen::PartialPivLU<Eigen::MatrixXd> _solver;
};

class SparsePrecisionMatrix : public Dottable {
public:
	SparsePrecisionMatrix(Eigen::SparseMatrix<double> matrix)
	{
		_solver.compute(matrix);
	};
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const;
	virtual ~SparsePrecisionMatrix() { };
private:
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> _solver;
};

class IncompleteLUMatrix : public Dottable {
public:
	IncompleteLUMatrix(Eigen::SparseMatrix<double> base, double drop_tol, int fill_factor, bool sym) 
	{
		_sym = sym;
		/*
		std::cout << "Sym = " << (sym ? "True" : "False") << std::endl;
		std::cout << "Drop_tol = " << drop_tol << "; Fill factor = " << fill_factor << std::endl;
		*/
		_solver.setDroptol(drop_tol);
		_solver.setFillfactor(fill_factor);
		
		_solver.compute(base);
		if(!sym) {
			_transpose_solver.setDroptol(drop_tol);
			_transpose_solver.setFillfactor(fill_factor);
			_transpose_solver.compute(base.transpose());
		}
	};
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const;
	virtual ~IncompleteLUMatrix() { };
private:
	Eigen::IncompleteLUT<double> _solver;
	Eigen::IncompleteLUT<double> _transpose_solver;
	bool _sym;
};

class IncompleteCholeskyMatrix : public Dottable {
public:
	IncompleteCholeskyMatrix(const Eigen::SparseMatrix<double> &factor) : _L(factor) { };
	virtual Eigen::VectorXd dot(const Eigen::VectorXd &rhs, bool transpose=false) const;
	virtual ~IncompleteCholeskyMatrix() { };
private:
	const Eigen::SparseMatrix<double> _L;
};

#define DOTTABLE_H
#endif