#include "bcg_sparse.hpp"
#include "bcg.hpp"
#include <vector>
#include <iostream>
#include "ichol.hpp"
#include "memory_util.hpp"

std::unique_ptr<BCGOutput> bcg_sparse(
	const Eigen::SparseMatrix<double> &A,
	const Eigen::VectorXd &b, 
	const Eigen::VectorXd &prior_mean, 
	const Eigen::SparseMatrix<double> &prior_cov,
	double eps,
	int max_iter,
	int min_iter,
	bool sym,
	bool Sigma0_is_precision,
	bool detailed,
	bool batch_directions
)
{
	std::unique_ptr<Dottable> A_dot = make_unique<SparseMatrix>(A, sym);
	std::unique_ptr<Dottable> Sigma0_dot;
	if(Sigma0_is_precision)
		Sigma0_dot = make_unique<SparsePrecisionMatrix>(prior_cov);
	else
		Sigma0_dot = make_unique<SparseMatrix>(prior_cov, true);
	
	auto ret = bcg(
		*A_dot, 
		b, 
		prior_mean, 
		*Sigma0_dot, 
		eps, 
		max_iter, 
		min_iter,
		detailed,
		batch_directions
	);

	return ret;
}

std::unique_ptr<BCGOutput> bcg_sparse_ichol(
	const Eigen::SparseMatrix<double> &A,
	const Eigen::VectorXd &b,
	const Eigen::VectorXd &prior_mean,
	const Eigen::SparseMatrix<double> &ichol_factor,
	double eps,
	int max_iter,
	int min_iter,
	bool detailed,
	bool batch_directions
)
{
	std::unique_ptr<Dottable> A_dot = make_unique<SparseMatrix>(A, true);
	std::unique_ptr<Dottable> Sigma0_dot = make_unique<IncompleteCholeskyMatrix>(ichol_factor);
	
	auto ret = bcg(
		*A_dot, 
		b, 
		prior_mean, 
		*Sigma0_dot, 
		eps, 
		max_iter, 
		min_iter,
		detailed,
		batch_directions
	);

	return ret;
}