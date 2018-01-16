#include <Eigen/Core>
#include<Eigen/SparseCore>
#include <memory>
#include "bcg_output.hpp"

std::unique_ptr<BCGOutput> bcg_sparse(
	const Eigen::SparseMatrix<double> &A,
	const Eigen::VectorXd &b, 
	const Eigen::VectorXd &prior_mean, 
	const Eigen::SparseMatrix<double> &prior_cov,
	double eps,
	int max_iter,
	int min_iter=0,
	bool sym=false,
	bool Sigma0_is_precision=false,
	bool detailed=false,
	bool batch_directions=false
);

std::unique_ptr<BCGOutput> bcg_sparse_ichol(
	const Eigen::SparseMatrix<double> &A,
	const Eigen::VectorXd &b,
	const Eigen::VectorXd &prior_mean,
	const Eigen::SparseMatrix<double> &ichol_factor,
	double eps,
	int max_iter,
	int min_iter=0,
	bool detailed=false,
	bool batch_directions=false
);