#include "bcg.hpp"
#include "bcg_dense.hpp"
#include "memory_util.hpp"

std::unique_ptr<BCGOutput> bcg_dense(
	const Eigen::MatrixXd &A,
	const Eigen::VectorXd &b, 
	const Eigen::VectorXd &prior_mean, 
	const Eigen::MatrixXd &prior_cov,
	double eps,
	int max_iter,
	int min_iter,
	bool sym,
	bool Sigma0_is_precision,
	bool detailed,
	bool batch_directions
)
{
	std::unique_ptr<Dottable> A_dot = make_unique<DenseMatrix>(A, sym);
	std::unique_ptr<Dottable> Sigma0_dot;
	if(Sigma0_is_precision)
		Sigma0_dot = make_unique<DensePrecisionMatrix>(prior_cov);
	else
		Sigma0_dot = make_unique<DenseMatrix>(prior_cov, true);
	
	auto ret = bcg(
		*A_dot, 
		b, 
		prior_mean, 
		*Sigma0_dot, 
		eps, 
		max_iter, 
		min_iter,
		detailed,
		batch_directions);

	return ret;
}