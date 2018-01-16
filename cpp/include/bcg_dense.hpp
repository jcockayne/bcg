#include <Eigen/Core>
#include <memory>
#include "bcg_output.hpp"

std::unique_ptr<BCGOutput> bcg_dense(
	const Eigen::MatrixXd &A,
	const Eigen::VectorXd &b, 
	const Eigen::VectorXd &prior_mean, 
	const Eigen::MatrixXd &prior_cov,
	double eps,
	int max_iter,
	int min_iter=0,
	bool sym=false,
	bool Sigma0_is_precision=false,
	bool detailed=false,
	bool batch_directions=false
);