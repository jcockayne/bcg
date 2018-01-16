#include <Eigen/Core>
#include <memory>
#include "bcg_output.hpp"
#include "dottable.hpp"

std::unique_ptr<BCGOutput> bcg(
	Dottable &A, 
	const Eigen::VectorXd &b, 
	const Eigen::VectorXd &prior_mean, 
	Dottable &prior_cov,
	double eps,
	int max_iter,
	int min_iter=0,
	bool detailed=false,
	bool batch_directions=false
);