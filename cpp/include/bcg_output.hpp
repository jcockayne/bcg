#include <Eigen/Core>
#ifndef PCGOutput_H

struct BCGOutput {
	int return_code;
	int n_iter;
	Eigen::VectorXd x_m;
	Eigen::MatrixXd Sigma_F;
	double nu_m;

	// fields will only be populated if called with detailed = true
	Eigen::MatrixXd previous_x;
	Eigen::MatrixXd search_directions;
	Eigen::VectorXd search_normalisations;

	BCGOutput(int return_code, int n_iter, Eigen::VectorXd x_m, Eigen::MatrixXd Sigma_F, double nu_m) :
		return_code(return_code), n_iter(n_iter), x_m(x_m), Sigma_F(Sigma_F), nu_m(nu_m) { };
	BCGOutput() { };
};

#define PCGOutput_H
#endif