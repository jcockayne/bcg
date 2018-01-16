#include "bcg.hpp"
#include "memory_util.hpp"

std::unique_ptr<BCGOutput> bcg(
	Dottable &A, 
	const Eigen::VectorXd &b, 
	const Eigen::VectorXd &prior_mean, 
	Dottable &prior_cov,
	double eps,
	int max_iter,
	int min_iter,
	bool detailed,
	bool batch_directions
)
{
	std::vector<Eigen::VectorXd> sigma_F;

	// these will only be used if detailed
	std::vector<Eigen::VectorXd> mean_estimates;
	std::vector<Eigen::VectorXd> search_directions;
	std::vector<Eigen::VectorXd> A_cov_A_search_directions;
	std::vector<double> search_normalisations;

	Eigen::VectorXd r_m = b - A.dot(prior_mean);
	double r_m_dot_r_m = r_m.dot(r_m);
	Eigen::VectorXd s_m = r_m;
	Eigen::VectorXd x_m = prior_mean;

	double nu_m = 0;
	int m = 0;
	int d = b.rows();
	int return_code = -1;

	while(true) {
		// do required matrix-vector-multiplications
		Eigen::VectorXd tmp = A.dot(s_m, true);
		Eigen::VectorXd cov_A_s = prior_cov.dot(tmp);
		Eigen::VectorXd A_cov_A_s = A.dot(cov_A_s);

		double norm_factor_sq = 1./s_m.dot(A_cov_A_s);
		double norm_factor = sqrt(norm_factor_sq);
		double alpha_m = r_m_dot_r_m * norm_factor_sq;
		
		x_m += alpha_m * cov_A_s;
		r_m -= alpha_m * A_cov_A_s;
		if(detailed) {
			mean_estimates.push_back(x_m);
		}
		if(detailed or batch_directions) {
			search_directions.push_back(s_m);
			A_cov_A_search_directions.push_back(A_cov_A_s);
			search_normalisations.push_back(norm_factor_sq);
		}
		
		nu_m += r_m_dot_r_m * r_m_dot_r_m * norm_factor_sq;
		double sigma_m = sqrt((d-1-m)*nu_m * 1./ (m+1));

		double prev_r_m_dot_r_m = r_m_dot_r_m;
		r_m_dot_r_m = r_m.dot(r_m);

		sigma_F.push_back(cov_A_s * norm_factor);


		// update state
		++m;
		if(batch_directions) {
			s_m = r_m;
			for(int i = 0; i < m; i++) {
				double coeff = r_m.dot(A_cov_A_search_directions[i])*search_normalisations[i];
				s_m -= coeff*search_directions[i];
			}
		}
		else {
			double beta_m = r_m_dot_r_m / prev_r_m_dot_r_m;
			s_m = r_m + beta_m *s_m;
		}
		
		#ifndef TERMINATE_CLASSIC
			// termination criteria from the paper
			if(m >= min_iter and sigma_m < eps)
		#else
			// a more traditional residual-minimising strategy
			if(m >= min_iter and sqrt(r_m_dot_r_m) < eps) 
		#endif
		{
			return_code = 0;
			break;
		}

		if(m == max_iter or m == d) {
			return_code = 1;
			break;
		}
	}

	Eigen::MatrixXd sigma_F_mat(d, m);
	// form the return object
	for(int i = 0; i < m; i++) {
		sigma_F_mat.col(i) = sigma_F.at(i);

	}
	auto ret = make_unique<BCGOutput>(return_code, m, x_m, sigma_F_mat, nu_m * 1./m);
	if(detailed) {
		Eigen::MatrixXd mean_estimates_mat(d, m);
		Eigen::MatrixXd search_directions_mat(d, m);
		Eigen::VectorXd search_normalisations_mat(m);
		for(int i = 0; i < m; i++) {
			mean_estimates_mat.col(i) = mean_estimates.at(i);
			search_directions_mat.col(i) = search_directions.at(i);
			search_normalisations_mat(i) = search_normalisations.at(i);
		}
		ret->previous_x = mean_estimates_mat;
		ret->search_directions = search_directions_mat;
		ret->search_normalisations = search_normalisations_mat;
	}
	return ret;
}