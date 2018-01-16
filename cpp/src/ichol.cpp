#include <Eigen/SparseCore>

Eigen::SparseMatrix<double> ichol(const Eigen::SparseMatrix<double> &A) {
	int n = A.rows();

	Eigen::SparseMatrix<double> L(n,n);
	L.reserve(A.nonZeros());

	for(int i = 0; i < n; i++) {
		double sqrt_diag = A.coeff(i,i);
		for(int k = 0; k < i; k++) {
			double tmp = L.coeff(i,k);
			sqrt_diag -= tmp*tmp;
		}
		sqrt_diag = sqrt(sqrt_diag);
		L.insert(i,i) = sqrt_diag;
		sqrt_diag = 1./sqrt_diag;
		for(int j = i+1; j < n; j++) {
			double tmp = A.coeff(j,i);
			if(tmp == 0) continue;

			for(int k = 0; k < i; k++) {
				tmp -= L.coeff(i,k)*L.coeff(j,k);
			}
			tmp *= sqrt_diag;
			L.insert(j,i) = tmp;
		}
	}
	return L;
}