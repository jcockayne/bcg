import itertools
import numpy as np
cimport numpy as np
from eigency.core cimport *
import cython
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from eigen_sparse_wrapper cimport *
from scipy import sparse

cdef extern from "bcg_output.hpp":
	cdef cppclass BCGOutput:
		int return_code
		int n_iter
		VectorXd x_m 
		MatrixXd Sigma_F
		double nu_m

		MatrixXd search_directions
		MatrixXd previous_x
		VectorXd search_normalisations

cdef extern from "bcg_sparse.hpp":
	cdef unique_ptr[BCGOutput] bcg_sparse_raw "bcg_sparse"(
		EigenSparseWrapper A,
		Map[VectorXd] b, 
		Map[VectorXd] prior_mean, 
		EigenSparseWrapper prior_cov,
		double eps,
		int max_iter,
		int min_iter,
		bint sym,
		bint Sigma0_is_precision,
		bint detailed,
		bint batch_directions
	);
	cdef unique_ptr[BCGOutput] bcg_sparse_ichol(
		EigenSparseWrapper A,
		Map[VectorXd] b,
		Map[VectorXd] prior_mean,
		EigenSparseWrapper ichol_factor,
		double eps,
		int max_iter,
		int min_iter,
		bint detailed,
		bint batch_directions
	)

cdef extern from "bcg_dense.hpp":
	cdef unique_ptr[BCGOutput] bcg_dense_raw "bcg_dense"(
		Map[MatrixXd] A,
		Map[VectorXd] b, 
		Map[VectorXd] prior_mean, 
		Map[MatrixXd] prior_cov,
		double eps,
		int max_iter,
		int min_iter,
		bint sym,
		bint Sigma0_is_precision,
		bint detailed,
		bint batch_directions
	);

class PythonBCGOutput:
	"""
	Wrapper for the output of a BCG run.
	"""
	def __init__(self, x_m, Sigma_F, nu_m, n_iter, return_code):
		self.__x_m__ = x_m
		self.__nu_m__ = nu_m
		self.__Sigma_F__ = Sigma_F
		self.__n_iter__ = n_iter
		self.__return_code__ = return_code
		self.means = None
		self.search_directions = None
		self.search_normalisations = None

	@property
	def x_m(self):
		"""
		The posterior mean after m iterations
		"""
		return self.__x_m__

	@property
	def Sigma_F(self):
		"""
		The posterior covariance factors. The posterior covariance is then computed as
		Sigma_m = Sigma_0 - Sigma_F.dot(Sigma_F.T)
		"""
		return self.__Sigma_F__

	@property 
	def nu_m(self):
		"""
		The normalising factor for the Student's-T posterior.
		"""
		return self.__nu_m__

	@property
	def n_iter(self):
		"""
		The number of iterations performed
		"""
		return self.__n_iter__

	@property
	def return_code(self):
		"""
		The return code from C++. 0 indicates a successful termination. 1 indicates that the
		maximal number of iterations was reached.
		"""
		return self.__return_code__

	@property
	def S_m_normalised(self):
		"""
		The normalised search directions. This will only be populated if the code is run with
		detailed=True.
		"""
		return self.search_directions * np.sqrt(self.search_normalisations.ravel()[None,:])

	@property
	def previous_x(self):
		"""
		Returns the posterior means from all previous iterations. This will only be populated
		if the code is run with detailed=True.
		"""
		return self.means

cdef make_output(BCGOutput *retptr, bint detailed):
	"""
	Converts a C++ BCGOutput to a PythonBCGOutput object so that it can be returned to Python code.
	"""
	cdef BCGOutput ret = deref(retptr)
	out = PythonBCGOutput(
		ndarray_copy(ret.x_m),
		ndarray_copy(ret.Sigma_F),
		ret.nu_m,
		ret.n_iter,
		ret.return_code
	)
	if detailed:
		out.means = ndarray_copy(ret.previous_x)
		out.search_directions = ndarray_copy(ret.search_directions)
		out.search_normalisations = ndarray_copy(ret.search_normalisations)
	return out

@cython.embedsignature
def bcg_dense(
	np.ndarray[ndim=2, dtype=np.float_t] A,
	np.ndarray[ndim=1, dtype=np.float_t] b,
	np.ndarray[ndim=1, dtype=np.float_t] prior_mean,
	np.ndarray[ndim=2, dtype=np.float_t] prior_cov,
	double eps,
	max_iter=None,
	min_iter=None,
	bint sym=False,
	bint prior_cov_is_precision=False,
	bint detailed=False,
	bint batch_directions=False
):
	"""
	Solve the linear system Ax = b using the probabilistic conjugate gradient method.
	This assumes both A and prior_cov are dense matrices (np.ndarray).

	Args:
		A (numpy matrix):		 	Sparse matrix describing the linear system.
		b (numpy array):			Right-hand-side of the system
		prior_mean (numpy array):	Prior mean vector
		prior_cov (numpy matrix):	Prior covariance matrix
		eps (double):				Tolerance for the solver
		max_iter (int):				Maximal iteration. Defaults to A.shape[0].
		min_iter (int):				Minimum number of iterations. Defaults to 0.1*A.shape[0]
		sym (bool):					Whether A is symmetric. The solution can be computed more efficiently if A is symmetric 
									as a sparse transpose is avoided. Default is False.
		prior_cov_is_precision (bool):	Whether the matrix prior_cov is a covariance matrix (False) or a preicision matrix
										(True). Default is False.
		detailed (bool):			Whether to return detailed output. If so, the posterior means and search directions will
									be recorded at each iteration.
		batch_directions (bool):	Whether to use batch computed search directions (True) or sequentially computed (False)

	Returns:
		A PythonBCGOutput object containing the result.
	"""
	assert not sparse.issparse(A)
	assert not sparse.issparse(prior_cov)
	assert not sparse.issparse(b)
	assert not sparse.issparse(prior_mean)

	if min_iter is None:
		min_iter = int(A.shape[0]*0.1)
	if max_iter is None:
		max_iter = A.shape[0]

	A = np.asfortranarray(A)
	prior_cov = np.asfortranarray(prior_cov)

	#b = np.asfortranarray(b)
	#prior_mean = np.asfortranarray(prior_mean)

	cdef unique_ptr[BCGOutput] retptr = bcg_dense_raw(
		Map[MatrixXd](A), 
		Map[VectorXd](b), 
		Map[VectorXd](prior_mean), 
		Map[MatrixXd](prior_cov), 
		eps, 
		max_iter, 
		min_iter,
		sym,
		prior_cov_is_precision,
		detailed,
		batch_directions
	)
	return make_output(retptr.get(), detailed)

@cython.embedsignature
def bcg_sparse(
	A,
	np.ndarray[ndim=1, dtype=np.float_t] b,
	np.ndarray[ndim=1, dtype=np.float_t] prior_mean,
	prior_cov,
	double eps,
	max_iter=None,
	min_iter=None,
	bint sym=False,
	bint prior_cov_is_precision=False,
	bint detailed=False,
	bint batch_directions=False
):
	"""
	Solve the linear system Ax = b using the probabilistic conjugate gradient method.
	This assumes both A and prior_cov are scipy sparse matrices.

	Args:
		A (sparse matrix):		 	Sparse matrix describing the linear system.
		b (numpy array):			Right-hand-side of the system
		prior_mean (numpy array):	Prior mean vector
		prior_cov (sparse matrix):	Prior covariance matrix
		eps (double):				Tolerance for the solver
		max_iter (int):				Maximal iteration. Defaults to A.shape[0].
		min_iter (int):				Minimum number of iterations. Defaults to 0.1*A.shape[0]
		sym (bool):					Whether A is symmetric. The solution can be computed more efficiently if A is symmetric 
									as a sparse transpose is avoided. Default is False.
		prior_cov_is_precision (bool):	Whether the matrix prior_cov is a covariance matrix (False) or a preicision matrix
										(True). Default is False.
		detailed (bool):			Whether to return detailed output. If so, the posterior means and search directions will
									be recorded at each iteration.
		batch_directions (bool):	Whether to use batch computed search directions (True) or sequentially computed (False)

	Returns:
		A PythonBCGOutput object containing the result.
	"""
	assert sparse.issparse(A)
	assert sparse.issparse(prior_cov)
	assert not sparse.issparse(b)
	assert not sparse.issparse(prior_mean)

	if min_iter is None:
		min_iter = int(A.shape[0]*0.1)
	if max_iter is None:
		max_iter = A.shape[0]

	cdef EigenSparseWrapper A_sp = make_sparse(A)
	cdef EigenSparseWrapper prior_cov_sp = make_sparse(prior_cov)

	#b = np.asfortranarray(b)
	#prior_mean = np.asfortranarray(prior_mean)

	cdef unique_ptr[BCGOutput] retptr = bcg_sparse_raw(
		A_sp, 
		Map[VectorXd](b), 
		Map[VectorXd](prior_mean), 
		prior_cov_sp, 
		eps, 
		max_iter, 
		min_iter,
		sym,
		prior_cov_is_precision,
		detailed,
		batch_directions
	)
	return make_output(retptr.get(), detailed)

@cython.embedsignature
def bcg_preconditioned_ichol(
	A,
	np.ndarray[ndim=1, dtype=np.float_t] b,
	np.ndarray[ndim=1, dtype=np.float_t] prior_mean,
	ichol_factor,
	double eps,
	max_iter=None,
	min_iter=None,
	bint detailed=False,
	bint batch_directions=False
):
	"""
	Solve the linear system Ax = b using the probabilistic conjugate gradient method.
	This assumes both A and prior_cov are scipy sparse matrices.

	Args:
		A (sparse matrix):		 	Sparse matrix describing the linear system.
		b (numpy array):			Right-hand-side of the system
		prior_mean (numpy array):	Prior mean vector
		ichol_factor (sparse matrix):	Incomplete Cholesky factor of A
		eps (double):				Tolerance for the solver
		max_iter (int):				Maximal iteration. Defaults to A.shape[0].
		min_iter (int):				Minimum number of iterations. Defaults to 0.1*A.shape[0]
		detailed (bool):			Whether to return detailed output. If so, the posterior means and search directions will
									be recorded at each iteration.
		batch_directions (bool):	Whether to use batch computed search directions (True) or sequentially computed (False)

	Returns:
		A PythonBCGOutput object containing the result.
	"""
	
	assert sparse.issparse(A)
	assert sparse.issparse(ichol_factor)
	assert not sparse.issparse(b)
	assert not sparse.issparse(prior_mean)

	if min_iter is None:
		min_iter = int(A.shape[0]*0.1)
	if max_iter is None:
		max_iter = A.shape[0]

	cdef EigenSparseWrapper A_sp = make_sparse(A)
	cdef EigenSparseWrapper ichol_factor_sp = make_sparse(ichol_factor)
	cdef unique_ptr[BCGOutput] retptr = bcg_sparse_ichol(
		A_sp,
		Map[VectorXd](b),
		Map[VectorXd](prior_mean),
		ichol_factor_sp,
		eps,
		max_iter,
		min_iter,
		detailed,
		batch_directions
	)

	return make_output(retptr.get(), detailed)