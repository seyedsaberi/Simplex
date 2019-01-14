
import numpy as np
import rcch
import utility

if __name__ == '__main__':
	k = 3 # No. polytope vertices
	m = 2 # No. space dimensions
	n = 200 # No. samples
	sigma = 0 # Std. deviation of noise
	#0 ->0
	#0.1 -> e-9
	#1 ->e-2
	rcch.SetSeed(701)
	X = rcch.GenerateRandomPolytope(m = m, k = k, lo = 20, hi = 40) # m * k
	Y = rcch.GenerateSampleFromPolytope(X = X, n = n, sigma = sigma) # m * n
	Xhat = rcch.LearnSimplex(
		X = X, # Set to None if ground-truth is not available.
		Y = Y,
		k = k,
		f = utility.identity, # Wrapper of l2-norm in loss function.
		df = utility.one, # Derivative of wrapper function.
		recordVideo = True,
		config = {
			'learning_rate': 3e-9,
			'learning_decay': 1, # learning_rate_new = learning_rate_init * (learning_decay) ^ (#iter)
			'lambda': 1e-9,
			'pca_components': None, # Set to None if you don't need PCA.
			'whiten': True, # Whitening data or not.
			'gd_max_iter': 100,
			'epsilon': 1e-6,
			'inf': 1e20,
			'output_dir': 'output/simulation/',
			'num_runs': 1,
			'minavg_period': 1,
			'snapshots': [], # List of iteration for snapshots. For capturing snapshots, recordVideo should be set to True.
			'init_type': 'random', # Choose from {'random', 'hull', 'value'}.
								 # 'random': select (k+1) random points from the data points.
								 # 'hull': select (k+1) random points from vertices of convex hull of data points.
								 # 'value': initial simplex directly set by the user. In this case, 'init_value' should be set.
			'init_value': np.asarray([[0.0, 1.0, 2.0],
					[1.0, 2.0, 0.0]]) # An m * k matrix represents the initial simplex. This option is considered only when init_type = 'value'.
		})

