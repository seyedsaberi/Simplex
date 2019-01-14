import numpy as np
import rcch
import utility

if __name__ == '__main__':
    k = 3 # No. polytope vertices
    m = 2 # No. space dimensions
    n = 200 # No. samples
    sigmalist = [i/20.0 for i in range(21)] # Std. deviation of noise
    #[0,0.2,0.4,0.8] -> [(3e-8,qe-8),(3e-4,1e-4),(1e-3,1e-3),(0.005,0.005)
    lambdaList = [1e-8,1e-7,1e-6,1e-5,1e-4,2.5e-4,5e-4,7.5e-4,1e-3,1.5e-3,2e-3,2.5e-3,3e-3,3.5e-3,4e-3,4.5e-3,5e-3,6.25e-3,7.5e-3,8.75e-3,1e-2]
    for i in range(21) :
        sigma = sigmalist[i]
        lamb = lambdaList[i]
        for iii in range(10):
            rcch.SetSeed(701+iii)
            X = rcch.GenerateRandomPolytope(m=m, k=k, lo=20, hi=40)  # m * k
            Y = rcch.GenerateSampleFromPolytope(X = X, n = n, sigma = sigma) # m * n
            Xhat = rcch.LearnSimplex(
                X = X, # Set to None if ground-truth is not available.
                Y = Y,
                k = k,
                f = utility.identity, # Wrapper of l2-norm in loss function.
                df = utility.one, # Derivative of wrapper function.
                recordVideo = False,
                config = {
                    'learning_rate': 2*lamb,
                    'learning_decay': 1, # learning_rate_new = learning_rate_init * (learning_decay) ^ (#iter)
                    'lambda': lamb,
                    'pca_components': None, # Set to None if you don't need PCA.
                    'whiten': True, # Whitening data or not.
                    'gd_max_iter': 200,
                    'epsilon': 1e-6,
                    'inf': 1e20,
                    'output_dir': 'output/error/',
                    'num_runs': 1,
                    'minavg_period': 100,
                    'snapshots': [], # List of iteration for snapshots. For capturing snapshots, recordVideo should be set to True.
                    'init_type': 'random', # Choose from {'random', 'hull', 'value'}.
                                         # 'random': select (k+1) random points from the data points.
                                         # 'hull': select (k+1) random points from vertices of convex hull of data points.
                                         # 'value': initial simplex directly set by the user. In this case, 'init_value' should be set.
                    'init_value': np.asarray([[0.0, 1.0, 2.0],
                            [1.0, 2.0, 0.0]]) # An m * k matrix represents the initial simplex. This option is considered only when init_type = 'value'.
                })
