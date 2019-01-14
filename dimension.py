import numpy as np
import rcch
import utility

if __name__ == '__main__':
    sigma = 0 # Std. deviation of noise
    for k in range(8,11) :
        m = k-1  # No. space dimensions
        n = int(100*(k/3.0)**2)  # No. samples
        rcch.SetSeed(702)
        X = rcch.GenerateRandomPolytope(m=m, k=k, lo=0, hi=1)  # m * k
        Y = rcch.GenerateSampleFromPolytope(X = X, n = n, sigma = sigma) # m * n
        Xhat = rcch.LearnSimplex(
            X = X, # Set to None if ground-truth is not available.
            Y = Y,
            k = k,
            f = utility.identity, # Wrapper of l2-norm in loss function.
            df = utility.one, # Derivative of wrapper function.
            recordVideo = False,
            config = {
                'learning_rate': 2e-4,
                'learning_decay': 1, # learning_rate_new = learning_rate_init * (learning_decay) ^ (#iter)
                'lambda': 2e-4,
                'pca_components': None, # Set to None if you don't need PCA.
                'whiten': True, # Whitening data or not.
                'gd_max_iter': 200,
                'epsilon': 1e-6,
                'inf': 1e20,
                'output_dir': 'output/dimension3/',
                'num_runs': 10,
                'minavg_period': 100,
                'snapshots': [i for i in range(0,400,20)], # List of iteration for snapshots. For capturing snapshots, recordVideo should be set to True.
                'init_type': 'random', # Choose from {'random', 'hull', 'value'}.
                                     # 'random': select (k+1) random points from the data points.
                                     # 'hull': select (k+1) random points from vertices of convex hull of data points.
                                     # 'value': initial simplex directly set by the user. In this case, 'init_value' should be set.
                'init_value': np.asarray([[0.0, 1.0, 2.0],
                        [1.0, 2.0, 0.0]]) # An m * k matrix represents the initial simplex. This option is considered only when init_type = 'value'.
            })
