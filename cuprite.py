import numpy as np
import rcch
from sklearn.decomposition import PCA
import utility

CUPRITE_PATH = 'data/cuprite/cuprite.txt'
GTRUTH = 'data/cuprite/groundTruth.txt'

SAVEVAR_PATH = 'var/'

PCA_COMPONENTS = 11

if __name__ == '__main__':
	with open(CUPRITE_PATH) as f:
		lines = f.readlines()[1:]
		Y = np.asarray([list(map(float, x.split())) for x in lines])
	with open(GTRUTH) as f:
		lines = f.readlines()[1:]
		X = np.asarray([list(map(float, x.split())) for x in lines])
	k = 3
	pca = PCA(n_components = PCA_COMPONENTS).fit(Y.T)
	YPCA = pca.transform(Y.T).T
	XPCA = pca.transform(X.T).T
  # No. polytope vertices
	#sigma = 0.1  # Std. deviation of noise
	rcch.SetSeed(701)
	Xhat = rcch.LearnSimplex(
		X=X,  # Set to None if ground-truth is not available.
		Y=Y,
		k=k,
		f=utility.identity,  # Wrapper of l2-norm in loss function.
		df=utility.one,  # Derivative of wrapper function.
		recordVideo=True,
		config={
			'learning_rate': 13e-2,
			'learning_decay': 1,  # learning_rate_new = learning_rate_init * (learning_decay) ^ (#iter)
			'lambda': 1e-2,
			'pca_components': None,  # Set to None if you don't need PCA.
			'whiten': True,  # Whitening data or not.
			'gd_max_iter': 200,
			'epsilon': 1e-3,
			'inf': 1e20,
			'output_dir': 'output/cuprite/',
			'num_runs': 10,
			'minavg_period': 1,
			'snapshots': [0, 20, 50, 100 , 150 ,199],
		# List of iteration for snapshots. For capturing snapshots, recordVideo should be set to True.
			'init_type': 'random',  # Choose from {'random', 'hull', 'value'}.
			# 'random': select (k+1) random points from the data points.
			# 'hull': select (k+1) random points from vertices of convex hull of data points.
			# 'value': initial simplex directly set by the user. In this case, 'init_value' should be set.
			'init_value': np.asarray([[0.0, 1.0, 2.0],
									  [1.0, 2.0, 0.0],
									  [0.0, 0.0, 0.0]])
		# An m * k matrix represents the initial simplex. This option is considered only when init_type = 'value'.
		})






	#bulk.Log('Start optimization...')
	#XhatPCA = bulk.Optimize(X = XPCA, Y = YPCA)
	#Xhat = pca.inverse_transform(XhatPCA.T).T
	#mmDistance, bestPerm = bulk.MinMaxDistance(X = X, Xhat = Xhat)
	#bulk.Log('Min-Max in original space: {:f}'.format(mmDistance))
	#Xhat = Xhat[:,bestPerm]
	#PhiHat = []
	#for y in Y.T:
	#	_, phi = bulk.NearestPoint(Xhat, y)
	#	PhiHat.append(phi)
	#Phi = []
	#for y in Y.T:
	#	_, phi = bulk.NearestPoint(X, y)
	#	Phi.append(phi)
	#PhiHat = np.asarray(PhiHat)
	#Phi = np.asarray(Phi)
	#np.savetxt(SAVEVAR_PATH + 'Phi.txt', Phi)
	#np.savetxt(SAVEVAR_PATH + 'PhiHat.txt', PhiHat)
	#np.savetxt(SAVEVAR_PATH + 'Xhat.txt', Xhat)
	#np.savetxt(SAVEVAR_PATH + 'X.txt', X)
	#delta = np.sum(np.abs(PhiHat - Phi)) / Phi.shape[0] / Phi.shape[1]
	#print('Delta phi: {:f}'.format(delta))
