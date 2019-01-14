import os
import sys
import itertools
import numpy as np
import cvxopt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import scipy
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import networkx as nx
from time import gmtime, strftime

epsilon = 1e-6
inf = 1e20 #
outputDir = 'output/' # Output directory.
alpha = 1e-2 # Learning rate
learningDecay = 0.99 # Learning decay
mLambda = 1 # Regularization coefficient.
pcaComponents = None # number of pca components
whiten = False # Whitening data or not.
maxIter = 1000 # No. iterations in gradient descent
numRuns = 1 # No. Runs
minAvgPeriod = 10 # Period for calculating min-avg distance.
initType = 'random' # Type of initializartion from {'random', 'hull', 'value'}
initValue = None # An m * k matrix represents the initial simplex.
snapshots = [] # List of iteration for snapshots.

logFile = None
startTime = None

def Log(msg):
	global logFile
	print(msg)
	sys.stdout.flush()
	if logFile is not None:
		logFile.write(msg + '\n')
	else:
		print('>>>>>>>>> log is not writing in the file [logFile is None].')

def SetSeed(seed):
	np.random.seed(seed = seed)

def GenerateRandomPolytope(m, k, lo, hi):
	return np.random.uniform(low = lo, high = hi, size = (m, k))

def GenerateRandomConvexComb(k, n):
	combs = np.empty([k, n], dtype=float)
	for i in range(n):
		X = np.empty([1, k + 1], dtype=float)
		X[0,0] = 0
		X[0,1:k] = np.random.uniform(0, 1,k-1)[0:k-1]
		X[0,k] = 1
		X = np.sort(X,axis=1)
		for j in range(1,k+1):
			combs[j-1,i] = X[0,j] - X[0,j-1]
	combs = np.random.dirichlet([1 for i in range(k)], n).T
	return combs

def GenerateSampleFromPolytope(X, n, sigma):
	return np.matmul(X, GenerateRandomConvexComb(X.shape[1], n)) \
		+ (0 if sigma is None else np.random.normal(loc = 0.0, scale = sigma, size = (X.shape[0], n)))

def NearestPoint(X, y):
	P = cvxopt.matrix(2 * np.matmul(X.T, X))
	q = cvxopt.matrix(-2 * np.matmul(y.copy(), X))
	G = cvxopt.matrix(-np.eye(X.shape[1]))
	h = cvxopt.matrix(np.zeros((X.shape[1], 1)))
	A = cvxopt.matrix(np.ones((1, X.shape[1])))
	b = cvxopt.matrix(np.ones((1, 1)))
	sol = cvxopt.solvers.qp(P = P, q = q, G = G, h = h, A = A, b = b)
	phi = np.asarray(sol['x']).reshape((X.shape[1],))
	return np.matmul(X, phi), phi

def AffToSub(X):
	return (X.T - X[:,0])[1:].T

def Likelihood(X, Y, f, df):
	loss = 0
	grad = np.zeros(X.shape);
	for i in range(Y.shape[1]):
		proj, phiOpt = NearestPoint(X = X, y = Y[:,i])
		error = np.sum((proj - Y[:,i]) ** 2)
		loss += f(error)
		S = []
		for index, p in enumerate(phiOpt):
			if p > epsilon:
				S.append(index)
		phiOptS = phiOpt[S[1:]]
		XS = X[:,S]
		gradS = 2 * np.matmul(XS, np.outer(phiOpt[S], phiOpt[S])) - 2 * np.outer(Y[:,i], phiOpt[S])
		for index, g in zip(S, gradS.T):
			grad[:,index] += g * df(error)
	factor = Y.shape[1] * mLambda
	loss /= factor
	grad /= factor
	return loss, grad

def Vol(X):
	XTilde = AffToSub(X)
	vol = np.sqrt(np.linalg.det(np.matmul(XTilde.T, XTilde))) / (np.math.factorial(X.shape[1] - 1))
	gradTilde = np.matmul(XTilde, np.linalg.inv(np.matmul(XTilde.T, XTilde)))
	grad = np.column_stack([-np.sum(gradTilde, axis = 1), gradTilde])
	return vol, vol * grad

def GetLoss(X, Y, f, df):
	lossLikelihood, gradLikelihood = Likelihood(X = X, Y = Y, f = f, df = df)
	lossVol, gradVol = Vol(X)
	return lossLikelihood + lossVol, gradLikelihood + gradVol

def Initialize(Y, k, t):
	if t == 'hull':
		hull = (Y.T[ConvexHull(Y.T).vertices]).T
		indicator = list(range(hull.shape[1]))
		np.random.shuffle(indicator)
		indicator = indicator[0:k]
		return hull[:,indicator]
	elif t == 'random':
		indicator = list(range(Y.shape[1]))
		np.random.shuffle(indicator)
		indicator = indicator[0:k]
		return Y[:,indicator]
	else:
		Log('type \'{:s}\' is not supported in Initialize'.format(t))
		exit(1)
	
def MinMaxDistance(X, Xhat):
	if X is None:
		return inf
	mmDistance = inf
	bestPerm = None
	for perm in itertools.permutations(list(range(X.shape[1]))):
		distance = 0.0
		for i in range(X.shape[1]):
			distance = max(distance, np.linalg.norm(X[:,i] - Xhat[:,perm[i]]))
		if distance < mmDistance:
			mmDistance = distance
			bestPerm = perm
	return mmDistance, bestPerm

def MinAvgDistance(X, Xhat):
	if X is None:
		return inf, []
	maDistance = inf
	bestPerm = None
	for perm in itertools.permutations(list(range(X.shape[1]))):
		distance = 0.0
		for i in range(X.shape[1]):
			distance += np.linalg.norm(X[:,i] - Xhat[:,perm[i]])
		distance /= (X.shape[1] + 0.0)
		if distance < maDistance:
			maDistance = distance
			bestPerm = perm
	return maDistance, bestPerm

def MinAvgDistanceFast(X, Xhat):
	if X is None:
		return inf, []
	G = nx.Graph()
	maxw = 0
	for i in range(X.shape[1]):
		for j in range(Xhat.shape[1]):
			maxw = max(maxw, np.linalg.norm(X[:,i] - Xhat[:, j]))
	maxw = maxw + 1
	for i in range(X.shape[1]):
		for j in range(Xhat.shape[1]):
			G.add_edge('a_{:d}'.format(i), 'b_{:d}'.format(j), weight = maxw - np.linalg.norm(X[:,i] - Xhat[:, j]))
	matched = nx.max_weight_matching(G, maxcardinality = True)
	total = 0.0
	perm = []
	for i in range(X.shape[1]):
		u = 'a_{:d}'.format(i)
		v = matched[u]
		total += -G[u][v]['weight'] + maxw
		perm.append(int(v[2:]))
	total = total / (X.shape[1] + 0.0)
	return total, perm

def MaxCorr(X, Xhat):
	if X is None:
		return inf, []
	G = nx.Graph()
	for i in range(X.shape[1]):
		for j in range(Xhat.shape[1]):
			u = X[:, i] / np.linalg.norm(X[:, i])
			v = Xhat[:, j] / np.linalg.norm(Xhat[:, j])
			G.add_edge('a_{:d}'.format(i), 'b_{:d}'.format(j), weight = np.dot(u, v))
	matched = nx.max_weight_matching(G, maxcardinality = True)
	total = 0.0
	perm = []
	vals = []
	for i in range(X.shape[1]):
		u = 'a_{:d}'.format(i)
		if u not in matched:
			continue
		v = matched[u]
		total += G[u][v]['weight']
		vals.append(G[u][v]['weight'])
		perm.append(int(v[2:]))
	vals.sort()
	total = total / (X.shape[1] + 0.0)
	return total, perm
	
def Optimize(X, Y, k, f, df, recordVideo = True):
	if X is not None:
		Log('X.shape: ' + str(X.shape))
		volume, _ = Vol(X)
		Log('Volume X: {:f}'.format(volume))
		XTilde = AffToSub(X)
		M = np.matmul(XTilde.T, XTilde)
		w = scipy.linalg.eigh(M, eigvals_only = True)
		w.sort()
		Log('Min eigenvalue X: {:f}'.format(np.sqrt(w[0])))
		Log('Max eigenvalue X: {:f}'.format(np.sqrt(w[-1])))
		Log('Condition number X: {:f}'.format(np.sqrt(w[-1]/w[0])))
	Log('Y.shape: ' + str(Y.shape))
	m = Y.shape[0]
	n = Y.shape[1]
	if X is not None:
		minGTDistance = inf
		for i in range(k):
			for j in range(i+1, k):
				minGTDistance = min(minGTDistance, np.linalg.norm(X[:,i] - X[:,j]))
		minAvgSample = 0.0
		for i in range(k):
			minDistance = inf
			for j in range(n):
				minDistance = min(minDistance, np.linalg.norm(X[:,i] - Y[:,j]))
			minAvgSample += minDistance
		minAvgSample /= k
		Log('min-distance in ground-truth: {:f}'.format(minGTDistance))
		Log('min-average distance in sample: {:f}'.format(minAvgSample))
	Xhat = initValue if initType == 'value' else Initialize(Y = Y, k = k, t = initType)
	if X is not None and X.shape[1] == k:
		gtLoss, _ = GetLoss(X = X, Y = Y, f = f, df = df)
	else:
		gtLoss = inf
	if recordVideo:
		FFMpegWriter = manimation.writers['ffmpeg']
		fig = plt.figure()
		writer = FFMpegWriter(fps = 10)
		plotTitle = 'n = {:d}, lambda = {:.4f}, alpha = {:.4f}'.format(n, mLambda, alpha)
		writer.setup(fig, outputDir + 'rcch-{:s}.mp4'.format(startTimeId), 100)
	
	xlim, ylim = (1, -1), (1, -1)
	outputs = [Xhat]
	for it in range(-1, maxIter):
		loss, grad = GetLoss(X = Xhat, Y = Y, f = f, df = df)
		if it >= 0:
			Xhat -= alpha * (learningDecay ** it) * grad
		outputs.append(Xhat)
		#Log('(estimate loss, ground-truth loss) after iteration {:d}: {:f}, {:f}'.format(it + 1, loss, gtLoss))
		if (it + 1) % minAvgPeriod == 0 or (it + 1 == maxIter):
			Log('{:f}'.format( MinAvgDistanceFast(X, Xhat)[0]))
#			Log('Max-Corr distance after iteration {:d}: {:f}'.format(it + 1, MaxCorr(X, Xhat)[0]))
		if recordVideo and ((it + 1) % 5 == 0 or (it + 1) in snapshots):
			XBase = X if X is not None else Xhat
			origin = XBase[:,0]
			basis = scipy.linalg.orth(AffToSub(XBase)).T
			plt.title(plotTitle + ' ' + '[iter: {:d}]'.format(it + 1))
			if X is not None:
				XP = np.matmul(basis, (X.T - origin).T)
				plt.plot(np.concatenate([XP[0,:],XP[0,0:1]]), np.concatenate([XP[1,:], XP[1,0:1]]), color = 'green')
			XhatP = np.matmul(basis, (Xhat.T - origin).T)
			plt.plot(np.concatenate([XhatP[0,:], XhatP[0,0:1]]), np.concatenate([XhatP[1,:], XhatP[1,0:1]]), color = 'red')
			YP = np.matmul(basis, (Y.T - origin).T)
			plt.scatter(YP[0,:], YP[1,:])
			minx, maxx = min(min(XhatP[0,:]), min(YP[0,:])), max(max(XhatP[0,:]), max(YP[0,:]))
			miny, maxy = min(min(XhatP[1,:]), min(YP[1,:])), max(max(XhatP[1,:]), max(YP[1,:]))
			if X is not None:
				minx, maxx = min(minx, min(XP[0,:])), max(maxx, max(XP[0,:]))
				miny, maxy = min(miny, min(XP[1,:])), max(maxy, max(XP[1,:]))
			if minx < xlim[0] + epsilon or maxx > xlim[1] + epsilon:
				minx = min(minx, xlim[0])
				maxx = max(maxx, xlim[1])
				padding = (maxx - minx) * 0.1
				xlim = (minx - padding, maxx + padding)
			if miny < ylim[0] + epsilon or maxy > ylim[1] + epsilon:
				miny = min(miny, ylim[0])
				maxy = max(maxy, ylim[1])
				padding = (maxy - miny) * 0.1
				ylim = (miny - padding, maxy + padding)
			plt.xlim(xlim[0], xlim[1])
			plt.ylim(ylim[0], ylim[1])
			writer.grab_frame()
			if (it + 1) in snapshots:
				plt.savefig(outputDir + 'rcch-{:s}-snapshot-iter{:d}.png'.format(startTimeId, it + 1))
			plt.clf()
	return outputs

def LearnSimplex(X, Y, k, f, df, recordVideo, config):
	global alpha, learningDecay, pcaComponents, whiten, maxIter, numRuns, epsilon, inf
	global outputDir, mLambda, minAvgPeriod, initType, initValue, snapshots
	if 'learning_rate' in config:
		alpha = config['learning_rate']
	if 'learning_decay' in config:
		learningDecay = config['learning_decay']
	if 'lambda' in config:
		mLambda = config['lambda']
	if 'pca_components' in config:
		pcaComponents = config['pca_components']
	if 'whiten' in config:
		whiten = config['whiten']
	if 'gd_max_iter' in config:
		maxIter = config['gd_max_iter']
	if 'num_runs' in config:
		numRuns = config['num_runs']
	if 'epsilon' in config:
		epsilon = config['epsilon']
	if 'inf' in config:
		inf = config['inf']
	if 'output_dir' in config:
		outputDir = config['output_dir']
	if 'minavg_period' in config:
		minAvgPeriod = config['minavg_period']
	if 'init_type' in config:
		initType = config['init_type']
		if initType == 'value':
			initValue = config['init_value']
	if 'snapshots' in config:
		snapshots = config['snapshots']
	
	global startTimeId, logFile
	startTimeId = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

	cvxopt.solvers.options['show_progress'] = False
	cvxopt.solvers.options['abstol'] = epsilon/10.0
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
	XReduced, YReduced = X.copy() if X is not None else None, Y.copy()
	if pcaComponents is None:
		pcaComponents = min(Y.shape[0], Y.shape[1])
	pca = PCA(n_components = pcaComponents, whiten = whiten).fit(Y.T)
	YReduced = pca.transform(Y.T).T
	if X is not None:
		XReduced = pca.transform(X.T).T
	outputs = []
	startTimeIdOld = startTimeId
	for run in range(numRuns):
		startTimeId = startTimeIdOld + '-run{:d}'.format(run)
		logFile = open(outputDir + '/rcch-{:s}.log'.format(startTimeId), 'w')
		XhatReducedList = Optimize(X = XReduced, Y = YReduced, k = k, f = f, df = df, recordVideo = recordVideo)
		XhatList = [pca.inverse_transform(XhatReduced.T).T for XhatReduced in XhatReducedList]
		outputs.append(XhatList)
	return outputs
