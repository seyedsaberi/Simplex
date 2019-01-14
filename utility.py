import numpy as np

def identity(x):
	return x

def zero(x):
	return 0

def one(x):
	return 1

def tanh(x):
	return np.tanh(x)

def dtanh(x):
	return 1 - (tanh(x) ** 2)
