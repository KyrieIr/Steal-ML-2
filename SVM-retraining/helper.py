import numpy as np

"""
This module contains some general python helper functions.
"""


def UniformPoints(N, boundaries):
	"""
	Function that returns N points uniformly distributed over a space 
	specified by the boundaries (2-Dimensional)

	@parameters: N: Number of points
	@parameters: boundaries: an array of 2-element arrays indicating the 
	"""
	number = int(np.floor(np.sqrt(N-1))) # I do minus 1 so there is at least 1 point I can set to the middle of the domain
	output = []
	dim1 = boundaries[:][0]
	dim2 = boundaries[:][1]
	l1 = dim1[1] - dim1[0]
	l2 = dim2[1] - dim2[0]
	basicint1 = np.linspace(dim1[0],dim1[1],num=number, endpoint=False) + l1/(2*number)
	basicint2 = np.linspace(dim2[0],dim2[1],num=number, endpoint=False) + l2/(2*number)
	for i in range(0,number):
		for j in range(0,number):
			output.append([basicint1[i],basicint2[j]])

	output.append([0,0])
	i = 0
	while (number**2 + 1 + i<N):
		rand1 = np.random.uniform(dim1[0], dim1[1])
		rand2 = np.random.uniform(dim2[0], dim2[1])
		output.append([rand1,rand2])
		i += 1
	return output

def rv_bi(size):
		a = 2 * np.random.randint(2, size=(size,)) - 1
		return a

def rv_norm(size):
	if spec is not None and spec.type == 'norm':
		assert len(spec.mean) == self.n_features
		r = np.zeros(size)
		for i in range(0, size):
			r[i] = np.random.normal(loc=spec.mean[i])
		return r
	else:
		return np.random.normal(loc=mean, size=size)

def rv_uni(low,high,size):
		return np.random.uniform(low, high, size)