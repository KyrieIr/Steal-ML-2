import numpy as np
import matplotlib.pyplot as plt


"""
This module contains some general python helper functions.
"""

def UniformPoints(N, n_features):
	"""
	Function that returns N points uniformly distributed over a space 
	specified by the boundaries (2-Dimensional)

	@parameters: N: Number of points
	@parameters: boundaries: an array of 2-element arrays indicating the 
	"""
	output = []
	boundaries = [-1, 1]
	i = 0

	while (i<N):
		output.append(rv_uni(boundaries[0],boundaries[1],n_features))
		i += 1
	return np.array(output)

def ToySet(n_samples = 100, n_features= 2, factor=0.5, offset = 0):
	"""
	Function that returns a toy dataset with factor*N features having Y==1 
	and (1-factor)*N features having Y==0

	@parameters: N, number of points
	@parameters: n_features: the number of features
	@parameters: factor: the relative amount of y==1 samples;
	@parameters: offset
	"""				
	v_offset = UniformPoints(1,n_features)[0]
	v_offset = offset*v_offset/np.linalg.norm(v_offset)
	X = []
	Y = []
	assert n_features>1, 'The number of features has to be larger than 2 for the toy set'
	assert (factor <1 and factor >0), 'Factor has to satisfy 0<factor<1'
	N_pos = int(round(n_samples*factor))
	N_neg = n_samples-N_pos
	n_pos = 0
	n_neg = 0
	while (n_pos<N_pos or n_neg<N_neg): 
		v = UniformPoints(1,n_features)[0]
		if (IsInToySet(v,n_features) and n_pos<N_pos):
			X.append(v+v_offset)
			Y.append(1)
			n_pos += 1
		if (not IsInToySet(v,n_features) and n_neg<N_neg):
			X.append(v+v_offset)
			Y.append(0)
			n_neg += 1
	return X, Y

def IsInToySet(v,n_features):
	check = True
	i = 0
	while (i<n_features):
		if (i%3 == 0): # Square check
			if (abs(v[i])>0.5):
				check = False
				break
		elif (i%3 == 1): # circle radius 0.8 check with previous
			if (v[i]**2 + v[i-1]**2 > 0.5 or abs(v[i])<0.1):
				check = False
				break
		elif (i%3 == 2): # triangle check with previous
			if (abs(v[i] + v[i-1])>0.5):
				check = False
				break
		i += 1
	return check

def rv_bi(size, n_features):
		a = 2 * np.random.randint(2, size=(size,)) - 1
		return a

def rv_norm(size, n_features):
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

def boxplot_log(X, title):
	fs = 22  # fontsize

	# demonstrate how to toggle the display of different elements:
	fig, ax = plt.subplots()
	ax.boxplot(X, labels=['models'], showmeans=True)
	ax.set_title(title, fontsize=fs)

	ax.set_yscale('log')

	fig.subplots_adjust(hspace=0.4)
	plt.show()

def boxplot(X, title):
	fs = 22  # fontsize

	# demonstrate how to toggle the display of different elements:
	fig, ax = plt.subplots()
	ax.boxplot(X, labels=['models'], showmeans=True)
	ax.set_title(title, fontsize=fs)

	fig.subplots_adjust(hspace=0.4)
	plt.show()
