# Deleted or adjusted functions, older versions


def UniformPoints(N, n_features):
	"""
	Function that returns N points uniformly distributed over a space 
	specified by the boundaries (2-Dimensional)

	@parameters: N: Number of points
	@parameters: boundaries: an array of 2-element arrays indicating the 
	"""
	number = int( np.floor( np.pow(N-1, 1/float(n_features)) ) ) # I do minus 1 so there is at least 1 point I can set to the middle of the domain
	output = []
	boundaries = [-1, 1]
	l = dim[1] - dim[0]
	basicint = np.linspace(dim[0],dim[1],num=number, endpoint=False) + l/(2*number)
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