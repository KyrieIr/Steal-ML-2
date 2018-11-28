import matplotlib.pyplot as plt
import numpy as np
import helper as hp
import time
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn import svm
from sklearn.metrics import accuracy_score
from Adversary import Adversary
from mpl_toolkits.mplot3d import Axes3D

"""
This function uses the adaptive retraining strategy from the Steal-ML paper to 
crack a Ml algorithm. We will try to reach a high accuracy with a signficantly smaller
number of queries than the number of initial training points.
"""

def main(budget, rounds):
	n_features = 2
	X1, Y1 = make_circles(n_samples=800, noise=0.1, factor=0.4) # defined in sklearn.datasets
	#X1, Y1 = make_blobs(n_samples=800, n_features=n_features, centers=2, random_state=0)
	#X1, Y1 = hp.ToySet(n_samples = 2000, n_features=n_features,factor=0.5, offset=0.1)
	scaling = np.max(np.abs(X1))

	X1 = X1/(100*scaling)
	# gererates a data set X1 and labels Y1 with data from two circles, an inner circle 
	# and an outer circle. The labels in Y1 are 0 or 1, indiciating the inner or outer circle.
	# n_samples is the number of data points, noise is the noise on the data, factor is the 
	# ratio between the radius of the inner circle to the radius of the outer circle

	X2, Y2 = make_circles(n_samples=2000, noise=0.3, factor=0.4) # the reference data set
	#X2, Y2 = make_blobs(n_samples=8000, n_features = n_features, centers=2, random_state=0)
	#X2, Y2 = hp.ToySet(n_samples = 2000, n_features=n_features,factor=0.5)

	X2 = X2/(100*scaling)
	#frac0 = Y1.count(0) / float(len(Y1)) # the number of points in the inner circle
	#frac1 = Y1.count(1) / float(len(Y1)) # the number of points in the outer circle
	
	#print("Percentage of '0' labels:", frac0)
	#print("Percentage of '1' labels:", frac1)

	clf = svm.SVC() # creates a support vector classification object.
	clf.fit(X1, Y1) # fits the SVC to the data given
	# This model clf will be seen as the blackblox we have to crack.

	print('accuracy score', accuracy_score(Y1, clf.predict(X1))) # prints the accuracy of the model on the training data

	
	nExp = 1000
	error = 0
	error_list = []
	time_list = []
	for i in range(nExp):
		print('nExp = %d' %(i))
		time_i = time.clock()
		adv = Adversary(200,n_features,'adaptive',clf)
		adv.SetRounds(100)
		adv.SetValidationSet(X2,clf.predict(X2))
		adv.StealAPIModel(.1)
		time_f = time.clock()
		error = (1-adv.GetAccuracy())
		error_list.append(error)
		time_list.append(time_f-time_i)

	print('Average error 	:', sum(error_list)/nExp)
	print('Average time 	:', sum(time_list)/nExp)

	X1 = np.array(X1)
	Y1 = np.array(Y1)

	if (n_features == 3):
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(X1[Y1==1,0],X1[Y1==1,1],X1[Y1==1,2],c='b')
		ax.scatter(X1[Y1==0,0],X1[Y1==0,1],X1[Y1==0,2],c='r')
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		plt.show()
	elif (n_features==2):
		plt.figure()
		plt.scatter(X1[Y1==1,0],X1[Y1==1,1],c='b')
		plt.scatter(X1[Y1==0,0],X1[Y1==0,1],c='r')
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		plt.show()		

	hp.boxplot_log(error_list, 'Error, n=%d' %(nExp))
	hp.boxplot(time_list, 'Calculation time, n=%d' %(nExp) )

if __name__== '__main__':
	budget = 200
	rounds = 10
	main(budget, rounds)