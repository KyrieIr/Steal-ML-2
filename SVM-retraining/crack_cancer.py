import matplotlib.pyplot as plt
import numpy as np
import helper as hp
import time
from sklearn.datasets import load_breast_cancer
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
	data = load_breast_cancer()
	X1 = data.data
	Y1 = data.target
	n_features = 30

	scaling = np.max(X1)

	X1 = X1/(0.5*scaling)
	# gererates a data set X1 and labels Y1 with data from two circles, an inner circle 
	# and an outer circle. The labels in Y1 are 0 or 1, indiciating the inner or outer circle.
	# n_samples is the number of data points, noise is the noise on the data, factor is the 
	# ratio between the radius of the inner circle to the radius of the outer circle

	X2 = 2*hp.UniformPoints(10000, n_features)

	frac0 = len(np.where(Y1 == 0)[0]) / float(len(Y1)) # the number of points in the inner circle
	frac1 = len(np.where(Y1 == 1)[0]) / float(len(Y1)) # the number of points in the outer circle
	
	print("Percentage of '0' labels:", frac0)
	print("Percentage of '1' labels:", frac1)

	clf = svm.SVC() # creates a support vector classification object.
	clf.fit(X1, Y1) # fits the SVC to the data given
	# This model clf will be seen as the blackblox we have to crack.

	print('accuracy score', accuracy_score(Y1, clf.predict(X1))) # prints the accuracy of the model on the training data

	budget = 100*n_features
	budget = 1200
	nExp = 1
	error = 0
	error_list = []
	time_list = []
	for i in range(nExp):
		print('nExp = %d' %(i))
		time_i = time.clock()
		adv = Adversary(budget,n_features,'adaptive',clf)
		adv.SetRounds(20)
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

	hp.boxplot_log(error_list, 'Error, n=%d' %(nExp))
	hp.boxplot(time_list, 'Calculation time, n=%d' %(nExp) )

if __name__== '__main__':
	budget = 100
	rounds = 10
	main(budget, rounds)