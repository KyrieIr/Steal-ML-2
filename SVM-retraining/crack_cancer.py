import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.metrics import accuracy_score
from Adversary import Adversary
from mpl_toolkits.mplot3d import Axes3D
from helper import ToySet, UniformPoints
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

	X1 = X1/scaling
	# gererates a data set X1 and labels Y1 with data from two circles, an inner circle 
	# and an outer circle. The labels in Y1 are 0 or 1, indiciating the inner or outer circle.
	# n_samples is the number of data points, noise is the noise on the data, factor is the 
	# ratio between the radius of the inner circle to the radius of the outer circle

	X2 = 2*UniformPoints(10000, n_features)

	frac0 = len(np.where(Y1 == 0)[0]) / float(len(Y1)) # the number of points in the inner circle
	frac1 = len(np.where(Y1 == 1)[0]) / float(len(Y1)) # the number of points in the outer circle
	
	print("Percentage of '0' labels:", frac0)
	print("Percentage of '1' labels:", frac1)

	clf = svm.SVC() # creates a support vector classification object.
	clf.fit(X1, Y1) # fits the SVC to the data given
	# This model clf will be seen as the blackblox we have to crack.

	print('accuracy score', accuracy_score(Y1, clf.predict(X1))) # prints the accuracy of the model on the training data

	adv = Adversary(budget*n_features,n_features,'adaptive',clf)
	adv.SetRounds(rounds*n_features)
	adv.SetValidationSet(X2,clf.predict(X2))
	adv.StealAPIModel(.1)

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

if __name__== '__main__':
	budget = 100
	rounds = 10
	main(budget, rounds)