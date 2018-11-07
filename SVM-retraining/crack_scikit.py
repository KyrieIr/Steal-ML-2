import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn import svm
from sklearn.metrics import accuracy_score

"""
This function uses the retraining strategy from the Steal-ML paper to 
crack a Ml algorithm. 
"""

def main():
	X1, Y1 = make_circles(n_samples=800, noise=0.1, factor=0.4) # defined in sklearn.datasets
	# gererates a data set X1 and labels Y1 with data from two circles, an inner circle 
	# and an outer circle. The labels in Y1 are 0 or 1, indiciating the inner or outer circle.
	# n_samples is the number of data points, noise is the noise on the data, factor is the 
	# ratio between the radius of the inner circle to the radius of the outer circle

	frac0 = len(np.where(Y1 == 0)[0]) / float(len(Y1)) # the number of points in the inner circle
	frac1 = len(np.where(Y1 == 1)[0]) / float(len(Y1)) # the number of points in the outer circle
	
	print("Percentage of '0' labels:", frac0)
	print("Percentage of '1' labels:", frac1)

	clf = svm.SVC() # creates a support vector classification object.
	clf.fit(X1, Y1) # fits the SVC to the data given

	print(accuracy_score(Y1, clf.predict(X1))) # prints the accuracy of the model on the training data

	print(X1)
	plt.figure()
	plt.scatter(X1[Y1==0,0],X1[Y1==0,1],c='r')
	plt.scatter(X1[Y1==1,0],X1[Y1==1,1],c='b')
	plt.show()

if __name__== '__main__':
	main()