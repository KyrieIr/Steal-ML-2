""" 
This script is used to generate the data comparing the relation between the accuracy and the number of rounds used in the adaptive retraining procedure. 

This script does this comparising for the ToySet without offset. 

The budget of the adversary is scaled with the number of parameters he wants to obtain (which is in our case the number of features plus 2 hyperparameters)
"""
import sys
sys.path.append('../')

from sklearn import svm
import numpy as np
import helper as hp
import matplotlib.pyplot as plt
import time
from Adversary import Adversary
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer



def main():
	t_tot_start = time.clock()
	
	data_cancer = load_breast_cancer()
	X = data_cancer.data
	Y = data_cancer.target
	n_features = 30
	budget = (n_features + 2)*100
	budget = 1200

	print(X)
	print(np.max(X))
	X = np.tanh(X)
	print(np.max(X))

	clf = svm.SVC() # creates a support vector classification object. Default with an rbf kernel
	clf.fit(X, Y) # fits the SVC to the data given
	# This model clf will be seen as the blackblox we have to crack.

	Y_clf = clf.predict(X)

	X_val = hp.UniformPoints(10000,n_features)
	Y_val = clf.predict(X_val)

	print("Accuracy of the model %f" % ( accuracy_score(Y, Y_clf))) # prints the accuracy of the model on the training data
	print('Nb support vectors 0 : %d' %(clf.n_support_[0]))
	print("Nb support vectors 1 : %d" %(clf.n_support_[1]))

	font = {'family' : 'normal',
        'size'   : 16}

	plt.rc('font', **font)

	averaging = 1
	plt.figure()
	qprrnd = [2,4,6,10,20,30,40,50,60,80,100,120,150,200,300,400,600,1200]
	data = np.zeros(len(qprrnd))
	i = 0
	for d in qprrnd:
		for j in range(averaging):
			adv = Adversary(budget,n_features,'adaptive',clf)
			adv.SetRounds(budget/d)
			adv.SetValidationSet(X_val,Y_val)
			adv.StealAPIModel(.1)
			data[i] += (1-adv.GetAccuracy())
			print[data]
		i += 1


	data = data/averaging
	print(data)
	plt.loglog(qprrnd, data)

	print(data)
	plt.xlabel('queries per round', fontsize=22)
	plt.ylabel('error', fontsize=22)
	plt.title('Error in function of queries per round', fontsize=22)	
	plt.grid()	
	plt.show()

	t_tot_end = time.clock()
	print(t_tot_end-t_tot_start)


if __name__== '__main__':
	main()