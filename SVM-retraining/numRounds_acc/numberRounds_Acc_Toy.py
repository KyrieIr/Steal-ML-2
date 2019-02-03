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
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from Adversary import Adversary
from sklearn.metrics import accuracy_score



def main():
	t_tot_start = time.clock()
	
	n_features = 2
	budget = (n_features + 2)*10
	
	X, Y = hp.ToySet(n_samples = 200, n_features=n_features)

	clf = svm.SVC() # creates a support vector classification object. Default with an rbf kernel
	clf.fit(X, Y) # fits the SVC to the data given
	# This model clf will be seen as the blackblox we have to crack.

	Y_clf = clf.predict(X)

	X_val = hp.UniformPoints(10000,n_features)
	Y_val = clf.predict(X_val)

	print("Accuracy of the model %f" % (accuracy_score(Y_clf, Y))) # prints the accuracy of the model on the training data
	print('Nb support vectors 0 : %d' %(clf.n_support_[0]))
	print("Nb support vectors 1 : %d" %(clf.n_support_[1]))


	font = {'family' : 'normal',
        'size'   : 16}

	plt.rc('font', **font)

	data ={}
	plt.figure()
	qprrnd = [10,20]

	for d in qprrnd:
		adv = Adversary(budget,n_features,'adaptive',clf)
		adv.SetRounds(budget/d)
		adv.SetValidationSet(X_val,Y_val)
		adv.StealAPIModel(.1)
		data[d] = np.asarray(adv.rundata)

	for i in range(1):
		for d in qprrnd:
			adv = Adversary(budget,n_features,'adaptive',clf)
			adv.SetRounds(budget/d)
			adv.SetValidationSet(X_val,Y_val)
			adv.StealAPIModel(.1)
			data[d] += np.asarray(adv.rundata)
			print(data[d][:,0])

	for d in qprrnd:
		data[d] = data[d]/3
		plt.semilogy(data[d][:,0],data[d][:,1],label=d)

	print(data)
	plt.legend()
	plt.xlabel('queries', fontsize=22)
	plt.ylabel('error', fontsize=22)
	plt.title('Error of different choices of d', fontsize=22)	
	plt.grid()	
	plt.show()

	t_tot_end = time.clock()
	print(t_tot_end-t_tot_start)













if __name__== '__main__':
	main()