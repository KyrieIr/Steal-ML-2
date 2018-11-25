import helper as hp
import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.spatial import distance
import os


"""
This is the object class Adversary. It contains all info and properties of an adversary and has only black-box acces to 
a ML model.
"""

class Adversary(object):
	def __init__(self, b, n_features, strategy, API):
		if NotValidBudget(b):
			raise Exception('not a valid initialisation: budget')
		self.b = b
		self.q = 0
		self.SetStrategy(strategy)

		self.API = API

		self.NEG = 0
		self.POS = 1

		self.n_features = n_features

		self.x_trn = []
		self.y_trn = []

		self.x_val = [] # Normally, the adversary does not have this data set
		self.y_val = [] # Normally, the adversary does not have this data set

		self.x_new = []
		self.y_new = []

		self.time_start = 0
		self.time_end = 0

		self.rundata = []

		self.model = None

	# -------------------------------Simple checkers and setters------------------------------

	def SetStrategy(self, strategy):
		if strategy == 'monoAdaptive':
			self.nbinit = 0
			self.qprrnd = 2
			self.strategy = strategy
		elif strategy == 'adaptive':
			self.strategy = 'adaptive'
			self.SetRounds(1)
		else:
			raise ValidStrategyException

	def RemoveFromBudget(self,rm):
		if (self.b-rm)<0:
			raise NotEnoughBudget()
		else:
			self.b -= rm
			self.q += rm

	def SetRounds(self, r):
		assert (r>0), "the number of rounds must be positive"
		assert (r<self.b), "the number of rounds must be less than the number of queries" 
		self.rounds = r # total number of rounds
		self.qprrnd = np.floor(self.b/self.rounds) # number of queries per round

	def SetValidationSet(self, x_val, y_val):
		self.x_val = x_val
		self.y_val = y_val

	# ---------------------------------Stealing the model------------------------------------

	def FindInitialPoints(self):
	# We will assume in this step that the data is scaled within the interval [-1,1] in each feature direction
		try:
			posFound = 0
			negFound = 0
			while (posFound<3 or negFound<3 or (posFound+negFound)%(10*self.n_features)!=0 or (posFound+negFound)%(self.qprrnd)!=0): # must be more than the number of classes + 1
				self.RemoveFromBudget(1)
				new_x = hp.UniformPoints(1,self.n_features)
				new_y = self.API.predict(new_x)
				if (new_y==self.POS):
					posFound +=1
				if (new_y==self.NEG):
					negFound +=1
				self.x_new.extend(new_x)
				self.y_new.extend(new_y)
			self.x_trn.extend(self.x_new)
			self.y_trn.extend(self.y_new)

		except NotEnoughBudget as exc:
			print('You need at least enough budget for a first query')
			raise exc
			return None

	def StealAPIModel(self, error):
		print('queries, acc_trn, acc_val, time_step')
		self.error = error
		self.time_start = time.clock()
		try:
			self.FindInitialPoints()
		except NotEnoughBudget as exc:
			raise exc
		gamma_range = np.logspace(-15, 3, 19, base=2) # returns array with 19 elements ranging from 2^-15 untill 2^3
		param_grid = dict(gamma=gamma_range)
		while True:
			try:
				# update trainig data

				cv = StratifiedShuffleSplit(test_size=.2) # creates an object thatcontains a partioned y_ex into 2 groups with 
				# test_size of the all points in the test-grid, the rest in the train-set
				grid = GridSearchCV(SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)


				grid.fit(self.x_trn,self.y_trn)
				self.model = grid
				self.benchmark()
				time_end = time.clock()



				self.SearchNew()
			# TODO Uniform Stop criterium?
			# TODO find points on the decision boundary of our trained model. Use these points to blackbox the API
			except (NotEnoughBudget, KeyboardInterrupt) as e:
				print(e)
				print('Done')
				break


	# TODO: in Steal-ML, they keep track of how many queries this takes, However, since it is quering our 
	# own model, we do not pay for those queries and there is only an extra computational cost.
	def SearchNew(self):
		n = self.qprrnd
		if (n % 2 != 0):
			print('number of points in round was not even')
			n += 1
		m = int(n/2)

		try:
			self.RemoveFromBudget(n)
		except NotEnoughBudget as e:
			raise e

		#print(self.x_trn)
		for i in range(0,m):
			neg_x = self.RandomVector(self.n_features,self.NEG)
			pos_x = self.RandomVector(self.n_features,self.POS)

			neg_x, pos_x = self.push_to_b(neg_x, pos_x)

			#print('1', neg_x)

			self.x_trn.extend([neg_x])


			self.y_trn.extend(self.API.predict(neg_x.reshape(1,-1)))

			self.x_trn.extend([pos_x])
			self.y_trn.extend(self.API.predict(pos_x.reshape(1,-1)))

		#print(self.x_trn)

	def RandomVector(self, length, label=None):
		mean = 0
		low = -1
		high = +1
		#if label is not None:
		#	assert label in (self.NEG, self.POS), 'unknown label %d' % label

		#rv_gen = rv_uni

		if label is not None:
			while True:
				a = hp.rv_uni(low,high,length)
				l = self.model.predict(a.reshape(1,-1))
				if l == label:
					return a
		else:
			return rv_gen(length)


	def push_to_b(self, xn, xp):
		assert self.model.predict(xn.reshape(1,-1)) == self.NEG
		assert self.model.predict(xp.reshape(1,-1)) == self.POS

		d = distance.euclidean(xn, xp) / \
			distance.euclidean(np.ones(self.n_features), np.zeros(self.n_features))
		if d < self.error:
			return xn, xp

		mid = .5 * np.add(xn, xp)
		l = self.model.predict(mid.reshape(1,-1))
		if l == self.NEG:
			return self.push_to_b(mid, xp)
		else:
			return self.push_to_b(xn, mid)

	def benchmark(self):
		if (len(self.x_val) == 0):
			y_self = self.model.predict(self.x_trn)
			score = float(sum(y_self == self.y_trn))/float(len(self.x_trn))
			print('After {0} queries our model has a score {1}'.format(self.q, score))
		else:
			y_self = self.model.predict(self.x_trn)
			score_trn = float(sum(y_self == self.y_trn))/float(len(self.x_trn))
			y_self = self.model.predict(self.x_val)
			score_val = float(sum(y_self == self.y_val))/float(len(self.x_val))
			self.time_end = time.clock()
			time_spent = self.time_end - self.time_start			
			print('%d   %f   %f   %f' %(self.q, score_trn, score_val, time_spent))

			self.rundata.append([self.q, (1-score_val), time_spent])
			self.time_start = time.clock()

	def predict(self, X):
		return self.model.predict(X)

def NotValidStrategy(strategy):
	if strategy != 'adaptive':
		return True
	elif strategy != 'monoAdaptive':
		return True
	else:
		return False

def NotValidBudget(b):
	if b<=0:
		return True
	else:
		return False

class NotEnoughBudget(Exception):
	pass

class ValidStrategyException(Exception):
	pass


