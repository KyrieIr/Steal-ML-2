import helper as hp
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from scipy.spatial import distance
import os


"""
This is the object class Adversary. It contains all info and properties of an adversary and has only black-box acces to 
a ML model.
"""

class Adversary(object):
	def __init__(self, b, strategy, API):
		if self.NotValidBudget(b):
			raise Exception('not a valid initialisation: budget')
		self.b = b
		self.q = 0
		if self.NotValidStrategy(strategy):
			raise Exception('not a valid initialisation: strategy')
		self.strategy = strategy
		self.SetAttributesStrategy()

		self.API = API

		self.NEG = 0
		self.POS = 1

		self.n_features = 2

		self.x_trn = []
		self.y_trn = []

		self.x_val = [] # Normally, the adversary does not have this data set
		self.y_val = [] # Normally, the adversary does not have this data set

		self.x_new = []
		self.y_new = []

		self.model = None

	# -------------------------------Simple checkers and setters------------------------------

	def NotValidStrategy(self,strategy):
		if strategy != 'adaptive':
			return True
		else:
			return False

	def SetAttributesStrategy(self):
		if (self.strategy == 'adaptive'):
			self.SetRounds(1)

	def NotValidBudget(self,b):
		if b<=0:
			return True
		else:
			return False

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
		self.nbinit = self.qprrnd + self.b % self.rounds # number of queries in the initial round
		print('nbinit', self.nbinit)

	def SetValidationSet(self, x_val, y_val):
		self.x_val = x_val
		self.y_val = y_val

	# ---------------------------------Stealing the model------------------------------------

	def FindInitialPoints(self):
		try:
			self.RemoveFromBudget(self.nbinit)
			new_x = hp.UniformPoints(self.nbinit,[[-1.5, 1.5], [-1, 2]])
			new_y = self.API.predict(new_x)
			self.x_new.extend(new_x)
			self.y_new.extend(new_y)
			self.x_trn.extend(self.x_new)
			self.y_trn.extend(self.y_new)
			return [new_x, new_y]
		except NotEnoughBudget as exc:
			print('You need at least enough budget for a first query')
			raise exc
			return None

	def StealAPIModel(self, error):
		self.error = error
		try:
			self.FindInitialPoints()
		except NotEnoughBudget as exc:
			raise exc
		gamma_range = np.logspace(-15, 3, 19, base=2) # returns array with 19 elements ranging from 2^-15 untill 2^3
		param_grid = dict(gamma=gamma_range)
		while True:
			try:
				# update trainig data

				cv = StratifiedShuffleSplit(self.y_trn, n_iter=5, test_size=.2) # creates an object that contains a partioned y_ex into 2 groups with 
				# test_size of the all points in the test-grid, the rest in the train-set
				grid = GridSearchCV(SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)

				grid.fit(self.x_trn,self.y_trn)
				self.model = grid
				self.benchmark()

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
			neg_x = self.RandomVector(2,self.NEG)
			pos_x = self.RandomVector(2,self.POS)

			neg_x, pos_x = self.push_to_b(neg_x, pos_x)

			#print('1', neg_x)

			self.x_trn.extend([neg_x])


			self.y_trn.extend(self.API.predict(neg_x))

			self.x_trn.extend([pos_x])
			self.y_trn.extend(self.API.predict(pos_x))

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
				l = self.model.predict(a)
				if l == label:
					return a
		else:
			return rv_gen(length)


	def push_to_b(self, xn, xp):
		assert self.model.predict(xn) == self.NEG
		assert self.model.predict(xp) == self.POS

		d = distance.euclidean(xn, xp) / \
			distance.euclidean(np.ones(self.n_features), np.zeros(self.n_features))
		if d < self.error:
			return xn, xp

		mid = .5 * np.add(xn, xp)
		l = self.model.predict(mid)
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
			print('After {0} queries our model has score_trn {1} and score_val {2}'.format(self.q, score_trn, score_val))

class NotEnoughBudget(Exception):
	pass


