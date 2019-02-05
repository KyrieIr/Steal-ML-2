import numpy as np
import math

#  The adversary works with numpy arrays to keep its data. Therefore, every time
#  that there is a new training vector, the complete array is copied. This
#  maybe can be optimised. (sklearn works with numpy)

class Adversary_m(object):
    def __init__(self, budget, n_features, labels, strategy, api, n_init = 0,
                 min_n_lab = 3):
        self.set_budget(budget)
        self.set_n_features(n_features)
        self.set_labels(labels)

        self.set_strategy(strategy)

        self.api = api

        self.x_trn = np.empty([0, n_features])
        self.y_trn = np.empty([0])

        self.x_val = np.empty([0, n_features])
        self.y_val = np.empty([0])

        self.model = None

        self.queries = 0

        if n_init == 0:
            self.n_init = math.ceil(budget/10)

        self.min_n_lab = min_n_lab

    def set_strategy(self, strat):
        if strat == 'adaptive':
            self.strategy = strat
        else:
            raise Exception('The given strategy is not defined')

    def set_budget(self, b):
        if b>0:
            self.budget = b
        else:
            raise Exception('the budget of the adversary should be positive')

    def set_n_features(self, n):
        if n>0:
            self.n_features = n
        else:
            raise Exception('The number of features must be at least one')

    def set_labels(self, l):
        if len(l)>1:
            self.labels = l
        else:
            raise Exception('The adversary needs at least 2 different labels')

    def predict(self, x):
        '''
        The adversary predicts the class of x using its own trained model.
        :param x: The instances to be classified
        '''
        return self.model.predict(x)

    def query(self, x):
        '''
        Queries the point in x to the oracle.
        :param x: A numpy array
        :return: The class labels of the inputs in x.
        '''
        if self.n_features == 1:
            self.pay(x.size)
        elif len(x.shape) == 1:
            self.pay(1)
        else:
            self.pay(x.shape[0])

    def pay(self, price):
        if self.budget < price:
            raise NotEnoughBudget
        self.budget = self.budget-price

    def add_to_trn(self, x_new, y_new):
        '''
        Adds the instances x_new with labels y_new to the training set.
        :param x_new:
        :param y_new:
        '''
        if x_new == []:
            raise Exception('x_new cannot be empty')
        self.x_trn = np.vstack((self.x_trn, x_new))
        # This adds all new points to end the training vectors
        self.y_trn = np.append(self.y_trn, y_new)
        # This adds all labels to the end of the training labels

    def find_initial_points(self):
        '''
        Queries at least n_init random points to the oracle and adds them to the
        training set. If all class labels were returned at least min_n_lab the
        function stops, else the function continuous to query the oracle until
        all labels were recovered at least min_n_lab.

        Note, This step assumes that all features are scaled (not necessarily
        linear) to [-1,1].
        :return:
        '''
        x_new = np.random.rand(self.n_init, self.n_features)
        y_new = self.query(x_new)

        self.add_to_trn(x_new, y_new)


    def give_initial_points(self, x_init, y_init):
        '''
        If the adversary knows initial points of the classification problem he
        can add them to the training set before the attack.
        :param x_init: numpy array of dimensions (~, n_features)
        :param y_init: numpy array of dimensions (~)
        :return:
        '''
        self.add_to_trn(x_init, y_init)


class NotEnoughBudget(Exception):
    pass
