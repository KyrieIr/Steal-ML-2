from typing import Any, Union

import numpy as np
import math

from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score


#  The adversary works with numpy arrays to keep its data. Therefore, every time
#  that there is a new training vector, the complete array is copied. This
#  maybe can be optimised. (sklearn works with numpy)


class AdversaryM(object):
    def __init__(self, budget, n_features, labels, strategy, api, n_init=0,
                 min_n_lab=3):
        """

        :type min_n_lab: int
        """
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

        self.n_rounds = budget//(10*len(labels))

        if n_init == 0:
            self.n_init = math.ceil(budget/10)
        
        self.min_n_lab = min_n_lab

    def set_strategy(self, strat):
        if strat == 'adaptive':
            self.strategy = strat
        else:
            raise Exception('The given strategy is not defined')

    def get_budget(self):
        return self.budget

    def set_budget(self, b):
        if b > 0:
            self.budget = b
        else:
            raise Exception('the budget of the adversary should be positive')

    def set_n_features(self, n):
        if n > 0:
            self.n_features = n
        else:
            raise Exception('The number of features must be at least one')

    def set_labels(self, l):
        if len(l) > 1:
            self.labels = l
        else:
            raise Exception('The adversary needs at least 2 different labels')

    def set_n_rounds(self, n):
        if n >= 1 and n % len(self.labels) == 0:
            self.n_rounds = n
        else:
            raise Exception('n_rounds has to be >=0 and a multiple lf the ' +
                            'number of classes')

    def set_validation(self, x, y):
        self.x_val = x
        self.y_val = y

    def predict(self, x):
        """
        The adversary predicts the class of x using its own trained model.
        :param x: The instances to be classified
        """
        if self.model is None:
            raise Exception('model is not trained yet')
        return self.model.predict(x)

    def train(self):
        """
        This function trains the model of the adversary using the available training data. It trains a support vector
        machine with the one-versus-all approach and with balanced class_weights.
        :return:
        """
        # TODO: Do experiments on variation balanced, ovr, kernel...
        if np.size(self.x_trn) == 0:
            raise Exception('Cannot train a SVM in an empty training set')
        gamma_range = np.logspace(-15, 3, 19, base=2)  # returns array with 19 elements ranging from 2^-15 untill 2^3
        param_grid = dict(gamma=gamma_range)
        cv = StratifiedShuffleSplit(test_size=.2)  # creates an object thatcontains a partioned y_ex into 2 groups with
        svc = svm.SVC(C=1, kernel='rbf', gamma='scale',
                      decision_function_shape='ovr',
                      class_weight='balanced')
        grid = GridSearchCV(svc, param_grid=param_grid, cv=cv, n_jobs=-1)
        grid.fit(self.x_trn, self.y_trn)
        self.model = grid
        pass

    def query(self, x):
        """
        Queries the point in x to the oracle.
        :param x: A numpy array
        :return: The class labels of the inputs in x.
        """
        if self.n_features == 1:
            self.pay(x.size)
        elif len(x.shape) == 1:
            self.pay(1)
        else:
            self.pay(x.shape[0])
        return self.api.predict(x)

    def pay(self, price):
        if self.budget < price:
            raise NotEnoughBudget
        self.budget = self.budget-price

    def add_to_trn(self, x_new, y_new):
        """
        Adds the instances x_new with labels y_new to the training set.
        :param x_new:
        :param y_new:
        """
        if np.size(x_new) == 0:
            raise Exception('x_new cannot be empty')
        self.x_trn = np.vstack((self.x_trn, x_new))
        # This adds all new points to end the training vectors
        self.y_trn = np.append(self.y_trn, y_new)
        # This adds all labels to the end of the training labels

    def find_initial_points(self, n=-1):
        """
        Queries at least n_init random points to the oracle and adds them to the
        training set. If all class labels were returned at least min_n_lab the
        function stops, else the function continuous to query the oracle until
        all labels were recovered at least min_n_lab.

        Note, This step assumes that all features are scaled (not necessarily
        linear) to [-1,1].
        :return:
        """
        if n == -1:
            n = self.n_init
        try:
            x_new = self.get_random_instance(n)
            y_new = self.query(x_new)
            self.add_to_trn(x_new, y_new)
            for c in self.labels:
                while np.count_nonzero(self.y_trn == c) < self.min_n_lab:
                    x_new = self.get_random_instance(1)
                    y_new = self.query(x_new)
                    self.add_to_trn(x_new, y_new)
        except NotEnoughBudget:
            raise Exception('Not enough budget to find initial points')

    def find_q_rnd(self):
        # At the moment this function uses returns n_lab, maximum number of rounds
        n_lab = len(self.labels)
        q_rnd = n_lab
        rest = self.budget % n_lab
        try:
            self.find_initial_points(rest)  # adds the rest of the points to init
        except NotEnoughBudget:
            raise NotEnoughBudget('Not enough budget to divide rounds')
        return q_rnd

    def give_initial_points(self, x_init, y_init):
        """
        If the adversary knows initial points of the classification problem he
        can add them to the training set before the attack.
        :param x_init: numpy array of dimensions (~, n_features)
        :param y_init: numpy array of dimensions (~)
        :return:
        """
        self.add_to_trn(x_init, y_init)

    def steal_adaptive(self):
        q_rnd = self.find_q_rnd()
        threshold = 10**(-2)
        self.train()
        run_out = False
        while not run_out:
            try:
                self.adaptive_round(q_rnd, threshold)
            except NotEnoughBudget:
                run_out = True
                print('Adaptive stealing ended')

    def adaptive_round(self, q_rnd, threshold):
        if q_rnd % len(self.labels) != 0:
            raise Exception('q_rnd has to be a multiple of the number of classes')
        n_bat: int = q_rnd//len(self.labels)  # the number of batches in 1 round
        x_c = None
        x_r = None
        for i in range(0, n_bat):
            for c in self.labels:
                found = False
                while not found:  # Find an instance in class c
                    x_c = self.get_random_instance(1)
                    y_c = self.predict(x_c)
                    if y_c == c:
                        found = True
                found = False
                while not found:  # Find an instance not in class c
                    x_r = self.get_random_instance(1)
                    y_r = self.predict(x_r)
                    if y_r != c:
                        found = True
                x = self.push_to_boundary(x_c, x_r, threshold)
                y = self.query(x_c)
                self.add_to_trn(x, y)

    def push_to_boundary(self, x_c, x_r, threshold):
        return x_c+x_r

    def get_accuracy(self):
        return balanced_accuracy_score(self.predict(self.x_val), self.y_val)
    
    def get_random_instance(self, n):
        """
        Give n random instance of the input space
        :param n: number of instances
        :return: the instances
        """
        return 2*np.random.rand(n, self.n_features) - 1


class NotEnoughBudget(Exception):
    pass
