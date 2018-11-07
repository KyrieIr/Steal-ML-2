__author__ = 'Fan'

import logging
import sys
import time

from sklearn.svm import SVC
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn import pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as sm

from utils.result import Result
from OfflineBase import OfflineBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

import os

# if os.uname()[0] == 'Darwin':
#     n_job = -1
#     verbose = 0
# else:
#     n_job = 8
#     verbose = 0


class RBFKernelRetraining(OfflineBase):
    def __init__(self, oracle, retrain_xy, test_xy, n_features):
        X_ex, y_ex = retrain_xy
        X_test, y_test = test_xy
        super(self.__class__, self).__init__(
                oracle, X_ex, y_ex, X_test, y_test, n_features
        )

    def grid_retrain_in_x(self):
        """
        Retrain the model using an RBF-Kernel SVM
        """
        gamma_range = np.logspace(-15, 3, 19, base=2) # returns array with 19 elements ranging from 2^-15 untill 2^3
        param_grid = dict(gamma=gamma_range)

        if len(np.unique(self.y_ex)) < 2: # checks if the set of labels contains only 1 label, if so this returns 1, 1
            return 1, 1 # does not change this model

        try:
            cv = StratifiedShuffleSplit(self.y_ex, n_iter=5, test_size=.2) # creates an object that contains a partioned y_ex into 2 groups with 
            # test_size of the all points in the test-grid, the rest in the train-set
            grid = GridSearchCV(SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)
            # Makes an object that can search in the parameter values for the given model (SVC) the best parameters.
            # SVC(C=1e5) creates an support vector classification object with error 1e5
            # n_jobs=a is the number of processors the program can use, =-1 means that all available processors will be used

            grid.fit(self.X_ex, self.y_ex) # will upgrade grid so that it fits the data X_ex, y_ex
            rbf_svc2 = grid.best_estimator_ # returns the best estimator for the data sets given
            print(rbf_svc2)
        except ValueError:
            rbf_svc2 = SVC(C=1e5)
            rbf_svc2.fit(self.X_ex, self.y_ex)

        self.set_clf2(rbf_svc2)
        return self.benchmark()

    def do(self):
        return self.grid_retrain_in_x()
        # return self.grid_retrain_in_f()

    def grid_retrain_in_f(self, n_dim=500):
        rbf_map = RBFSampler(n_dim, random_state=1)
        fourier_approx_svm = pipeline.Pipeline([("mapper", rbf_map),
                                                ("svm", LinearSVC())])

        # C_range = np.logspace(-5, 15, 21, base=2)
        # gamma_range = np.logspace(-15, 3, 19, base=2)
        # param_grid = dict(mapper__gamma=gamma_range, svm__C=C_range)
        # cv = StratifiedShuffleSplit(Y, n_iter=5, test_size=0.2, random_state=42)
        # grid = GridSearchCV(fourier_approx_svm, param_grid=param_grid, cv=cv)
        # grid.fit(X, Y)
        #
        # rbf_svc2 = grid.best_estimator_

        rbf_svc2 = fourier_approx_svm
        rbf_svc2.fit(self.X_ex, self.y_ex)

        self.set_clf2(rbf_svc2)
        return self.benchmark()
