import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys

np.set_printoptions(threshold=np.nan)

sys.path.append('../..')
from algorithms.OnlineBase import OnlineBase
from algorithms.RBFTrainer import RBFKernelRetraining


def main():
    X1, Y1 = make_circles(n_samples=800, noise=0.07, factor=0.4) # defined in sklearn.datasets
    # gererates a data set X1 and labels Y1 with data from two circles, an inner circle 
    # and an outer circle. The labels in Y1 are 0 or 1, indiciating the inner or outer circle.
    # n_samples is the number of data points, noise is the noise on the data, factor is the 
    # ratio between the radius of the inner circle to the radius of the outer circle
    frac0 = len(np.where(Y1 == 0)[0]) / float(len(Y1)) # the number of points in the inner circle
    frac1 = len(np.where(Y1 == 1)[0]) / float(len(Y1)) # the number of points in the outer circle

    print("Percentage of '0' labels:", frac0)
    print("Percentage of '1' labels:", frac1)

    plt.figure()
    plt.subplot(121)
    plt.title(
        "Our Dataset: N=200, '0': {0} '1': {1} ".format(
            frac0,
            frac1), # format is a way of printing reals/integers 
        fontsize="large")

    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))

    clf = svm.SVC() # creates a support vector classification object.
    clf.fit(X1, Y1) # fits the SVC to the data given

    print(accuracy_score(Y1, clf.predict(X1))) # prints the accuracy of the model on the training data

    ex = OnlineBase('circle', 1, 0, clf.predict, 2, 'uniform', .1)
    step = 6
    train_x, train_y = [], []
    val_x, val_y = [], []
    while True:
        ex.collect_pts(step) # collects step points around the decision boundary of ex
        train_x.extend(ex.pts_near_b) # first step this list is empty.
        train_y.extend(ex.pts_near_b_labels) # first step this list is empty
        #val_x.extend(ex.support_pts)
        #val_y.extend(ex.support_labels)
        try:
            e = RBFKernelRetraining('circle', [train_x, train_y], [train_x, train_y], n_features=2) # creates a new object every time? is this the smartest way to retrain?
            print(ex.get_n_query(), e.grid_retrain_in_x()) # TODO I do not get how ex and e are connected, it seems to me that 
            # grid_retrain_in_x() indeeds does something like retraing the model, but there are no points added to pts_near_b or are there?
        except KeyboardInterrupt: ## TODO stop condition!!
            print('Done')
            break

    train_x = np.array(train_x)
    plt.subplot(122)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.show()


if __name__ == '__main__':
    main()
