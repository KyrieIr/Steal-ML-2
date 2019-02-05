import numpy as np
import helper as hp
import time
from sklearn.datasets import make_blobs
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.kernel_approximation import RBFSampler


def main():
    x, y = make_blobs(n_samples=10, n_features=4, centers=3, cluster_std=2.0)
    #  rbf_feature = RBFSampler(gamma=0.1, n_components=100)
    #  x_features = rbf_feature.fit_transform(x)
    classifier = svm.SVC(C=1, kernel='rbf', gamma='scale', decision_function_shape='ovr', class_weight='balanced')
    classifier.fit(x, y)
    print(balanced_accuracy_score(y, classifier.predict(x)))
    print(type(x[0]))

if __name__ == '__main__':
    main()
