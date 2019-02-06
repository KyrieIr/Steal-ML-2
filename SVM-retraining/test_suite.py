import unittest
import math
from sklearn.svm import SVC
from AdversaryM import AdversaryM
from sklearn.datasets import make_blobs
from sklearn import svm
from sklearn import preprocessing
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        x, y = make_blobs(n_samples=1000, n_features=4, centers=3,
                          cluster_std=2.0)
        #  rbf_feature = RBFSampler(gamma=0.1, n_components=100)
        #  x_features = rbf_feature.fit_transform(x)
        classifier: SVC = svm.SVC(C=1, kernel='rbf', gamma='scale',
                                  decision_function_shape='ovr',
                                  class_weight='balanced')
        x = preprocessing.normalize(x, axis=0, norm = 'max')
        # normalises the features with l2-norm

        classifier.fit(x, y)
        self.x = x
        self.y = y
        self.api = classifier
        self.adv = AdversaryM(10000, 4, [0, 1, 2], 'adaptive', self.api)

    def test_create_adversary(self):
        try:
            AdversaryM(100, 2, [0, 1], 'adaptive', self.api)
        except Exception:
            self.fail()
        with self.assertRaises(Exception):
            AdversaryM(0, 2, [0,1], 'adaptive', self.api)
        with self.assertRaises(Exception):
            AdversaryM(100, 0, [0, 1], 'adaptive', self.api)
        with self.assertRaises(Exception):
            AdversaryM(100, 2, [0], 'adaptive', self.api)
        with self.assertRaises(Exception):
            AdversaryM(100, 2, [0, 1], 'nonsense', self.api)

    def test_add_trn(self):
        self.assertTrue(np.size(self.adv.x_trn) == 0)
        self.assertTrue(np.size(self.adv.y_trn) == 0)
        a = np.random.rand(1, 4)
        a_lab = 1
        self.adv.add_to_trn(a, a_lab)
        self.assertTrue(self.adv.x_trn.shape[0] == 1)
        self.assertTrue(self.adv.x_trn.shape[1] == 4)
        self.assertTrue(self.adv.y_trn.size == 1)

        a = np.random.rand(2, 4)
        a_lab = [0,1]
        self.adv.add_to_trn(a, a_lab)
        self.assertTrue(self.adv.x_trn.shape[0] == 3)
        self.assertTrue(self.adv.x_trn.shape[1] == 4)
        self.assertTrue(self.adv.y_trn.size == 3)

        with self.assertRaises(Exception):
            a = np.random.rand(1,3)
            a_lab = 1
            self.adv.add_to_trn(a, a_lab)

    def test_find_initial_points(self):
        self.assertTrue(np.size(self.adv.x_trn) == 0)
        self.assertTrue(np.size(self.adv.y_trn) == 0)
        self.adv.find_initial_points(10)
        self.assertTrue(self.adv.x_trn.shape[0] >= 10)
        self.assertTrue(self.adv.y_trn.size >= 10)

    def test_train(self):
        with self.assertRaises(Exception):
            self.adv.train()
        self.adv.add_to_trn(self.x, self.y)
        self.adv.train()
        self.adv.set_validation(self.x, self.y)
        print(self.adv.get_accuracy())
        self.assertTrue(self.adv.get_accuracy() > 0.9)


if __name__ == '__main__':
    unittest.main()
