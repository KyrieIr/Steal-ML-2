import unittest

from sklearn.svm import SVC

from Adversary_m import Adversary_m
from sklearn.datasets import make_blobs
from sklearn import svm


class MyTestCase(unittest.TestCase):
    def setUp(self):
        x, y = make_blobs(n_samples=1000, n_features=4, centers=3,
                          cluster_std=2.0)
        #  rbf_feature = RBFSampler(gamma=0.1, n_components=100)
        #  x_features = rbf_feature.fit_transform(x)
        classifier: SVC = svm.SVC(C=1, kernel='rbf', gamma='scale',
                                  decision_function_shape='ovr',
                                  class_weight='balanced')
        classifier.fit(x, y)
        self.api = classifier
        self.adv = Adversary_m(100, 2, [0, 1], 'adaptive', self.api)

    def test_create_adversary(self):
        try:
            Adversary_m(100, 2, [0, 1], 'adaptive', self.api)
        except Exception:
            self.fail()
        with self.assertRaises(Exception):
            Adversary_m(0, 2, [0,1], 'adaptive', self.api)
        with self.assertRaises(Exception):
            Adversary_m(100, 0, [0, 1], 'adaptive', self.api)
        with self.assertRaises(Exception):
            Adversary_m(100, 2, [0], 'adaptive', self.api)
        with self.assertRaises(Exception):
            Adversary_m(100, 2, [0, 1], 'nonsense', self.api)

    def test_add_trn(self):
        self.assertTrue(self.adv.x_val == [])
        self.assertTrue(self.adv.x_val == [])


if __name__ == '__main__':
    unittest.main()
