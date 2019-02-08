import numpy as np
import pandas as pd
from sklearn import svm
from AdversaryM import AdversaryM
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from helper import benchmark_failed_initial, get_random_instance


with open('data/wine/winequality_red.csv') as csv_file:
    csv_reader = pd.read_csv(csv_file, sep=';')
x = np.array(csv_reader.values)[:, 0:11]
x = preprocessing.normalize(x, axis=0, norm='max')
y = np.int_(np.array(csv_reader.values)[:, 11])

labels = np.int_(np.unique(y))
x_random = get_random_instance(1000, 11)

gamma_range = np.logspace(-15, 3, 19, base=2)  # returns array with 19 elements ranging from 2^-15 untill 2^3
param_grid = dict(gamma=gamma_range)
cv = StratifiedShuffleSplit(test_size=.2)  # creates an object thatcontains a partioned y_ex into 2 groups with
svc = svm.SVC(C=1, kernel='rbf', gamma='scale',
              decision_function_shape='ovr',
              class_weight='balanced')
grid = GridSearchCV(svc, param_grid=param_grid, cv=cv, n_jobs=-1)
grid.fit(x, y)

classifier_acc = balanced_accuracy_score(y, grid.predict(x))
print('The oracle has balanced accuracey {0:10.5f}'.format(classifier_acc))
print('The oracle as kernel parameters: {0}'.format(grid.best_params_))
benchmark_failed_initial(y, labels)
benchmark_failed_initial(grid.predict(x_random), labels)

adv = AdversaryM(11*100, 11, labels, 'adaptive', grid)
adv.steal_adaptive()
print('The adversary has balanced accuracy {0:10.5f}'.format(adv.get_accuracy()))
print('The adversary has kernel parameters: {0}'.format(adv.model.best_params_))

'''
Data was not clean, this was how I found the rows that were not, I adjusted them (I did not leave the rows out)
for i in range(0,len(a[:,10])):
    print(i)
    a[i, :] = np.float_(a[i, :])
'''
