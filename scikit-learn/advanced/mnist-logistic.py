"""
An example of using scikit learn and integrating missinglink
"""

import time
import numpy as np

from sklearn.datasets import fetch_openml, get_data_home
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import missinglink

print(__doc__)

# Author: Arthur Mensch <arthur.mensch@m4x.org>
# License: BSD 3 clause

# Turn down for faster convergence
t0 = time.time()
train_samples = 5000

# Load data from https://www.openml.org/d/554
print("Fetching dataset")
print("Data home: {}".format(get_data_home()))
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial', verbose=1,
                         penalty='l1', solver='saga', tol=0.1)

print("training")
project = missinglink.SkLearnProject()
with project.train(clf) as train:
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    train.add_metric('score', train_score)
    print("Train score: %.4f" % train_score)

print("testing")
with project.test() as test:
    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(X_test, y_test)

    test.add_metric('sparsity', sparsity)
    test.add_metric('score', train_score)

    # print('Best C % .4f' % clf.C_)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    print("Test score with L1 penalty: %.4f" % score)


coef = clf.coef_.copy()
run_time = time.time() - t0
print('Example run in %.3f s' % run_time)

