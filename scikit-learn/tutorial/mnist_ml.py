"""
From: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
This is an example of using scikit learn and integrating missinglink
"""

from sklearn.datasets import fetch_openml, get_data_home
from sklearn.neural_network import MLPClassifier

import missinglink

project = missinglink.SkLearnProject()

print(__doc__)

# Load data from https://www.openml.org/d/554
print("Loading data")
print("Data home: {}".format(get_data_home()))
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
print("Rescaling data")
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print("Instantiating Multi-layer-perceptron")
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=6, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

print("fit")
with project.train(mlp) as train:
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
