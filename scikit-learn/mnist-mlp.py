"""
From: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
This is an example of using scikit learn and integrating missinglink
"""

#import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

import missinglink

print(__doc__)

project = missinglink.SkLearnProject()

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
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=6, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

print("fit")
with project.train(mlp) as train:
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())

# plt.show()
