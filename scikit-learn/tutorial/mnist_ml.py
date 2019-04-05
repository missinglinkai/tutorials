"""
From: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
This is an example of using scikit learn and integrating missinglink
"""

from sklearn.datasets import fetch_openml, get_data_home
from sklearn import neural_network
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from skimage.transform import rotate

import missinglink

project = missinglink.SkLearnProject()

print(__doc__)

# Load data from https://www.openml.org/d/554
print("Loading data")
print("Data home: {}".format(get_data_home()))
data, target = fetch_openml('mnist_784', version=1, return_X_y=True)
model_type = "forest"
#model_type = "mlp"

# rescale the data, use the traditional train/test split
print("Rescaling {} datapoints".format(data.shape))
data = data / 255.
split = 60000 # out of 70000
data_train, data_test = data[:split], data[split:]
target_train, target_test = target[:split], target[split:]

print("Instantiating Multi-layer-perceptron")
if model_type == "mlp":
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,),
    max_iter=6,
    alpha=1e-4,
    solver='sgd',
    verbose=10,
    tol=1e-4,
    random_state=1,
    learning_rate_init=.1)
elif model_type == "forest":
    model = ensemble.RandomForestClassifier()
elif model_type == "linear":
    model = linear_model.LinearRegression()

project.set_hyperparams(split=split)
print("fit")

def str_predictions(array):
    'Linear models output floats, convert them to int-strings.'
    if isinstance(array[0], float):
        return array.astype(int).astype(str)
    else:
        return array

with project.train(model) as train:
    model.fit(data_train, target_train)
    data_train_pred = str_predictions(model.predict(data_train))
    accuracy = accuracy_score(target_train, data_train_pred)
    train.add_metric('accuracy', accuracy)
    print("Training set accuracy: %f" % accuracy)

print("test")
with project.test() as test:
    data_test_pred = str_predictions(model.predict(data_test))
    accuracy = accuracy_score(target_test, data_test_pred)
    test.add_metric('accuracy', accuracy)
    print("Test set accuracy: %f" % accuracy)
    test.add_test_data(target_test, data_test_pred)
