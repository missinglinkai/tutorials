"""
From: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
This is an example of using scikit learn and integrating missinglink
"""

import numpy as np
from sklearn import neural_network, linear_model, ensemble
from sklearn.datasets import fetch_openml, get_data_home
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.transform import rotate

import missinglink

project = missinglink.SkLearnProject()

# Optional: Name this experiment. `display_name` is always visible in the experiments
# table. While the `description` is accessible by clicking the note icon.
project.set_properties(
    display_name="MNIST",
    description="Using scikit-learn")

print(__doc__)

# Load data from https://www.openml.org/d/554
print("Loading data")
print("Data home: {}".format(get_data_home()))
data, target = fetch_openml('mnist_784', version=1, return_X_y=True)
rotate = False
model_type = "forest"
#model_type = "mlp"

# rescale the data, use the traditional train/test split
print("Rescaling {} datapoints".format(data.shape))
data = data / 255.
split = 10000 # out of 70000
data_train, data_test = data[:split], data[split:]
target_train, target_test = target[:split], target[split:]

if rotate:
    print("Adding rotation")
    data_train = np.append(data_train, [rotate(im.reshape((28, 28)), 90).reshape(784) for im in data_train], axis=0)
    target_train = np.append(target_train, target_train, axis=0)

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
    model = ensemble.RandomForestClassifier(n_estimators=20)

project.set_hyperparams(split=split, rotate=rotate)

with project.train(model) as train:
    print("fit")
    model.fit(data_train, target_train)
    data_train_pred = model.predict(data_train)
    accuracy = accuracy_score(target_train, data_train_pred)
    train.add_metric('accuracy', accuracy)
    print("Training set accuracy: %f" % accuracy)

# Optianl: You can replace the class names in your confusion matrix using
# `set_properties(class_mapping=`. In the case of MNIST it's not too useful,
# but for example when training on Imagenet it's super helpful.
class_mapping = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
project.set_properties(class_mapping=class_mapping)

with project.test() as test:
    print("test")
    data_test_pred = model.predict(data_test)
    accuracy = accuracy_score(target_test, data_test_pred)
    test.add_metric('accuracy', accuracy)
    test.add_test_data(target_test, data_test_pred)
    print("Test set accuracy: %f" % accuracy)
    print("Confusion matrix:")
    print(confusion_matrix(target_test, data_test_pred))
