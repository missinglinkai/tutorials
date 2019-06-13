# Adopted and modified from Keras's MNIST example.
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
#
# In this example, we will train a simple convolutional neural network on the MNIST dataset
# We will then integrate MissingLink SDK in order to remotely monitor our training, validation
# and testing process.

from __future__ import print_function

import argparse
import missinglink

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Input params
NUM_CLASSES = 10  # The MNIST dataset has 10 classes, representing the digits 0 through 9.
IMAGE_SIZE = 28  # The MNIST images are always 28x28 pixels.
IMAGE_ROWS, IMAGE_COLUMNS = IMAGE_SIZE, IMAGE_SIZE  # Image dimensions

# Network params
NUM_FILTERS_CONV_1 = 32  # Number of filters to use for first convolutional layer
NUM_FILTERS_CONV_2 = 64  # Number of filters to use for second convolutional layer
NUM_FILTERS_DENSE = 128  # Number of filters to use for dense layer
POOL_SIZE = (2, 2)  # Size of pooling area for max pooling
KERNEL_SIZE = (3, 3)  # Convolutional kernel size
ACTIVATION_RELU = 'relu'
ACTIVATION_SOFTMAX = 'softmax'

# Training params
EPOCHS = 8
BATCH_SIZE = 128
CONV_DROPOUT = 0.25
DENSE_DROPOUT = 0.5

# Validation params
VALIDATION_SPLIT = 0.2

# MissingLink credentials
OWNER_ID = 'Fill in your owner id'
PROJECT_TOKEN = 'Fill in your project token'

# The MNIST data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, IMAGE_ROWS, IMAGE_COLUMNS)
    x_test = x_test.reshape(x_test.shape[0], 1, IMAGE_ROWS, IMAGE_COLUMNS)
    input_shape = (1, IMAGE_ROWS, IMAGE_COLUMNS)
else:
    x_train = x_train.reshape(x_train.shape[0], IMAGE_ROWS, IMAGE_COLUMNS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMAGE_ROWS, IMAGE_COLUMNS, 1)
    input_shape = (IMAGE_ROWS, IMAGE_COLUMNS, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()

model.add(Conv2D(NUM_FILTERS_CONV_1,
                 kernel_size=KERNEL_SIZE,
                 activation=ACTIVATION_RELU,
                 input_shape=input_shape))
model.add(Conv2D(NUM_FILTERS_CONV_2, KERNEL_SIZE, activation=ACTIVATION_RELU))

model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(CONV_DROPOUT))
model.add(Flatten())

model.add(Dense(NUM_FILTERS_DENSE, activation=ACTIVATION_RELU))
model.add(Dropout(DENSE_DROPOUT))
model.add(Dense(NUM_CLASSES, activation=ACTIVATION_SOFTMAX))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy', 'categorical_accuracy', 'mean_squared_error', 'hinge'])

# Provide an alternative to provide MissingLinkAI credential
parser = argparse.ArgumentParser()
parser.add_argument('--owner-id')
parser.add_argument('--project-token')

# Override credential values if provided as arguments
args = parser.parse_args()
OWNER_ID = args.owner_id or OWNER_ID
PROJECT_TOKEN = args.project_token or PROJECT_TOKEN

missinglink_callback = missinglink.KerasCallback(
    owner_id=OWNER_ID, project_token=PROJECT_TOKEN)

missinglink_callback.set_properties(
    display_name='Keras convolutional neural network',
    description='Two dimensional convolutional neural network')

model.fit(
    x_train, y_train, batch_size=BATCH_SIZE,
    nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT,
    callbacks=[missinglink_callback])

with missinglink_callback.test(model):
    score = model.evaluate(x_test, y_test)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])