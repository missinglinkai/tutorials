# Adopted and modified from PyCaffe's MNIST example.
# https://github.com/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb
#
# In this example, we will train a LeNet network on the MNIST dataset
# We will then integrate MissingLink SDK in order to remotely monitor our training, validation
# and testing process.

import caffe
import os

from caffe import layers as L, params as P
from subprocess import call

caffe_root = os.environ['CAFFE_ROOT']

os.environ['GLOG_minloglevel'] = '1'  # Set the logging level

os.chdir(caffe_root)  # Preare to run scripts from caffe root
call('data/mnist/get_mnist.sh')  # Download MNIST data
call('examples/mnist/create_mnist.sh')  # Prepare MNIST data
os.chdir('examples')  # Back to examples

def lenet(lmdb, batch_size):
    # Our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

caffe.set_mode_cpu()  # Or use gpu by running the next line instead if your machine has access to a GPU
# caffe.set_mode_gpu()

solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')  # Load the solver

solver.solve()