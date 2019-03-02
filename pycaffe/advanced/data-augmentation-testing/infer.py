import sys
import caffe
from create_net import build_net
from pylab import *
import time

caffe.set_device(0)
caffe.set_mode_gpu()

#Load weights of model to be evaluated
weights = '../../datasets/102flowers/snapshots/dataAugmentationAll_iter_200000.caffemodel'
# weights = 'models/bvlc_reference_caffenet.caffemodel'

#Number of image to be tested are batch size (100) * test iterations
test_iters = 2

config = {}
config['dir'] = '../../datasets/flickr_style'
config['mean'] = (104, 117, 123)
config['num_labels'] = 102
config['batch_size'] = 500 #AlexNet 100, VGG 40
config['resize'] = False #Resize the image to the given size before cropping
config['resize_w'] = 224
config['resize_h'] = 224
config['cropping'] = False #True
config['always_center_crop'] = False
config['crop_w'] = 224 #Train with a random crop of this size
config['crop_h'] = 224 #227 AlexNet, 224 VGG16, Inception
config['crop_margin'] = 0 #The crop won't include the margin in pixels
config['mirror'] = False#True #Mirror images with 50% prob
config['rotate_prob'] = 0 #.2 #Rotation probability
config['rotate_angle'] = 8 #15,8 #Rotate with angle between -a and a
config['HSV_prob'] = 0 #.2 #Jitter saturation and vale of the image with this prob
config['HSV_jitter'] = 0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter
config['color_casting_prob'] = 0 #0.05  #Alterates each color channel (with the given prob for each channel) suming jitter
config['color_casting_jitter'] = 10 #Sum/substract 10 from the color channel
config['scaling_prob'] = 0 #.5 #Rescale the image with the factor given before croping
config['scaling_factor'] = 1.3 #Rescaling factor jitter
config['split'] = 'test'
config['train'] = False


#Compute test accuracy
def eval_net(weights, test_iters):
    test_net = caffe.Net(build_net(config), weights, caffe.TEST)
    accuracy = 0

    t = time.time()

    for it in xrange(test_iters):
        print it
        accuracy += test_net.forward()['acc']

    elapsed = time.time() - t

    accuracy /= test_iters

    return test_net, accuracy, elapsed

#Print global accuracy
test_net, accuracy, elapsed = eval_net(weights, test_iters)
print 'Accuracy: %3.1f%%' % (100*accuracy, )
print 'Elapsed time: ' + str(elapsed)


