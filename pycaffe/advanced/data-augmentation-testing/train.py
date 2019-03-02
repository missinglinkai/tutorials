import caffe
from create_net import build_net
from create_solver import create_solver
from do_solve import do_solve
import os
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
missinglink_callback.set_properties(display_name="Data Augmentation Testing")

weights = 'bvlc_reference_caffenet.caffemodel'
#weights = '../../datasets/flickr_style/models/CNN/dataAugmentationNone-resizing_iter_200000.caffemodel'
assert os.path.exists(weights)

config = {}

id = 'dataAugmentation-None'

split_train = 'train'
split_val = 'valid'
config['dir'] = '102flowers'

config['mean'] = (104, 117, 123)
config['num_labels'] = 102
config['batch_size'] = 100 #40 #AlexNet 100, VGG 40
config['resize'] = True #Resize the image to the given size before cropping
config['resize_w'] = 224
config['resize_h'] = 224
config['cropping'] = False#True
config['crop_w'] = 224 #Train with a random crop of this size
config['crop_h'] = 224 #227 AlexNet, 224 VGG16, Inception
config['crop_margin'] = 0 #The crop won't include the margin in pixels
config['always_center_crop'] = False #The center crop of the image is always used (Used for testing)
config['mirror'] = False #True #Mirror images with 50% prob
config['rotate_prob'] = 0#0.2 #Rotation probability
config['rotate_angle'] = 10 #15,8 #Rotate with angle between -a and a
config['HSV_prob'] = 0#.2 #Jitter saturation and vale of the image with this prob
config['HSV_jitter'] = 0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter
config['color_casting_prob'] = 0#0.2  #Alterates each color channel (with the given prob for each channel) suming jitter
config['color_casting_jitter'] = 10 #Sum/substract 10 from the color channel
config['scaling_prob'] = 0#.5 #Rescale the image with the factor given before croping
config['scaling_factor'] = 1.3 #Rescaling factor jitter


#Create the net architecture
config['split'] = split_train
config['train'] = True
net_train = build_net(config)

#Prepare validation net. In validation (train=False) data augmentation is not active
config['always_center_crop'] = False
config['resize'] = True
config['split'] = split_val
config['train'] = False
net_val = build_net(config)

niter = 1000000
base_lr = 0.001
display_interval = 5 #150

#number of validating images  is  test_iters * batchSize
test_interval = 10
test_iters = 10 #150

#Set solver configuration
solver_filename = create_solver(net_train, net_val, base_lr, id)

#Load solver
solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, solver_filename)

missinglink_callback.set_expected_predictions_layers("label", "fc8C")

#Copy init weights
solver.net.copy_from(weights)

#Restore solverstate
#solver.restore('models/IIT5K/cifar10/IIT5K_iter_15000.caffemodel')


print 'Running solvers for %d iterations...' % niter
do_solve(niter, solver, display_interval, test_interval, test_iters, id)
print 'Done.'

