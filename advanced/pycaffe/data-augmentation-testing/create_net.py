#DEFINING AND RUNNING THE NET
import caffe
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]


boosted_weight_param = dict(lr_mult=10, decay_mult=10)
boosted_bias_param   = dict(lr_mult=20, decay_mult=0)
boosted_param = [boosted_weight_param, boosted_bias_param]


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def ave_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)

def build_net(config):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()

    pylayer = 'customDataLayer'

    n.data, n.label = L.Python(module='layers', layer=pylayer,
            ntop=2, param_str=str(config))

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=learned_param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=learned_param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=learned_param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=learned_param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=learned_param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=learned_param)  # 4096
    if config['train']:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=learned_param)  # 4096
    if config['train']:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5)
    else:
        fc8input = n.relu7

    n.fc8C = L.InnerProduct(fc8input, num_output=config['num_labels'], weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0.1), param=boosted_param)

    if not config['train']:
        n.probs = L.Softmax(n.fc8C)

    n.loss = L.SoftmaxWithLoss(n.fc8C, n.label)

    n.acc = L.Accuracy(n.fc8C, n.label)


    if config['train']:
        with open('train.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name
    else:
        with open('val.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name

